"""The council: 4 members (different vendors) -> anonymous peer review -> Chair -> Auditor  (6 agents)."""
from __future__ import annotations

import json
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import providers

DEFAULT_MEMBERS = ("anthropic:claude-sonnet-5-5,openai:gpt-4o,gemini:gemini-2.5-flash,"
                   "groq:llama-3.3-70b-versatile")
MEMBER_SYSTEM = ("You are a member of an expert council answering a hard question. Give your best, well-reasoned "
                 "answer in under 300 words. State key assumptions and your confidence (low/medium/high).")
REVIEW_SYSTEM = ("You are reviewing anonymous answers from other council members. Judge accuracy, reasoning and "
                 "completeness only. Reply with ONLY JSON: {\"ranking\": [labels best-to-worst], "
                 "\"critiques\": {label: one-sentence critique}}.")
CHAIR_SYSTEM = ("You are the Chair. Write the council's final answer: merge the strongest points, resolve conflicts "
                "using the peer rankings and critiques, and explicitly mention any point where members genuinely "
                "disagree. Under 350 words. Do not mention that you are an AI council.")
AUDITOR_SYSTEM = ("You are the Auditor. Compare the Chair's final answer with the members' original answers. Reply "
                  "with ONLY JSON: {\"omitted_disagreements\": [str], \"unsupported_claims\": [str], "
                  "\"confidence\": \"low|medium|high\", \"verdict\": \"OK|REVISE\"}.")


@dataclass
class Member:
    provider: str
    model: str
    label: str = ""
    answer: str = ""
    seconds: float = 0.0
    error: str = ""
    ranking: list[str] = field(default_factory=list)
    critiques: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_members(spec: str | None = None) -> list[Member]:
    spec = spec or os.environ.get("COUNCIL", DEFAULT_MEMBERS)
    out = []
    for part in spec.split(","):
        provider, _, model = part.strip().partition(":")
        out.append(Member(provider, model))
    return out


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def run(question: str, members: list[Member], chair: tuple[str, str], log=print) -> dict:
    # --- stage 1: members answer in parallel (members without credentials are skipped) ---
    live = [m for m in members if providers.available(m.provider)]
    for m in members:
        if m not in live:
            log(f"  skipping {m.name}: credentials not configured")
    if len(live) < 2:
        raise SystemExit("Need at least 2 council members with credentials configured (see .env.example).")

    def answer(m: Member) -> None:
        try:
            m.answer, m.seconds = providers.call(m.provider, m.model, MEMBER_SYSTEM, question)
        except providers.ProviderError as exc:
            m.error = str(exc)

    with ThreadPoolExecutor(max_workers=len(live)) as pool:
        list(pool.map(answer, live))
    live = [m for m in live if m.answer]
    for m in members:
        if m.error:
            log(f"  {m.name} failed: {m.error}")
    if len(live) < 2:
        raise SystemExit("Fewer than 2 members answered.")
    for m, label in zip(live, string.ascii_uppercase):
        m.label = label
        log(f"  [{label}] {m.name} answered in {m.seconds:.1f}s")

    # --- stage 2: anonymous peer review ---
    def review(m: Member) -> None:
        others = "\n\n".join(f"### Answer {o.label}\n{o.answer}" for o in live if o is not m)
        try:
            text, _ = providers.call(m.provider, m.model, REVIEW_SYSTEM, f"Question: {question}\n\n{others}")
        except providers.ProviderError:
            return
        data = _json(text)
        valid = {o.label for o in live if o is not m}
        m.ranking = [l for l in data.get("ranking", []) if l in valid]
        m.critiques = {k: v for k, v in data.get("critiques", {}).items() if k in valid}

    with ThreadPoolExecutor(max_workers=len(live)) as pool:
        list(pool.map(review, live))
    points: dict[str, int] = {m.label: 0 for m in live}          # Borda count over peer rankings
    for m in live:
        for pos, label in enumerate(m.ranking):
            points[label] += len(m.ranking) - pos
    ranking = sorted(points, key=lambda k: -points[k])
    log(f"  peer ranking (best first): {' > '.join(f'{l}({points[l]})' for l in ranking)}")

    # --- stage 3: chair ---
    bundle = "\n\n".join(f"### Answer {m.label} (peer score {points[m.label]})\n{m.answer}" for m in live)
    critiques = "\n".join(f"- reviewer {m.label} on {k}: {v}" for m in live for k, v in m.critiques.items())
    final, _ = providers.call(chair[0], chair[1], CHAIR_SYSTEM,
                              f"Question: {question}\n\n{bundle}\n\nPeer critiques:\n{critiques}", max_tokens=1800)

    # --- stage 4: auditor ---
    audit_text, _ = providers.call(chair[0], chair[1], AUDITOR_SYSTEM,
                                   f"Question: {question}\n\nMember answers:\n{bundle}\n\nChair final answer:\n{final}")
    return {"members": live, "ranking": ranking, "points": points, "final": final, "audit": _json(audit_text)}
