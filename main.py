"""LLM Council - ask one question to models from different vendors, let them review each other, get one answer.

    python main.py "Is it better to pay off a 4% mortgage early or invest the money? Explain the trade-offs."
    python main.py "..." --members "openai:gpt-4o,mistral:mistral-large-latest,deepseek:deepseek-chat,ollama:llama3.2" \
                         --chair openai:gpt-4o

Agents: 4 council members (each a different vendor), the Chair and the Auditor. Members without API keys are skipped.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from council import council as c


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--members", help="comma list of provider:model (default from COUNCIL env or built-in)")
    ap.add_argument("--chair", default="anthropic:claude-sonnet-5-5", help="provider:model for Chair and Auditor")
    ap.add_argument("--save", type=Path, help="write the full transcript to this Markdown file")
    a = ap.parse_args()

    chair_provider, _, chair_model = a.chair.partition(":")
    print("Council session:")
    result = c.run(a.question, c.parse_members(a.members), (chair_provider, chair_model))

    print("\n" + "=" * 70 + "\nFINAL ANSWER\n" + "=" * 70 + "\n" + result["final"])
    audit = result["audit"]
    print("\nAUDIT:", f"verdict={audit.get('verdict')} confidence={audit.get('confidence')}")
    for item in audit.get("omitted_disagreements", []):
        print("  - omitted disagreement:", item)
    for item in audit.get("unsupported_claims", []):
        print("  - unsupported claim:", item)

    if a.save:
        parts = [f"# Council transcript\n\n**Question:** {a.question}\n"]
        for m in result["members"]:
            parts.append(f"## Member {m.label}: {m.name} ({m.seconds:.1f}s, peer score {result['points'][m.label]})\n{m.answer}\n")
        parts.append(f"## Final answer\n{result['final']}\n\n## Audit\n```json\n{audit}\n```\n")
        a.save.write_text("\n".join(parts), encoding="utf-8")
        print(f"\nTranscript saved to {a.save}")


if __name__ == "__main__":
    main()
