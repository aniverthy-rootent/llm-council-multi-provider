"""Call many LLM vendors over plain HTTPS (stdlib only). One function: call(provider, model, system, user)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# provider -> (url, api-key env var, request style)
PROVIDERS = {
    "anthropic": ("https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY", "anthropic"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "openai"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", "GEMINI_API_KEY", "gemini"),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY", "openai"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "openai"),
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY", "openai"),
    "xai": ("https://api.x.ai/v1/chat/completions", "XAI_API_KEY", "openai"),
    "together": ("https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY", "openai"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "openai"),
    "fireworks": ("https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY", "openai"),
    "cerebras": ("https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY", "openai"),
    "perplexity": ("https://api.perplexity.ai/chat/completions", "PERPLEXITY_API_KEY", "openai"),
    "cohere": ("https://api.cohere.com/v2/chat", "COHERE_API_KEY", "cohere"),
    "azure": ("{endpoint}/openai/deployments/{model}/chat/completions?api-version={version}", "AZURE_OPENAI_API_KEY", "azure"),
    "ollama": ("http://localhost:11434/api/chat", None, "ollama"),
}


class ProviderError(RuntimeError):
    pass


def available(provider: str) -> bool:
    if provider not in PROVIDERS:
        return False
    env = PROVIDERS[provider][1]
    return env is None or bool(os.environ.get(env))


def _post(url: str, headers: dict, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"network error: {exc.reason}") from exc


def call(provider: str, model: str, system: str, user: str, max_tokens: int = 1500) -> tuple[str, float]:
    """Returns (text, seconds)."""
    if provider not in PROVIDERS:
        raise ProviderError(f"unknown provider {provider}")
    url, env, style = PROVIDERS[provider]
    key = os.environ.get(env, "") if env else ""
    if env and not key:
        raise ProviderError(f"{env} is not set")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.time()
    if style == "anthropic":
        d = _post(url, {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  {"model": model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]})
        text = "".join(b.get("text", "") for b in d["content"])
    elif style == "gemini":
        d = _post(url.format(model=model) + f"?key={key}", {},
                  {"system_instruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"maxOutputTokens": max_tokens}})
        text = d["candidates"][0]["content"]["parts"][0]["text"]
    elif style == "cohere":
        d = _post(url, {"Authorization": f"Bearer {key}"}, {"model": model, "messages": msgs})
        text = d["message"]["content"][0]["text"]
    elif style == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        d = _post(host + "/api/chat", {}, {"model": model, "stream": False, "messages": msgs})
        text = d["message"]["content"]
    elif style == "azure":
        full = url.format(endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"), model=model,
                          version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        d = _post(full, {"api-key": key}, {"messages": msgs, "max_tokens": max_tokens})
        text = d["choices"][0]["message"]["content"]
    else:  # openai-compatible
        d = _post(url, {"Authorization": f"Bearer {key}"}, {"model": model, "messages": msgs, "max_tokens": max_tokens})
        text = d["choices"][0]["message"]["content"]
    return text, time.time() - t0
