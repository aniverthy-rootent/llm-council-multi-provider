# LLM Council

**Framework:** Many providers over raw HTTP  
**Agents:** 3

Asks one question to models from different vendors in parallel, has them anonymously rank each other, then a Chair writes the final answer and an Auditor checks what the Chair glossed over.

> **Status:** this project was written as complete, runnable code but has **not been executed end to end against live LLM APIs** yet. It needs your own API key; expect to fix small issues from provider or library version changes. Issues and PRs welcome.

## Agents

| # | Agent | Role |
|---|---|---|
| 1 | Council Member 1-4 | four different vendors (default: Claude, GPT, Gemini, Llama via Groq) |
| 2 | Chair | synthesises using the peer rankings and critiques |
| 3 | Auditor | flags omitted disagreements |

## Run it

```bash
cp .env.example .env   # keys for at least two vendors
python main.py "Pay off a 4% mortgage early or invest?"
```

## How it works

Standard library only for the HTTP calls: Anthropic, OpenAI, Gemini, Mistral, Groq, DeepSeek, xAI, Together, OpenRouter, Fireworks, Cerebras, Perplexity, Cohere, Azure OpenAI, Ollama.

## Configuration

See `.env.example`. Models are chosen with one `LLM_SPEC` variable so you can switch provider without touching code.

## License

MIT
