# Veritas

Veritas is a zero-cost, source-grounded PDF knowledge assistant. Phases 1-3 currently provide PDF extraction, hybrid retrieval, and cited JSON answer generation.

## Phase 1 checks

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## Phase 3 provider setup

No secret is required for the tests. For a live Hugging Face request, copy `.env.example` to `.env` and set `HF_TOKEN` locally. If automatic routing rejects the model, set `HF_PROVIDER` to the provider shown in the model's Hugging Face inference mapping, for example `featherless-ai`. For a fully offline fallback, install Ollama and pull the configured model, then use `LLM_PROVIDERS=ollama`. Gemini is optional through `GEMINI_API_KEY`. Provider order is controlled by `LLM_PROVIDERS`; failed providers are retried and then skipped.
