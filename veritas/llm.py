"""Small provider adapters; only the answer-generation call may be remote."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen

from .config import Settings


class LLMProvider(Protocol):
    """Provider boundary keeps generation independent from any vendor SDK."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class HFProvider:
    """Use Hugging Face's free inference endpoint when a token is configured."""

    model: str
    token: str
    provider: str = "auto"

    def generate(self, prompt: str) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.model, provider=self.provider, token=self.token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=700, temperature=0.0
        )
        return response.choices[0].message.content or ""


@dataclass
class OllamaProvider:
    """Call a local Ollama server for an offline, unlimited fallback."""

    base_url: str
    model: str

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = Request(f"{self.base_url.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())["response"]


@dataclass
class GeminiProvider:
    """Use the Google AI Studio REST endpoint without adding a mandatory SDK."""

    model: str
    api_key: str

    def generate(self, prompt: str) -> str:
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode())
        return data["candidates"][0]["content"]["parts"][0]["text"]


@dataclass
class FallbackProvider:
    """Retry transient provider failures, then try configured providers in order."""

    providers: list[LLMProvider]
    attempts: int = 2
    base_delay: float = 1.0

    def generate(self, prompt: str) -> str:
        errors: list[str] = []
        for provider in self.providers:
            for attempt in range(self.attempts):
                try:
                    return provider.generate(prompt)
                except Exception as error:  # Provider SDKs expose different error types.
                    errors.append(f"{type(provider).__name__}: {error}")
                    if attempt + 1 < self.attempts:
                        time.sleep(self.base_delay * (2**attempt))
        raise RuntimeError("All configured LLM providers failed: " + " | ".join(errors))


def configured_provider(settings: Settings) -> LLMProvider:
    """Build configured providers in fallback order without exposing credentials."""
    providers: list[LLMProvider] = []
    provider_names = settings.llm_providers or (settings.llm_provider,)
    for name in provider_names:
        if name == "hf" and settings.hf_token:
            providers.append(HFProvider(settings.llm_model, settings.hf_token, settings.hf_provider))
        elif name == "gemini" and settings.gemini_api_key:
            providers.append(GeminiProvider(settings.llm_model, settings.gemini_api_key))
        elif name == "ollama":
            providers.append(OllamaProvider(settings.ollama_url, settings.llm_model))
    if not providers:
        raise RuntimeError("No LLM provider is configured. Set HF_TOKEN, GEMINI_API_KEY, or use Ollama.")
    return FallbackProvider(providers)
