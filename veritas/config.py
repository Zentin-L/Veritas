"""Central defaults for retrieval and grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _load_dotenv_file(path: Path | None = None) -> dict[str, str]:
    """Read the repo-level .env file without requiring an extra dependency."""
    target = path or Path.cwd() / ".env"
    if not target.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    """Keep retrieval knobs explicit so behavior is easy to explain and tune."""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    context_token_budget: int = 2000
    confidence_threshold: float = 0.15
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    supported_threshold: float = 0.70
    partial_threshold: float = 0.30
    trust_threshold: float = 0.70
    llm_provider: str = "hf"
    llm_providers: tuple[str, ...] = ("hf", "ollama")
    llm_model: str = "Qwen/Qwen3-0.6B"
    hf_provider: str = "auto"
    hf_token: str | None = None
    gemini_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Settings":
        """Read deployment-specific values from the project .env file and OS environment."""
        dotenv_values = _load_dotenv_file()
        merged = {**dotenv_values, **os.environ}
        providers = merged.get("LLM_PROVIDERS", "hf,ollama")
        return cls(
            llm_provider=merged.get("LLM_PROVIDER", "hf"),
            llm_providers=tuple(
                item.strip() for item in providers.split(",") if item.strip()
            ),
            llm_model=merged.get("LLM_MODEL", "Qwen/Qwen3-0.6B"),
            hf_provider=merged.get("HF_PROVIDER", "auto"),
            hf_token=merged.get("HF_TOKEN"),
            gemini_api_key=merged.get("GEMINI_API_KEY"),
            ollama_url=merged.get("OLLAMA_URL", "http://localhost:11434"),
        )
