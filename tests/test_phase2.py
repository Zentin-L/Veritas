import numpy as np

from veritas.config import Settings
from veritas.index import SearchIndex
from veritas.retrieve import reciprocal_rank_fusion, retrieve
from veritas.schemas import Chunk


def test_settings_from_env_reads_project_dotenv(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=ollama\nLLM_PROVIDERS=ollama\nLLM_MODEL=qwen2.5:0.5b\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings.llm_provider == "ollama"
    assert settings.llm_providers == ("ollama",)
    assert settings.llm_model == "qwen2.5:0.5b"


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.array([self.vectors[text] for text in texts], dtype="float32")

    def query(self, text: str) -> np.ndarray:
        return np.array(self.vectors[text], dtype="float32")


def chunks() -> list[Chunk]:
    return [
        Chunk("doc-p1-c1", "cats purr softly", "doc", 1, 0, 16),
        Chunk("doc-p1-c2", "dogs bark loudly", "doc", 1, 17, 34),
        Chunk("doc-p2-c1", "cats chase mice", "doc", 2, 0, 15),
    ]


def test_rrf_matches_hand_computed_order() -> None:
    result = reciprocal_rank_fusion([0, 1], [1, 2], k=60)
    assert result[0][0] == 1
    assert result[0][1] == (1 / 61) + (1 / 62)
    assert result[1][0] == 0
    assert result[2][0] == 2


def test_build_save_reload_and_hybrid_retrieve(tmp_path) -> None:
    items = chunks()
    vectors = np.eye(3, dtype="float32")
    built = SearchIndex.build(items, vectors, "abc123")
    built.save(tmp_path)
    loaded = SearchIndex.load(tmp_path, "abc123")
    assert [item.id for item in loaded.chunks] == [item.id for item in items]
    results = retrieve(
        loaded,
        "cats",
        lambda _: np.array([1, 0, 0], dtype="float32"),
        Settings(top_k=2, candidate_k=3),
    )
    assert len(results) == 2
    assert results[0].chunk.id in {"doc-p1-c1", "doc-p2-c1"}
    assert results[0].dense_score > 0
    assert results[1].dense_score >= 0


def test_load_rejects_stale_source_hash(tmp_path) -> None:
    built = SearchIndex.build(chunks(), np.eye(3, dtype="float32"), "current")
    built.save(tmp_path)
    try:
        SearchIndex.load(tmp_path, "old")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("stale indexes must be rejected")


def test_load_or_build_rebuilds_when_hash_changes(tmp_path) -> None:
    items = chunks()
    first = SearchIndex.load_or_build(tmp_path, items, np.eye(3, dtype="float32"), "first")
    replacement = [Chunk("new-p1-c1", "new content", "new", 1, 0, 11)]
    second = SearchIndex.load_or_build(
        tmp_path, replacement, np.array([[1, 0, 0]], dtype="float32"), "second"
    )
    assert first.source_hash == "first"
    assert second.source_hash == "second"
    assert second.chunks[0].id == "new-p1-c1"
