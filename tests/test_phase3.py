from veritas.config import Settings
from veritas.generate import answer_question, extract_json
from veritas.retrieve import RetrievalResult
from veritas.schemas import Chunk, GeneratedAnswer


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return next(self.responses)


def result(chunk_id: str = "doc-p1-c1", score: float = 0.8) -> RetrievalResult:
    chunk = Chunk(chunk_id, "The document says the sky is blue.", "doc", 1, 0, 35)
    return RetrievalResult(chunk, score, 1.0, 0.03)


def test_extract_json_ignores_fence_and_trailing_text() -> None:
    parsed = extract_json('```json\n{"abstain": false, "sentences": []}\n``` trailing notes')
    assert parsed.abstain is False


def test_extract_json_rejects_invalid_json() -> None:
    try:
        extract_json("not json")
    except ValueError as error:
        assert "JSON" in str(error)
    else:
        raise AssertionError("invalid JSON should fail clearly")


def test_answer_filters_unknown_citations() -> None:
    provider = FakeProvider(['{"abstain": false, "sentences": [{"text": "Claim.", "citations": ["bad", "doc-p1-c1"]}]}'])
    answer = answer_question("What does it say?", [result()], provider)
    assert answer.sentences[0].citations == ["doc-p1-c1"]
    assert provider.calls == 1


def test_answer_retries_once_for_malformed_json() -> None:
    provider = FakeProvider(["not json", '{"abstain": false, "sentences": [{"text": "Claim.", "citations": ["doc-p1-c1"]}]}'])
    answer = answer_question("What does it say?", [result()], provider)
    assert isinstance(answer, GeneratedAnswer)
    assert provider.calls == 2


def test_confidence_gate_skips_provider() -> None:
    provider = FakeProvider([])
    answer = answer_question("Unknown?", [result(score=0.01)], provider, Settings(confidence_threshold=0.1))
    assert answer.abstain is True
    assert provider.calls == 0


def test_answer_abstains_when_provider_fails() -> None:
    class FailingProvider:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("provider unavailable")

    answer = answer_question("What does it say?", [result()], FailingProvider())
    assert answer.abstain is True
    assert answer.sentences == []
