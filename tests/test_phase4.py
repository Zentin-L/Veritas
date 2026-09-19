import math

from veritas.config import Settings
from veritas.retrieve import RetrievalResult
from veritas.schemas import AnswerSentence, Chunk, GeneratedAnswer
from veritas.verify import _softmax, _windows, verify_answer


class FakeVerifier:
    def __init__(self, logits):
        self.logits_value = logits
        self.calls = []

    def logits(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.logits_value


def source(chunk_id="doc-p1-c1"):
    chunk = Chunk(chunk_id, "The sky is blue. Water is wet. Clouds are white.", "doc", 1, 0, 48)
    return RetrievalResult(chunk, 0.8, 1.0, 0.03)


def test_softmax_uses_entailment_at_index_one():
    probabilities = _softmax([0.0, 2.0, 0.0])
    assert probabilities[1] > probabilities[0]
    assert math.isclose(sum(probabilities), 1.0)


def test_windows_overlap_sentences():
    windows = _windows("One. Two. Three. Four.", size=3, overlap=1)
    assert windows == ["One. Two. Three.", "Three. Four."]


def test_entailment_is_supported_and_calls_all_windows():
    verifier = FakeVerifier([-3.0, 5.0, -2.0])
    answer = GeneratedAnswer(abstain=False, sentences=[AnswerSentence(text="The sky is blue.", citations=["doc-p1-c1"])])
    result = verify_answer(answer, [source()], verifier, Settings(trust_threshold=0.7))
    assert result.claims[0].label == "supported"
    assert result.trust_score == 1.0
    assert len(verifier.calls) == 2


def test_contradiction_abstains_with_source_passage():
    verifier = FakeVerifier([5.0, -3.0, -2.0])
    answer = GeneratedAnswer(abstain=False, sentences=[AnswerSentence(text="The sky is green.", citations=["doc-p1-c1"])])
    result = verify_answer(answer, [source()], verifier, Settings(trust_threshold=0.7, top_k=1))
    assert result.claims[0].label == "unsupported"
    assert result.abstain is True
    assert result.source_passages