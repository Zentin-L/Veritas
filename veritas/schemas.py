"""Small data objects shared by ingestion and chunking."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PageText:
    """A cleaned PDF page kept separate so page citations remain reliable."""

    doc_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A sentence-aligned passage with enough location data for citations."""

    id: str
    text: str
    doc_name: str
    page: int
    start_char: int
    end_char: int


class AnswerSentence(BaseModel):
    """A claim and its source IDs, validated before verification or display."""

    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    """The deliberately small JSON contract expected from every generation provider."""

    abstain: bool
    sentences: list[AnswerSentence]


class ClaimVerification(BaseModel):
    """Evidence scores for one answer claim, kept separate for UI badges."""

    text: str
    citations: list[str]
    entailment_probability: float
    label: str


class VerificationResult(BaseModel):
    """Aggregate trust and the safe fallback passages when claims are weak."""

    abstain: bool
    trust_score: float
    claims: list[ClaimVerification]
    source_passages: list[str] = Field(default_factory=list)
