# app/graph/state.py
import operator
from typing import Annotated, Literal, TypedDict

Mode = Literal["qa", "summary", "risk"]


class RetrievedChunkDict(TypedDict):
    chunk_id: str
    page: int
    text: str
    score: float


class CitationDict(TypedDict):
    chunk_id: str
    page: int
    quote: str
    verified: bool


class RiskDict(TypedDict):
    title: str
    severity: str
    description: str
    citation: CitationDict


class TraceEventDict(TypedDict):
    node: str
    duration_ms: int
    llm_calls: int
    tokens_in: int
    tokens_out: int
    model: str | None
    error: str | None


class VerificationDict(TypedDict):
    passed: bool
    unsupported: list[str]
    retries: int


class AnalysisState(TypedDict, total=False):
    analysis_id: str
    document_id: str
    query: str
    mode: Mode
    retrieved: list[RetrievedChunkDict]
    draft_answer: str
    insufficient_evidence: bool
    risks: list[RiskDict]
    citations: list[CitationDict]
    verification: VerificationDict
    retry_count: int
    should_retry: bool
    warnings: list[str]
    trace: Annotated[list[TraceEventDict], operator.add]
    error: str | None
