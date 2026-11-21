# app/models/schemas.py
from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["qa", "summary", "risk"]


class Citation(BaseModel):
    chunk_id: str
    page: int
    quote: str
    verified: bool


class Risk(BaseModel):
    title: str
    severity: Literal["low", "medium", "high"]
    description: str
    citation: Citation


class TraceEvent(BaseModel):
    node: str
    duration_ms: int
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str | None = None
    error: str | None = None


class VerificationResult(BaseModel):
    passed: bool
    unsupported: list[str] = Field(default_factory=list)
    retries: int = 0


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    mime: str
    num_pages: int
    num_chunks: int
    created_at: str


class AnalysisRequest(BaseModel):
    document_id: str
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["auto", "qa", "summary", "risk"] = "auto"


class AnalysisResponse(BaseModel):
    analysis_id: str
    document_id: str
    mode: Mode
    answer: str
    insufficient_evidence: bool
    risks: list[Risk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verification: VerificationResult
    warnings: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    llm_calls_total: int = 0
    total_duration_ms: int = 0
    error: str | None = None
