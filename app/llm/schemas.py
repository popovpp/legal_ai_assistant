# app/llm/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class CitationSchema(BaseModel):
    chunk_id: str = Field(description="ID фрагмента из контекста, например 'chunk_0007'")
    quote: str = Field(description="Дословная цитата из этого фрагмента")


class RiskSchema(BaseModel):
    title: str = Field(description="Краткое название риска")
    severity: Literal["low", "medium", "high"]
    description: str = Field(description="Описание риска и его последствий")
    citation: CitationSchema


class AnalystOutputSchema(BaseModel):
    answer: str = Field(description="Ответ на вопрос или резюме документа")
    insufficient_evidence: bool = Field(
        default=False,
        description="True, если в предоставленных фрагментах нет достаточных оснований для ответа",
    )
    citations: list[CitationSchema] = Field(default_factory=list)
    risks: list[RiskSchema] = Field(default_factory=list)


class RouterOutputSchema(BaseModel):
    mode: Literal["qa", "summary", "risk"]
