# app/api/analyses.py
import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from app.cache import InMemoryLRU
from app.config import settings
from app.graph.builder import build_graph
from app.llm.client import LLMClient
from app.models.db import get_conn
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    Citation,
    Risk,
    TraceEvent,
    VerificationResult,
)
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyses", tags=["analyses"])


def _llm(request: Request) -> LLMClient:
    return request.app.state.llm


def _retriever(request: Request) -> Retriever:
    return request.app.state.retriever


@router.post("", response_model=AnalysisResponse)
def run_analysis(
    request: Request,
    payload: AnalysisRequest,
    llm: LLMClient = Depends(_llm),
    retriever: Retriever = Depends(_retriever),
) -> AnalysisResponse:

    cache: InMemoryLRU = request.app.state.cache
    cache_key = InMemoryLRU.make_key(
        payload.document_id,
        payload.query,
        payload.mode,
        retriever.index_version,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        # возвращаем кэш, но с новым analysis_id и признаком cache_hit
        cached.analysis_id = uuid.uuid4().hex
        cached.warnings = list(cached.warnings) + ["cache_hit"]
        return cached

    with get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM documents WHERE id = ?", (payload.document_id,)).fetchone()
    if not exists:
        raise HTTPException(404, "Document not found")

    t0 = time.monotonic()
    analysis_id = uuid.uuid4().hex

    graph = build_graph(retriever, llm)
    state_in = {
        "analysis_id": analysis_id,
        "document_id": payload.document_id,
        "query": payload.query,
        "mode": payload.mode,
    }

    try:
        final = graph.invoke(state_in, config={"recursion_limit": settings.recursion_limit})
    except Exception as exc:
        logger.exception("graph failed")
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    total_ms = int((time.monotonic() - t0) * 1000)
    llm_calls = sum(t.get("llm_calls", 0) for t in final.get("trace", []))

    mode = final.get("mode") or (payload.mode if payload.mode != "auto" else "qa")
    response = AnalysisResponse(
        analysis_id=analysis_id,
        document_id=payload.document_id,
        mode=mode,
        answer=final.get("draft_answer", ""),
        insufficient_evidence=bool(final.get("insufficient_evidence", False)),
        risks=[Risk(**r) for r in final.get("risks", [])],
        citations=[Citation(**c) for c in final.get("citations", [])],
        verification=VerificationResult(
            **(final.get("verification") or {"passed": True, "unsupported": [], "retries": 0})
        ),
        warnings=final.get("warnings", []),
        trace=[TraceEvent(**t) for t in final.get("trace", [])],
        llm_calls_total=llm_calls,
        total_duration_ms=total_ms,
        error=final.get("error"),
    )

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analyses ("
            "id, document_id, query, mode, status, result_json, error, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_id,
                payload.document_id,
                payload.query,
                response.mode,
                "error" if response.error else "ok",
                response.model_dump_json(),
                response.error,
                datetime.now(UTC).isoformat(),
            ),
        )

    if response.error is None:
        cache.put(cache_key, response.model_copy(deep=True))

    return response


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str) -> AnalysisResponse:
    with get_conn() as conn:
        row = conn.execute("SELECT result_json FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Analysis not found")
    return AnalysisResponse.model_validate_json(row["result_json"])
