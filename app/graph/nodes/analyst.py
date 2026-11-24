# app/graph/nodes/analyst.py
import time

from app.graph.prompts import ANALYST_SYSTEM, build_analyst_prompt
from app.graph.state import AnalysisState, RetrievedChunkDict
from app.llm.client import LLMClient, LLMError
from app.llm.schemas import AnalystOutputSchema


def _page_of(retrieved: list[RetrievedChunkDict], chunk_id: str) -> int:
    for c in retrieved:
        if c["chunk_id"] == chunk_id:
            return c["page"]
    return -1


def make_analyst_node(llm: LLMClient):
    def analyst(state: AnalysisState) -> dict:
        t0 = time.monotonic()
        retrieved = state.get("retrieved", [])

        if not retrieved:
            trace = {
                "node": "analyst",
                "duration_ms": int((time.monotonic() - t0) * 1000),
                "llm_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "error": None,
                "model": None,
            }
            return {
                "draft_answer": "Insufficient evidence in the provided document.",
                "insufficient_evidence": True,
                "citations": [],
                "risks": [],
                "trace": [trace],
            }

        feedback = None
        if state.get("retry_count", 0) > 0:
            feedback = state.get("verification", {}).get("unsupported") or None

        language = state.get("language", "en")
        prompt = build_analyst_prompt(
            query=state["query"],
            mode=state["mode"],
            chunks=retrieved,
            language=language,
            feedback=feedback,
        )

        try:
            out, meta = llm.generate_json(
                prompt=prompt,
                schema=AnalystOutputSchema,
                system_instruction=ANALYST_SYSTEM[language],
            )
        except LLMError as exc:
            trace = {
                "node": "analyst",
                "duration_ms": int((time.monotonic() - t0) * 1000),
                "llm_calls": 1,
                "tokens_in": 0,
                "tokens_out": 0,
                "error": str(exc),
                "model": None,
            }
            return {
                "error": str(exc),
                "draft_answer": "LLM unavailable.",
                "insufficient_evidence": True,
                "trace": [trace],
            }

        citations = [
            {
                "chunk_id": c.chunk_id,
                "page": _page_of(retrieved, c.chunk_id),
                "quote": c.quote,
                "verified": False,
            }
            for c in out.citations
        ]
        risks = [
            {
                "title": r.title,
                "severity": r.severity,
                "description": r.description,
                "citation": {
                    "chunk_id": r.citation.chunk_id,
                    "page": _page_of(retrieved, r.citation.chunk_id),
                    "quote": r.citation.quote,
                    "verified": False,
                },
            }
            for r in out.risks
        ]

        trace = {
            "node": "analyst",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "llm_calls": 1,
            "tokens_in": meta.tokens_in,
            "tokens_out": meta.tokens_out,
            "error": None,
            "model": meta.model,
        }
        return {
            "draft_answer": out.answer,
            "insufficient_evidence": out.insufficient_evidence,
            "citations": citations,
            "risks": risks,
            "trace": [trace],
        }

    return analyst
