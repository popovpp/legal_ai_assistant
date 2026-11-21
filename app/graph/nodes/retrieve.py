# app/graph/nodes/retrieve.py
import time

from app.graph.state import AnalysisState


def make_retrieve_node(retriever, top_k: int):
    def retrieve(state: AnalysisState) -> dict:
        t0 = time.monotonic()
        chunks = retriever.retrieve(state["query"], k=top_k)
        retrieved = [{"chunk_id": c.chunk_id, "page": c.page, "text": c.text, "score": c.score} for c in chunks]
        trace = {
            "node": "retrieve",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "llm_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "error": None,
            "model": None,
        }
        return {"retrieved": retrieved, "trace": [trace]}

    return retrieve
