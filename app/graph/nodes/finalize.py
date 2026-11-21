# app/graph/nodes/finalize.py
import time

from app.graph.state import AnalysisState


def finalize_node(state: AnalysisState) -> dict:
    t0 = time.monotonic()
    warnings: list[str] = []
    verif = state.get("verification") or {"passed": True, "unsupported": [], "retries": 0}
    if not verif.get("passed"):
        warnings.append(f"Unverified citations: {verif.get('unsupported')}")
    if state.get("insufficient_evidence"):
        warnings.append("Insufficient evidence in retrieved chunks.")
    if state.get("error"):
        warnings.append(f"LLM error: {state['error']}")

    trace = {
        "node": "finalize",
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "llm_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "error": None,
        "model": None,
    }
    return {"warnings": warnings, "trace": [trace]}
