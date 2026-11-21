# app/graph/nodes/router.py
import re
import time

from app.graph.state import AnalysisState

_RISK_RE = re.compile(
    r"\b(risk|liabilit|breach|penalt|indemnif|terminat|dispute|governing|confidential)",
    re.I,
)
_SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|overview|brief|describe)\b", re.I)


def route_node(state: AnalysisState) -> dict:
    t0 = time.monotonic()
    requested = state.get("mode", "auto")
    if requested in ("qa", "summary", "risk"):
        mode = requested
    elif _SUMMARY_RE.search(state["query"]):
        mode = "summary"
    elif _RISK_RE.search(state["query"]):
        mode = "risk"
    else:
        mode = "qa"
    trace = {
        "node": "router",
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "llm_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "error": None,
        "model": None,
    }
    return {"mode": mode, "trace": [trace]}
