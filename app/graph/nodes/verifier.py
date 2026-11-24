# app/graph/nodes/verifier.py
import re
import time

from app.config import settings
from app.graph.state import AnalysisState

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_verifier_node():
    def verifier(state: AnalysisState) -> dict:
        t0 = time.monotonic()
        by_id = {c["chunk_id"]: c for c in state.get("retrieved", [])}
        citations = list(state.get("citations", []))
        risks = list(state.get("risks", []))
        unsupported: list[str] = []

        def check(cit) -> bool:
            cid = cit["chunk_id"]
            if cid not in by_id:
                return False
            chunk_text = by_id[cid]["text"]
            if _norm(cit["quote"]) in _norm(chunk_text):
                return True
            return _jaccard(_tokens(cit["quote"]), _tokens(chunk_text)) >= 0.75

        for i, cit in enumerate(citations):
            ok = check(cit)
            citations[i] = {**cit, "verified": ok}
            if not ok:
                unsupported.append(cit["chunk_id"])

        for i, r in enumerate(risks):
            ok = check(r["citation"])
            risks[i] = {**r, "citation": {**r["citation"], "verified": ok}}
            if not ok:
                unsupported.append(r["citation"]["chunk_id"])

        passed = not unsupported
        retry_count = state.get("retry_count", 0)
        should_retry = (not passed) and retry_count < settings.max_verifier_retries

        verification = {"passed": passed, "unsupported": unsupported, "retries": retry_count}
        trace = {
            "node": "verifier",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "llm_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "error": None,
            "model": None,
        }
        result = {
            "citations": citations,
            "risks": risks,
            "verification": verification,
            "should_retry": should_retry,
            "trace": [trace],
        }
        if should_retry:
            result["retry_count"] = retry_count + 1
        return result

    return verifier
