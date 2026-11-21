# tests/unit/test_verifier.py
from app.graph.nodes.verifier import make_verifier_node


def _state(retrieved, citations, risks=None, retry_count=0):
    return {
        "analysis_id": "a1",
        "document_id": "d1",
        "query": "q",
        "mode": "qa",
        "retrieved": [{"chunk_id": c.chunk_id, "page": c.page, "text": c.text, "score": c.score} for c in retrieved],
        "citations": citations,
        "risks": risks or [],
        "retry_count": retry_count,
    }


def _cit(chunk_id, quote):
    return {"chunk_id": chunk_id, "page": 1, "quote": quote, "verified": False}


def test_verifier_passes_exact_substring(retrieved_chunks):
    node = make_verifier_node()
    citations = [_cit("doc:0001:0001", "sixty (60) days prior written notice")]
    out = node(_state(retrieved_chunks, citations))
    assert out["verification"]["passed"] is True
    assert out["should_retry"] is False
    assert out["citations"][0]["verified"] is True


def test_verifier_passes_normalized_whitespace(retrieved_chunks):
    node = make_verifier_node()
    # в чанке переносы строк, в цитате — пробелы
    citations = [_cit("doc:0001:0001", "Either  party  may  terminate  for  convenience")]
    out = node(_state(retrieved_chunks, citations))
    assert out["citations"][0]["verified"] is True


def test_verifier_rejects_nonexistent_chunk_id(retrieved_chunks):
    node = make_verifier_node()
    citations = [_cit("doc:9999:9999", "anything")]
    out = node(_state(retrieved_chunks, citations))
    assert out["verification"]["passed"] is False
    assert "doc:9999:9999" in out["verification"]["unsupported"]
    assert out["should_retry"] is True  # retry_count=0 < max=1


def test_verifier_rejects_fabricated_quote(retrieved_chunks):
    node = make_verifier_node()
    citations = [_cit("doc:0001:0001", "the party shall pay ten thousand dollars immediately")]
    out = node(_state(retrieved_chunks, citations))
    assert out["citations"][0]["verified"] is False
    assert out["should_retry"] is True


def test_verifier_accepts_high_jaccard_paraphrase(retrieved_chunks):
    node = make_verifier_node()
    # перефраз той же фразы — Jaccard должен быть >= 0.75
    citations = [
        _cit(
            "doc:0001:0001",
            "Either party may terminate for convenience upon sixty days prior written notice",
        )
    ]
    out = node(_state(retrieved_chunks, citations))
    assert out["citations"][0]["verified"] is True


def test_verifier_does_not_retry_beyond_limit(retrieved_chunks):
    node = make_verifier_node()
    citations = [_cit("doc:0001:0001", "completely fabricated text here")]
    out = node(_state(retrieved_chunks, citations, retry_count=1))
    assert out["verification"]["passed"] is False
    assert out["should_retry"] is False
    assert out["verification"]["retries"] == 1


def test_verifier_checks_risk_citations_too(retrieved_chunks):
    node = make_verifier_node()
    risks = [
        {
            "title": "X",
            "severity": "high",
            "description": "d",
            "citation": _cit("doc:0001:9999", "fabricated"),
        }
    ]
    out = node(_state(retrieved_chunks, citations=[], risks=risks))
    assert out["risks"][0]["citation"]["verified"] is False
    assert "doc:0001:9999" in out["verification"]["unsupported"]
