# tests/integration/test_graph.py
from app.graph.builder import build_graph
from app.llm.schemas import AnalystOutputSchema, CitationSchema, RiskSchema

from tests.mocks import MockLLM, MockRetriever


def _analyst_output(answer, citations, risks=None, insufficient=False):
    return AnalystOutputSchema(
        answer=answer,
        insufficient_evidence=insufficient,
        citations=[CitationSchema(chunk_id=cid, quote=q) for cid, q in citations],
        risks=[
            RiskSchema(
                title=t,
                severity=s,
                description=d,
                citation=CitationSchema(chunk_id=cid, quote=q),
            )
            for t, s, d, cid, q in (risks or [])
        ],
    )


def _invoke(graph, query="What is the termination notice period?", mode="auto"):
    return graph.invoke(
        {
            "analysis_id": "test-analysis",
            "document_id": "doc1",
            "query": query,
            "mode": mode,
        },
        config={"recursion_limit": 8},
    )


def _nodes(final):
    return [t["node"] for t in final.get("trace", [])]


# ---------------------------------------------------------------- 1. happy path


def test_graph_happy_path(mock_retriever):
    out = _analyst_output(
        answer="Either party may terminate upon sixty (60) days prior written notice.",
        citations=[("doc:0001:0001", "sixty (60) days prior written notice")],
    )
    llm = MockLLM(responses=[out])
    graph = build_graph(mock_retriever, llm)

    final = _invoke(graph)

    assert final["verification"]["passed"] is True
    assert final["verification"]["retries"] == 0
    assert final["insufficient_evidence"] is False
    assert final["warnings"] == []
    assert len(final["citations"]) == 1
    assert final["citations"][0]["verified"] is True
    assert final["draft_answer"].startswith("Either party")

    assert _nodes(final) == ["router", "retrieve", "analyst", "verifier", "finalize"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["schema"] == "AnalystOutputSchema"
    assert mock_retriever.calls == ["What is the termination notice period?"]


# ------------------------------------------------------------ 2. retry in verifier


def test_graph_retries_after_verification_failure(mock_retriever):
    bad = _analyst_output(
        answer="The party must pay immediately.",
        citations=[("doc:0001:0001", "the party shall pay immediately in gold")],  # fabricated
    )
    good = _analyst_output(
        answer="Either party may terminate upon sixty (60) days prior written notice.",
        citations=[("doc:0001:0001", "sixty (60) days prior written notice")],
    )
    llm = MockLLM(responses=[bad, good])
    graph = build_graph(mock_retriever, llm)

    final = _invoke(graph)

    assert final["verification"]["passed"] is True
    assert final["verification"]["retries"] == 1
    assert final["warnings"] == []

    assert _nodes(final) == [
        "router",
        "retrieve",
        "analyst",
        "verifier",
        "analyst",
        "verifier",
        "finalize",
    ]
    assert len(llm.calls) == 2

    # вторая попытка должна нести feedback от верификатора
    second_prompt = llm.calls[1]["prompt"]
    assert "PREVIOUS ATTEMPT FAILED VERIFICATION" in second_prompt
    assert "doc:0001:0001" in second_prompt


# ------------------------------------------------ 3. insufficient evidence (empty)


def test_graph_insufficient_evidence_when_retrieval_empty():
    empty = MockRetriever([])
    llm = MockLLM(responses=[])  # если Analyst вызовет LLM — будет ошибка
    graph = build_graph(empty, llm)

    final = _invoke(graph, query="What about flying saucers?")

    assert final["insufficient_evidence"] is True
    assert final["citations"] == []
    assert final["risks"] == []
    assert final["verification"]["passed"] is True
    assert any("Insufficient evidence" in w for w in final["warnings"])

    assert _nodes(final) == ["router", "retrieve", "analyst", "verifier", "finalize"]
    assert len(llm.calls) == 0  # LLM не вызывалась


# ------------------------------------------------------------------ 4. LLM error


def test_graph_handles_llm_error(mock_retriever):
    llm = MockLLM(raise_error=True)
    graph = build_graph(mock_retriever, llm)

    final = _invoke(graph)

    assert final["error"] is not None
    assert final["insufficient_evidence"] is True
    assert any("LLM error" in w for w in final["warnings"])
    assert final["draft_answer"] == "LLM unavailable."

    assert _nodes(final) == ["router", "retrieve", "analyst", "verifier", "finalize"]


# -------------------------------------------------------- 5. risk mode yields risks


def test_graph_risk_mode_returns_risks(mock_retriever):
    out = _analyst_output(
        answer="Several risks identified.",
        citations=[
            ("doc:0001:0003", "shall not exceed total fees paid in the preceding three (3) months"),
        ],
        risks=[
            (
                "Liability cap",
                "high",
                "Supplier liability is capped at 3 months of fees.",
                "doc:0001:0003",
                "shall not exceed total fees paid in the preceding three (3) months",
            ),
        ],
    )
    llm = MockLLM(responses=[out])
    graph = build_graph(mock_retriever, llm)

    final = _invoke(graph, query="Find all risks", mode="risk")

    assert final["mode"] == "risk"
    assert len(final["risks"]) == 1
    assert final["risks"][0]["severity"] == "high"
    assert final["risks"][0]["citation"]["verified"] is True
    assert final["verification"]["passed"] is True


# -------------------------------------------- 6. auto-mode routing via heuristic


def test_graph_auto_mode_routes_to_risk(mock_retriever):
    out = _analyst_output(
        answer="ok",
        citations=[("doc:0001:0001", "sixty (60) days prior written notice")],
    )
    llm = MockLLM(responses=[out])
    graph = build_graph(mock_retriever, llm)

    final = _invoke(graph, query="What are the liability risks?", mode="auto")

    assert final["mode"] == "risk"
