# tests/integration/test_api_cache.py
import pytest
from app.llm.schemas import AnalystOutputSchema, CitationSchema
from app.rag.vector_store import ScoredChunk
from fastapi.testclient import TestClient

from tests.mocks import MockLLM, MockRetriever


@pytest.fixture
def client(monkeypatch):
    # подменяем lifespan-состояния напрямую
    from app.main import app

    chunks = [
        ScoredChunk(
            chunk_id="doc:0001:0001",
            page=1,
            score=0.9,
            text="sixty (60) days prior written notice",
        )
    ]
    mock_retriever = MockRetriever(chunks)
    out = AnalystOutputSchema(
        answer="sixty days",
        insufficient_evidence=False,
        citations=[CitationSchema(chunk_id="doc:0001:0001", quote="sixty (60) days prior written notice")],
        risks=[],
    )
    mock_llm = MockLLM(responses=[out])

    # вручную подкладываем state, минуя реальный lifespan
    app.state.retriever = mock_retriever
    app.state.llm = mock_llm
    from app.cache import InMemoryLRU

    app.state.cache = InMemoryLRU()

    # мокаем проверку существования документа в БД
    monkeypatch.setattr(
        "app.api.analyses.get_conn",
        lambda: _FakeConn(),
    )

    return TestClient(app), mock_llm


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        class R:
            def fetchone(self):
                return (1,)

            def fetchall(self):
                return []

        return R()

    def commit(self):
        pass


def test_cache_hit_on_second_call(client):
    c, mock_llm = client
    body = {"document_id": "doc1", "query": "notice?", "mode": "qa"}

    r1 = c.post("/analyses", json=body)
    assert r1.status_code == 200
    assert "cache_hit" not in r1.json()["warnings"]
    assert len(mock_llm.calls) == 1

    r2 = c.post("/analyses", json=body)
    assert r2.status_code == 200
    assert "cache_hit" in r2.json()["warnings"]
    assert len(mock_llm.calls) == 1  # LLM больше не вызывался
    assert r1.json()["analysis_id"] != r2.json()["analysis_id"]
