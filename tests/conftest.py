# tests/conftest.py
import pytest

from tests.mocks import MockLLM, MockRetriever  # noqa: F401 — реэкспорт для удобства


@pytest.fixture
def retrieved_chunks():
    from app.rag.vector_store import ScoredChunk

    return [
        ScoredChunk(
            chunk_id="doc:0001:0001",
            page=1,
            score=0.72,
            text=("Either party may terminate for convenience upon sixty (60) days prior written notice."),
        ),
        ScoredChunk(
            chunk_id="doc:0001:0003",
            page=1,
            score=0.68,
            text=(
                "The Supplier's aggregate liability shall not exceed total fees paid in the preceding three (3) months."
            ),
        ),
        ScoredChunk(
            chunk_id="doc:0001:0005",
            page=1,
            score=0.55,
            text="All intellectual property rights in the deliverables shall vest in the Supplier.",
        ),
    ]


@pytest.fixture
def mock_retriever(retrieved_chunks):
    return MockRetriever(retrieved_chunks)
