from typing import Protocol

from app.ingestion.embedder import Embedder
from app.rag.vector_store import ScoredChunk, VectorStore


class RetrieverLike(Protocol):
    @property
    def index_version(self) -> int: ...

    def retrieve(self, query: str, k: int = 6) -> list[ScoredChunk]: ...


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, min_score: float = 0.30):
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    @property
    def index_version(self) -> int:
        return len(self.store)

    def retrieve(self, query: str, k: int = 6) -> list[ScoredChunk]:
        q = self.embedder.encode_one(query)
        candidates = self.store.search(q, k=k)
        return [c for c in candidates if c.score >= self.min_score]
