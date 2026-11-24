from collections.abc import Mapping
from typing import Protocol

from app.ingestion.embedder import Embedder
from app.rag.vector_store import ScoredChunk, VectorStore


class RetrieverLike(Protocol):
    def retrieve(self, language: str, query: str, k: int = 6) -> list[ScoredChunk]: ...


class RetrieverRegistry:
    def __init__(
        self,
        embedders: Mapping[str, Embedder],
        stores: Mapping[str, VectorStore],
        min_score: float,
    ):
        self._embedders = embedders
        self._stores = stores
        self.min_score = min_score

    def get(self, language: str) -> tuple[Embedder, VectorStore]:
        if language not in self._embedders:
            raise ValueError(f"Unsupported language: {language}")
        return self._embedders[language], self._stores[language]

    @property
    def index_version(self) -> dict[str, int]:
        return {lang: len(store) for lang, store in self._stores.items()}

    def retrieve(self, language: str, query: str, k: int = 6) -> list[ScoredChunk]:
        embedder, store = self.get(language)
        q = embedder.encode_one(query)
        candidates = store.search(q, k=k)
        return [c for c in candidates if c.score >= self.min_score]
