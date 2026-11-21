# app/rag/vector_store.py
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk_id: str
    page: int
    text: str
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]: ...
    def persist(self) -> None: ...
    def load(self) -> None: ...
    def __len__(self) -> int: ...


class FaissVectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "meta.json"

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self._index: faiss.Index | None = None
        self._ids: list[str] = []
        self._meta: dict[str, dict] = {}

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    def _ensure_index(self, dim: int) -> faiss.Index:
        index = self._index
        if index is None:
            index = faiss.IndexFlatIP(dim)
            self._index = index
        return index

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunks={len(chunks)} != vectors={vectors.shape[0]}")
        vecs = vectors.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        index = self._ensure_index(vecs.shape[1])
        if vecs.shape[1] != index.d:
            raise ValueError(f"vector dim {vecs.shape[1]} != index dim {index.d}")
        index.add(vecs)
        for c in chunks:
            self._ids.append(c.chunk_id)
            self._meta[c.chunk_id] = {"page": c.page, "text": c.text}

    def search(self, vector: np.ndarray, k: int) -> list[ScoredChunk]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = vector.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))
        out: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            cid = self._ids[int(idx)]
            meta = self._meta[cid]
            out.append(
                ScoredChunk(
                    chunk_id=cid,
                    page=int(meta["page"]),
                    text=meta["text"],
                    score=float(score),
                )
            )
        return out

    def persist(self) -> None:
        if self._index is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
        (self.index_dir / self.META_FILE).write_text(
            json.dumps({"ids": self._ids, "meta": self._meta}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> None:
        idx_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return
        index = faiss.read_index(str(idx_path))
        self._index = index
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self._ids = data["ids"]
        self._meta = data["meta"]
        logger.info("loaded faiss index: %d vectors, dim=%d", index.ntotal, index.d)
