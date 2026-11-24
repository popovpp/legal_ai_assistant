# app/ingestion/service.py
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import Embedder
from app.ingestion.extractor import extract
from app.ingestion.language import detect_language
from app.models.db import get_conn
from app.rag.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def ingest_document(
    source_path: Path,
    filename: str,
    mime: str,
    embedders: dict[str, Embedder],
    stores: dict[str, FaissVectorStore],
) -> dict:
    document_id = uuid.uuid4().hex

    pages = extract(source_path)
    if not pages:
        raise ValueError("Document has no pages")

    full_text = "\n".join(p.text for p in pages)
    language = detect_language(full_text)

    if language not in embedders or language not in stores:
        raise ValueError(f"No embedder/store configured for language '{language}'")

    embedder = embedders[language]
    store = stores[language]

    chunks = chunk_pages(
        document_id=document_id,
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("Document produced no chunks (empty or unreadable)")

    logger.info(
        "ingest document_id=%s file=%s language=%s pages=%d chunks=%d",
        document_id,
        filename,
        language,
        len(pages),
        len(chunks),
    )

    vectors = embedder.encode([c.text for c in chunks])

    created_at = datetime.now(UTC).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, mime, language, num_pages, num_chunks, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (document_id, filename, mime, language, len(pages), len(chunks), created_at),
        )
        conn.executemany(
            "INSERT INTO chunks (id, document_id, page, char_start, char_end, text) VALUES (?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.document_id, c.page, c.char_start, c.char_end, c.text) for c in chunks],
        )

    store.add(chunks, vectors)
    store.persist()

    return {
        "document_id": document_id,
        "filename": filename,
        "language": language,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }
