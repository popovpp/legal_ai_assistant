# app/api/documents.py
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_embedder, get_vector_store
from app.config import settings
from app.ingestion.embedder import Embedder
from app.ingestion.extractor import SUPPORTED_EXTENSIONS
from app.ingestion.service import ingest_document
from app.models.db import get_conn
from app.models.schemas import DocumentInfo, DocumentUploadResponse
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024

_MIME = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}


def _mime(suffix: str) -> str:
    return _MIME.get(suffix, "application/octet-stream")


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {suffix}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large: {len(data)} > {MAX_FILE_SIZE}")

    tmp_path = settings.uploads_dir / f"_tmp_{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(data)

    try:
        result = ingest_document(
            source_path=tmp_path,
            filename=file.filename or tmp_path.name,
            mime=_mime(suffix),
            embedder=embedder,
            store=store,
        )
    except Exception as exc:
        logger.exception("ingestion failed")
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc

    final_path = settings.uploads_dir / f"{result['document_id']}{suffix}"
    shutil.move(str(tmp_path), str(final_path))

    return DocumentUploadResponse(**result)


@router.get("", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, mime, num_pages, num_chunks, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [
        DocumentInfo(
            document_id=r["id"],
            filename=r["filename"],
            mime=r["mime"],
            num_pages=r["num_pages"],
            num_chunks=r["num_chunks"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
