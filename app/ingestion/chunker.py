# app/ingestion/chunker.py
import re

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.extractor import PageText


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page: int
    char_start: int
    char_end: int
    text: str


_SECTION_RE = re.compile(r"(?m)^(?=\d+\.\s)")


def _split_by_sections(text: str) -> list[tuple[int, str]]:
    """Split on numbered section headers like '1. ', '2. '.
    Returns (char_start, section_text) pairs."""
    starts = [m.start() for m in _SECTION_RE.finditer(text)]
    if not starts:
        return [(0, text)]
    parts: list[tuple[int, str]] = []
    if starts[0] > 0:
        parts.append((0, text[: starts[0]]))
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        parts.append((start, text[start:end]))
    return parts


def chunk_pages(
    document_id: str,
    pages: list[PageText],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    max_section_size: int = 1200,
) -> list[Chunk]:
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for offset, section in _split_by_sections(page.text):
            if len(section) <= max_section_size:
                pieces = [(offset, section)]
            else:
                pieces = [
                    (offset + int(d.metadata.get("start_index", 0)), d.page_content)
                    for d in fallback.create_documents([section])
                ]
            for i, (start, text) in enumerate(pieces):
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{page.page:04d}:{len(chunks):04d}",
                        document_id=document_id,
                        page=page.page,
                        char_start=start,
                        char_end=start + len(text),
                        text=text,
                    )
                )
    return chunks
