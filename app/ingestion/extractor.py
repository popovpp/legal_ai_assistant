# app/ingestion/extractor.py
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    page: int
    text: str


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [PageText(page=1, text=text)]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[PageText] = []
        for i, p in enumerate(reader.pages, start=1):
            pages.append(PageText(page=i, text=p.extract_text() or ""))
        return pages
    raise ValueError(f"Unsupported file type: {suffix}")
