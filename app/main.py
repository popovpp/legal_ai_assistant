# app/main.py (полный файл)
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import analyses, documents
from app.cache import InMemoryLRU
from app.config import settings
from app.ingestion.embedder import Embedder
from app.llm.client import LLMClient
from app.models.db import init_db
from app.observability.middleware import RequestIDFilter, RequestIDMiddleware
from app.rag.retriever import RetrieverRegistry
from app.rag.vector_store import FaissVectorStore

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s","request_id":"%(request_id)s",'
        '"logger":"%(name)s","msg":"%(message)s"}'
    )
)
handler.addFilter(RequestIDFilter())

root = logging.getLogger()
root.handlers = [handler]
root.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()

    embedders = {
        "en": Embedder(settings.embed_model_en),
        "ru": Embedder(settings.embed_model_ru),
    }
    stores = {
        "en": FaissVectorStore(settings.index_dir_en),
        "ru": FaissVectorStore(settings.index_dir_ru),
    }
    for s in stores.values():
        s.load()

    retriever = RetrieverRegistry(embedders, stores, min_score=settings.min_score)
    llm = LLMClient()

    app.state.embedders = embedders
    app.state.stores = stores
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.cache = InMemoryLRU(max_size=256)

    logging.getLogger(__name__).info(
        "startup ok vectors=%s models=%s",
        {lang: len(s) for lang, s in stores.items()},
        settings.llm_model_chain,
    )
    yield


tags_metadata = [
    {
        "name": "documents",
        "description": "Загрузка и список юридических документов. "
        "Язык определяется автоматически по тексту (кириллица → ru, латиница → en).",
    },
    {
        "name": "analyses",
        "description": "Анализ документа: QA, summary, risk. "
        "Параметр `language` переключает ветку (en/ru) — промпты и индекс.",
    },
    {"name": "health", "description": "Состояние сервиса и статистика индексов."},
]


app = FastAPI(
    lifespan=lifespan,
    title="Legal AI Assistant",
    version="0.2.0",
    description=(
        "Multi-agent pipeline for legal document analysis (EN/RU).\n\n"
        "**Architecture**: FastAPI → LangGraph (Router → Retrieve → Analyst → Verifier) "
        "→ Gemini with model fallback.\n\n"
        "**Languages**: `en` uses `all-MiniLM-L6-v2`; `ru` uses `Roflmax/bge-m3-russian-legal`. "
        "Indexes are separate — vectors from different models are not comparable."
    ),
    openapi_tags=tags_metadata,
    contact={"name": "Legal AI Assistant", "url": "https://example.local"},
)
app.add_middleware(RequestIDMiddleware)
app.include_router(documents.router)
app.include_router(analyses.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(settings.data_dir),
        "llm_models": settings.llm_model_chain,
        "embed_models": {
            "en": settings.embed_model_en,
            "ru": settings.embed_model_ru,
        },
        "vectors": {lang: len(s) for lang, s in app.state.stores.items()},
    }
