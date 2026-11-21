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
from app.rag.retriever import Retriever
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

    embedder = Embedder(settings.embed_model)
    store = FaissVectorStore(settings.index_dir)
    store.load()
    retriever = Retriever(embedder, store, min_score=settings.min_score)
    llm = LLMClient()

    app.state.embedder = embedder
    app.state.vector_store = store
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.cache = InMemoryLRU(max_size=256)

    logging.getLogger(__name__).info("startup ok vectors=%d models=%s", len(store), settings.llm_model_chain)
    yield


app = FastAPI(title="Legal AI Assistant", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.include_router(documents.router)
app.include_router(analyses.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(settings.data_dir),
        "llm_models": settings.llm_model_chain,
        "embed_model": settings.embed_model,
        "vectors": len(app.state.vector_store),
    }
