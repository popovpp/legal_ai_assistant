# app/main.py (полный файл)
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import analyses, documents
from app.config import settings
from app.ingestion.embedder import Embedder
from app.llm.client import LLMClient
from app.models.db import init_db
from app.rag.retriever import Retriever
from app.rag.vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)


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

    logging.getLogger(__name__).info(
        "startup ok vectors=%d models=%s", len(store), settings.llm_model_chain
    )
    yield


app = FastAPI(title="Legal AI Assistant", lifespan=lifespan)
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
