# app/api/deps.py
from fastapi import Request

from app.ingestion.embedder import Embedder
from app.rag.retriever import RetrieverRegistry
from app.rag.vector_store import FaissVectorStore


def get_embedders(request: Request) -> dict[str, Embedder]:
    return request.app.state.embedders


def get_stores(request: Request) -> dict[str, FaissVectorStore]:
    return request.app.state.stores


def get_retriever(request: Request) -> RetrieverRegistry:
    return request.app.state.retriever
