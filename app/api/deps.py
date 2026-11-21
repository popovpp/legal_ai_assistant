# app/api/deps.py
from fastapi import Request

from app.ingestion.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever
