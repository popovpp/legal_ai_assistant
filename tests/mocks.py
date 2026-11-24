# tests/mocks.py
from typing import Any

from app.llm.client import LLMClient, LLMError, LLMResponse
from app.rag.vector_store import ScoredChunk


class MockLLM(LLMClient):
    """Детерминированный мок LLMClient.

    responses: очередь ответов для generate_json. Элемент может быть:
      - экземпляром Pydantic-схемы (возвращается как есть)
      - dict (конвертируется в переданную schema)
      - Exception (будет поднят)
    Если очередь исчерпана — LLMError.
    raise_error=True — всегда бросать LLMError (для теста error path).
    """

    def __init__(self, responses: list[Any] | None = None, raise_error: bool = False):
        # намеренно не вызываем super().__init__, чтобы не создавать genai.Client
        self._responses = list(responses or [])
        self._idx = 0
        self.calls: list[dict] = []
        self.raise_error = raise_error

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        if self.raise_error:
            raise LLMError("mock error")
        self.calls.append({"kind": "generate", "prompt": prompt})
        return LLMResponse(text="mock", tokens_in=10, tokens_out=10, model="mock", duration_ms=1)

    def generate_json(
        self,
        prompt: str,
        schema: type,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> tuple[Any, LLMResponse]:
        if self.raise_error:
            raise LLMError("mock error")
        self.calls.append({"kind": "generate_json", "prompt": prompt, "schema": schema.__name__})
        if self._idx >= len(self._responses):
            raise LLMError(f"MockLLM exhausted after {len(self._responses)} responses")
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, dict):
            resp = schema(**resp)
        return resp, LLMResponse(
            text=resp.model_dump_json(),
            tokens_in=10,
            tokens_out=10,
            model="mock",
            duration_ms=1,
        )


class MockRetriever:
    def __init__(self, chunks: list[ScoredChunk]):
        self._chunks = chunks
        self.calls: list[str] = []

    @property
    def index_version(self) -> int:
        return len(self._chunks)

    def retrieve(self, language: str, query: str, k: int = 6) -> list[ScoredChunk]:
        self.calls.append(query)
        return self._chunks[:k]
