# app/llm/client.py
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    duration_ms: int


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self._models = models or settings.llm_model_chain
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)

    def _call_with_fallback(
        self,
        fn_factory: Callable[[str], Callable[[], R]],
        label: str,
    ) -> tuple[R, str]:
        last_exc: Exception | None = None
        for model in self._models:
            try:
                call = fn_factory(model)
                result = self._retry_loop(call, f"{label}[{model}]")
                return result, model
            except LLMError as exc:
                logger.warning("llm.fallback model=%s exhausted err=%s", model, exc)
                last_exc = exc
        raise LLMError(f"all models exhausted for {label}: {last_exc}") from last_exc

    def _usage(self, resp) -> tuple[int, int]:
        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            return 0, 0
        return (
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )

    def _retry_loop(self, fn, label: str):
        last_exc: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 — retry любых ошибок SDK
                last_exc = exc
                logger.warning("llm.%s failed attempt=%d err=%r", label, attempt, exc)
                if attempt < settings.llm_max_retries:
                    time.sleep(0.5 * (2**attempt))
        raise LLMError(f"{label} failed after retries: {last_exc}") from last_exc

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.monotonic()

        def _make_call(model: str) -> Callable[[], Any]:
            def _call():
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        system_instruction=system_instruction,
                    ),
                )

            return _call

        resp, model_used = self._call_with_fallback(_make_call, "generate")
        tokens_in, tokens_out = self._usage(resp)
        return LLMResponse(
            text=resp.text or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model_used,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ) -> tuple[T, LLMResponse]:
        budget = max_output_tokens
        last_raw = ""
        for attempt in range(2):  # первая попытка + один retry с удвоенным бюджетом
            start = time.monotonic()

            def _make_call(model: str, _budget: int = budget) -> Callable[[], Any]:
                def _call():
                    return self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=_budget,
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=schema,
                        ),
                    )

                return _call

            resp, model_used = self._call_with_fallback(_make_call, "generate_json")

            finish = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
            finish_name = getattr(finish, "name", str(finish)) if finish else "UNKNOWN"

            parsed = getattr(resp, "parsed", None)
            if parsed is not None and finish_name != "MAX_TOKENS":
                tokens_in, tokens_out = self._usage(resp)
                return parsed, LLMResponse(
                    text=resp.text or "",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model_used,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

            last_raw = resp.text or ""
            logger.warning(
                "generate_json truncation or no parsed: finish=%s budget=%d len=%d",
                finish_name,
                budget,
                len(last_raw),
            )
            if finish_name == "MAX_TOKENS" and attempt == 0:
                budget *= 2
                continue  # retry с большим бюджетом

            # Попытка распарсить то, что есть (может быть валидный JSON без parsed)
            try:
                parsed = schema.model_validate(json.loads(last_raw))
                tokens_in, tokens_out = self._usage(resp)
                return parsed, LLMResponse(
                    text=last_raw,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model_used,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                raise LLMError(
                    f"generate_json invalid (finish={finish_name}, budget={budget}): {exc}; raw={last_raw[:200]!r}"
                ) from exc

        raise LLMError(f"generate_json exhausted retries; last_raw={last_raw[:200]!r}")
