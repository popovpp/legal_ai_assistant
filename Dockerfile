# syntax=docker/dockerfile:1

# ---------- Builder ----------
    FROM python:3.12-slim-bookworm AS builder

    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    ENV UV_COMPILE_BYTECODE=1 \
        UV_LINK_MODE=copy \
        UV_PYTHON_DOWNLOADS=0 \
        UV_NO_DEV=1
    
    WORKDIR /app
    
    # 1. Только зависимости — слой кэшируется, пока не меняются pyproject.toml и uv.lock
    RUN --mount=type=cache,target=/root/.cache/uv \
        --mount=type=bind,source=uv.lock,target=uv.lock \
        --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
        uv sync --locked --no-install-project
    
    # 2. Код проекта + установка самого проекта
    COPY . /app
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --locked
    
    # 3. Бейкаем embedding-модель в образ, чтобы первый запрос не висел на скачивании
    RUN uv run python -c "from sentence_transformers import SentenceTransformer; \
        SentenceTransformer('all-MiniLM-L6-v2')"
    
    # ---------- Runtime ----------
    FROM python:3.12-slim-bookworm AS runtime
    
    RUN apt-get update && apt-get install -y --no-install-recommends curl \
     && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    COPY --from=builder /app/.venv /app/.venv
    COPY --from=builder /app/app /app/app
    
    ENV PATH="/app/.venv/bin:$PATH" \
        PYTHONUNBUFFERED=1 \
        HF_HOME=/app/.cache/huggingface
    
    RUN useradd -m appuser && mkdir -p /app/data /app/.cache && \
        chown -R appuser:appuser /app
    USER appuser
    
    EXPOSE 8000
    
    HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
      CMD curl -f http://localhost:8000/health || exit 1
    
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
 