# app/config.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    gemini_api_key: str = ""
    llm_models: str = "gemini-3.6-flash,gemini-3.8-flash"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # Embeddings
    embed_model_en: str = "all-MiniLM-L6-v2"
    embed_model_ru: str = "Roflmax/bge-m3-russian-legal"

    # Retrieval
    top_k: int = 6
    min_score: float = 0.30

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Graph
    recursion_limit: int = 8
    max_verifier_retries: int = 1

    # Storage
    data_dir: Path = Path("data")

    @property
    def index_dir_en(self) -> Path:
        return self.data_dir / "index_en"

    @property
    def index_dir_ru(self) -> Path:
        return self.data_dir / "index_ru"

    @property
    def llm_model_chain(self) -> list[str]:
        return [m.strip() for m in self.llm_models.split(",") if m.strip()]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.uploads_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
