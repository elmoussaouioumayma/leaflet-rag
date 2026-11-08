"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    index_dir: Path
    retriever: str  # "tfidf" or "embeddings"
    embedding_model: str
    top_k: int
    min_score: float  # below this retrieval score the assistant says it does not know
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    llm_temperature: float

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key or self.llm_base_url)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        raw_dir=Path(_env("LEAFLET_RAW_DIR", "data/raw")),
        index_dir=Path(_env("LEAFLET_INDEX_DIR", "data/index")),
        retriever=_env("LEAFLET_RETRIEVER", "tfidf"),
        embedding_model=_env("LEAFLET_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        top_k=int(_env("LEAFLET_TOP_K", "4")),
        min_score=float(_env("LEAFLET_MIN_SCORE", "0.12")),
        llm_base_url=os.environ.get("LLM_BASE_URL") or None,
        llm_api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None,
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0")),
    )
