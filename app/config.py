# app/config.py
# Central configuration using pydantic-settings.
#
# Why pydantic-settings instead of os.getenv() everywhere?
# Two reasons:
# 1. Type safety — CHROMA_K=4 is an int, not the string "4"
# 2. Single source of truth — all config in one place,
#    validated at startup. If OPENAI_API_KEY is missing,
#    the app fails immediately with a clear error instead of
#    crashing on the first API call with a confusing message.

from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"
class Settings(BaseSettings):
    # ── LLM settings ─────────────────────────────────────────────────────────
    openai_api_key: str           # Required — app will not start without this
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── LangSmith settings ────────────────────────────────────────────────────
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "langchain-copilot"

    # ── Tavily settings ───────────────────────────────────────────────────────
    tavily_api_key: str = ""      # Optional — web search disabled if not set

    # ── Vector store settings ─────────────────────────────────────────────────
    chroma_path: str = "chroma_db"
    chroma_k: int = 4             # Number of chunks to retrieve per query

    # ── Chunking settings ─────────────────────────────────────────────────────
    chunk_size: int = 1500
    chunk_overlap: int = 400

    model_config = {
        "env_file": str(ENV_FILE_PATH),
        "extra": "ignore",
        "case_sensitive": False,
    }

# Create a single instance used across the entire application.
# Import this object wherever you need a setting:
#   from app.config import settings
#   model = ChatOpenAI(model=settings.openai_chat_model)
settings = Settings()