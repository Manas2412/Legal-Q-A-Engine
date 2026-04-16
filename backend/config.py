from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/legal_qa",
        env="DATABASE_URL",
    )

    # ── Ollama ──────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL",
    )
    llm_model: str = Field(default="llama3", env="LLM_MODEL")
    embed_model: str = Field(default="nomic-embed-text", env="EMBED_MODEL")
    embed_dim: int = Field(default=768, env="EMBED_DIM")

    # ── Retrieval ───────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, env="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=6, env="RERANK_TOP_K")
    bm25_weight: float = Field(default=0.4, env="BM25_WEIGHT")
    dense_weight: float = Field(default=0.6, env="DENSE_WEIGHT")

    # ── Memory ──────────────────────────────────────────────────
    short_term_window: int = Field(default=8, env="SHORT_TERM_WINDOW")
    semantic_memory_top_k: int = Field(default=3, env="SEMANTIC_MEMORY_TOP_K")

    # ── CORS ─────────────────────────────────────────────────────
    # Comma-separated list of allowed frontend origins.
    # For development: "http://localhost:3000,http://localhost:5173"
    # For production: replace with your deployed frontend URL.
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        env="CORS_ORIGINS",
    )

    @property
    def cors_allow_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]

    # ── App ──────────────────────────────────────────────────────
    app_title: str = "Legal Q&A Engine — Indian Law"
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()