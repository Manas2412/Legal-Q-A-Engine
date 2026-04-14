from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database 
    database_url: str = Field(
        default="postgresql+psycopg2://user:password@host:port/database",
        validation_alias="DATABASE_URL"
    )
    
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="llama3", validation_alias="LLM_MODEL")
    embed_model: str = Field(default="nomic-embed-text", validation_alias="EMBED_MODEL")
    embed_dim: int = Field(default=768, validation_alias="EMBED_DIM")
    
    # Retrieval
    retrieval_top_k: int = Field(default=20, validation_alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=6, validation_alias="RERANK_TOP_K")
    bm25_weight: float = Field(default=0.4, validation_alias="BM25_WEIGHT")
    dense_weight: float = Field(default=0.6, validation_alias="DENSE_WEIGHT")
    
    # Memory
    short_term_window: int = Field(default=8, validation_alias="SHORT_TERM_WINDOW")
    semantic_memory_top_k: int = Field(default=5, validation_alias="SEMANTIC_MEMORY_TOP_K")
    
    # App
    app_title: str = Field(default="Legal Q&A Engine", validation_alias="APP_TITLE")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()