"""Configuration settings for WorldSignal application."""
import os
from pydantic import BaseModel
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    postgres_user: str = os.getenv("POSTGRES_USER", "worldsignal")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "worldsignal")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "worldsignal")
    
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "world_signal_news")
    vector_size: int = int(os.getenv("VECTOR_SIZE", "384"))
    
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    class Config:
        arbitrary_types_allowed = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
