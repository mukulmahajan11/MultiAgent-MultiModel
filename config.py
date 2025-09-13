# config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ASTRA_DB_APPLICATION_TOKEN: str = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
    ASTRA_DB_KEYSPACE: str = os.getenv("ASTRA_DB_KEYSPACE", "langchain_db")
    ASTRA_DB_ID: str = os.getenv("ASTRA_DB_ID", "")
    ASTRA_DB_ENDPOINT: str = os.getenv("ASTRA_DB_ENDPOINT", "")
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VECTOR_TABLE: str = os.getenv("VECTOR_TABLE", "finance_docs")
    TOP_K: int = int(os.getenv("TOP_K", "5"))

settings = Settings()
