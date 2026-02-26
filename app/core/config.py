import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from pydantic_settings import BaseSettings

def _get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    return key or ""


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = _get_api_key()

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ChromaDB
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/vectorstore")
    collection_name: str = "documents"

    # Upload
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    top_k: int = 8

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024

    class Config:
        env_file = ".env"


settings = Settings()