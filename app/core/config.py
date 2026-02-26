try:
    from pydantic_settings import BaseSettings
except Exception as e:
    import streamlit as st
    st.error(f"Import error: {type(e).__name__}: {e}")
    st.stop()

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

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
    top_k: int = 5

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024

    class Config:
        env_file = ".env"


settings = Settings()