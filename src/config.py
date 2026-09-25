"""
Configuración centralizada del proyecto.
Todo se lee de variables de entorno (ver .env.example) para no hardcodear
secretos ni acoplar el código a un proveedor específico.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- Fuente de datos ---
    github_repo: str = os.getenv("GITHUB_REPO", "dbt-labs/docs.getdbt.com")
    github_branch: str = os.getenv("GITHUB_BRANCH", "current")
    github_docs_path: str = os.getenv("GITHUB_DOCS_PATH", "website/docs/docs")
    github_token: str | None = os.getenv("GITHUB_TOKEN")  # opcional, sube el rate limit de 60 a 5000 req/h

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # --- Embeddings ---
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local | openai | voyage
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))  # 384 para bge-small; ajustar si cambiás de modelo

    # --- LLM de generación ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))

    # --- Base de datos vectorial (Chroma, embebido — sin servidor, sin Docker) ---
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "dbt_doc_chunks")


settings = Settings()
