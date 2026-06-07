"""
Configuration centralisée du système RAG.
Toutes les constantes sont ici — ne jamais les éparpiller dans le code.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # --- Chemins ---
    artifacts_dir: Path = Path("app/artifacts")

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_length: int = 50

    # --- Embedding (multilingue FR/AR/EN) ---
    embed_model: str = "intfloat/multilingual-e5-large"
    embed_batch_size: int = 32

    # --- Retrieval ---
    retrieval_k: int = 10
    rerank_top_n: int = 4
    mmr_lambda: float = 0.7

    # --- Reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM ---
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- Clé API (depuis env ou Streamlit Secrets) ---
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))


# Singleton — importer `cfg` partout
cfg = RAGConfig()
