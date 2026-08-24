"""
Configuración central del microservicio de crawling.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Embeddings (local, sin costo)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ChromaDB
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # Crawling
    MAX_PAGES_PER_SITE: int = int(os.getenv("MAX_PAGES_PER_SITE", "50"))
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "3"))

    # Chunking
    CHUNK_SIZE: int = 800       # tokens por chunk
    CHUNK_OVERLAP: int = 100    # overlap entre chunks

    # RAG
    TOP_K_RESULTS: int = 5      # resultados a recuperar por query


config = Config()
