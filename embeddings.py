"""
Motor de embeddings local usando sentence-transformers.
Sin costo, funciona offline, alta calidad para RAG en español e inglés.
"""
from functools import lru_cache
from typing import Union

from sentence_transformers import SentenceTransformer
from rich.console import Console

from config import config

console = Console()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Carga el modelo de embeddings (singleton, se carga una sola vez).
    all-MiniLM-L6-v2: rápido, ligero, buena calidad multilenguaje.
    Para mejor calidad en español usar: 'paraphrase-multilingual-MiniLM-L12-v2'
    """
    console.print(f"[dim]Cargando modelo de embeddings: {config.EMBEDDING_MODEL}...[/dim]")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    console.print(f"[dim green]✓ Modelo de embeddings listo[/dim green]")
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Genera embeddings para una lista de textos.

    Args:
        texts: Lista de textos a embedear
        batch_size: Tamaño de batch para procesamiento eficiente

    Returns:
        Lista de vectores de embedding (float32 normalizado)
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,     # normalizar para cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Genera embedding para una query de búsqueda.
    Optimizado para recuperación (prefijo de instrucción).
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
