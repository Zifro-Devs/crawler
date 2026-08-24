"""
Chunking inteligente de documentos para RAG.
Divide el contenido en fragmentos semánticamente coherentes,
respetando límites de párrafos y secciones.
"""
import re
from dataclasses import dataclass
from typing import Optional

import tiktoken

from config import config


@dataclass
class Chunk:
    """Un fragmento de texto listo para embedding."""
    text: str
    source_url: str
    source_title: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: dict


# Usamos cl100k_base (compatible con OpenAI y la mayoría de modelos)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Cuenta tokens de forma precisa."""
    return len(_tokenizer.encode(text))


def split_by_headings(markdown: str) -> list[str]:
    """
    Divide el markdown en secciones por headings (H1, H2, H3).
    Mantiene el heading con su sección para preservar contexto.
    """
    # Split en líneas de heading de nivel 1, 2 o 3
    pattern = r'(?=^#{1,3} )'
    sections = re.split(pattern, markdown, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def split_by_paragraphs(text: str) -> list[str]:
    """Divide texto en párrafos."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def merge_short_chunks(chunks: list[str], min_tokens: int = 50) -> list[str]:
    """Fusiona chunks demasiado cortos con el siguiente."""
    merged = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            candidate = buffer + "\n\n" + chunk
            if count_tokens(buffer) < min_tokens:
                buffer = candidate
                continue
        if buffer:
            merged.append(buffer)
        buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def chunk_text(
    text: str,
    source_url: str,
    source_title: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Chunk]:
    """
    Divide un texto en chunks optimizados para RAG.

    Estrategia:
    1. Divide por headings (secciones semánticas)
    2. Si una sección es muy grande, la subdivide por párrafos
    3. Si aún es grande, hace sliding window con overlap
    4. Fusiona chunks muy cortos

    Args:
        text: Texto en markdown a dividir
        source_url: URL de origen
        source_title: Título de la página de origen
        chunk_size: Tamaño máximo en tokens (default del config)
        chunk_overlap: Overlap en tokens (default del config)

    Returns:
        Lista de Chunk listos para embedding
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    if not text.strip():
        return []

    # Paso 1: dividir por headings
    sections = split_by_headings(text)
    if not sections:
        sections = [text]

    raw_chunks: list[str] = []

    for section in sections:
        tokens = count_tokens(section)

        if tokens <= chunk_size:
            # La sección entera cabe en un chunk
            raw_chunks.append(section)
        else:
            # La sección es muy grande: dividir por párrafos
            paragraphs = split_by_paragraphs(section)
            current_chunk = ""
            current_tokens = 0

            for para in paragraphs:
                para_tokens = count_tokens(para)

                if current_tokens + para_tokens <= chunk_size:
                    current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        raw_chunks.append(current_chunk)

                    # Si el párrafo solo ya es demasiado grande, hacer sliding window
                    if para_tokens > chunk_size:
                        sub_chunks = _sliding_window(para, chunk_size, chunk_overlap)
                        raw_chunks.extend(sub_chunks)
                        current_chunk = ""
                        current_tokens = 0
                    else:
                        # Iniciar nuevo chunk con overlap del anterior
                        overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                        current_chunk = (overlap_text + "\n\n" + para).strip() if overlap_text else para
                        current_tokens = count_tokens(current_chunk)

            if current_chunk:
                raw_chunks.append(current_chunk)

    # Paso 2: fusionar chunks demasiado cortos
    raw_chunks = merge_short_chunks(raw_chunks, min_tokens=50)

    # Paso 3: construir objetos Chunk con metadata
    total = len(raw_chunks)
    chunks: list[Chunk] = []

    for i, text_chunk in enumerate(raw_chunks):
        token_count = count_tokens(text_chunk)
        chunks.append(Chunk(
            text=text_chunk,
            source_url=source_url,
            source_title=source_title,
            chunk_index=i,
            total_chunks=total,
            token_count=token_count,
            metadata={
                "source_url": source_url,
                "source_title": source_title,
                "chunk_index": i,
                "total_chunks": total,
                "token_count": token_count,
            }
        ))

    return chunks


def _sliding_window(text: str, window_size: int, overlap: int) -> list[str]:
    """Sliding window sobre tokens para textos muy largos."""
    tokens = _tokenizer.encode(text)
    chunks = []
    step = window_size - overlap

    for start in range(0, len(tokens), step):
        end = min(start + window_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break

    return chunks


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Obtiene los últimos N tokens de un texto para usar como overlap."""
    if not text:
        return ""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    overlap = tokens[-overlap_tokens:]
    return _tokenizer.decode(overlap)
