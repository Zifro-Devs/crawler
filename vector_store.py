"""
Vector store usando ChromaDB para almacenamiento y búsqueda semántica.
Cada sitio web crawleado se guarda en una colección separada.
"""
import hashlib
import re
from typing import Optional

import chromadb
from chromadb.config import Settings
from rich.console import Console

from chunker import Chunk
from embeddings import embed_texts, embed_query
from config import config

console = Console()


def _collection_name(domain: str) -> str:
    """
    Genera un nombre válido de colección ChromaDB desde el dominio.
    ChromaDB requiere: 3-63 chars, solo letras/números/guiones, empieza y termina con letra/número.
    """
    # Limpiar el dominio
    name = re.sub(r'[^a-zA-Z0-9\-]', '-', domain)
    name = re.sub(r'-+', '-', name).strip('-')
    name = name[:60]  # máx 63 chars

    # Asegurar que empieza con letra o número
    if name and not name[0].isalnum():
        name = "site-" + name

    # Mínimo 3 caracteres
    if len(name) < 3:
        name = name + "-kb"

    return name.lower()


def _chunk_id(chunk: Chunk) -> str:
    """Genera un ID único y determinístico para cada chunk."""
    content = f"{chunk.source_url}::{chunk.chunk_index}::{chunk.text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()


class VectorStore:
    """
    Gestiona la base de conocimiento vectorial para un sitio web.
    Cada dominio tiene su propia colección en ChromaDB.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.collection_name = _collection_name(domain)

        # Cliente ChromaDB persistente
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        # Obtener o crear la colección
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "domain": domain,
                "hnsw:space": "cosine",         # cosine similarity
                "hnsw:construction_ef": 200,    # mejor precisión en construcción
                "hnsw:M": 16,                   # número de vecinos en el grafo
            }
        )
        console.print(f"[dim]📦 Colección ChromaDB: [cyan]{self.collection_name}[/cyan][/dim]")

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """
        Agrega chunks a la colección con sus embeddings.
        Evita duplicados usando IDs determinísticos.

        Args:
            chunks: Lista de chunks a indexar
            batch_size: Tamaño de batch para inserción eficiente

        Returns:
            Número de chunks nuevos insertados
        """
        if not chunks:
            return 0

        # Generar IDs y deduplicar dentro del batch actual
        seen_ids = set()
        deduped = []
        for chunk in chunks:
            cid = _chunk_id(chunk)
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append((cid, chunk))

        ids = [d[0] for d in deduped]

        # Verificar cuáles ya existen en ChromaDB (en batches para evitar límite de tamaño)
        existing = set()
        batch_size_check = 500
        for i in range(0, len(ids), batch_size_check):
            batch_ids = ids[i:i + batch_size_check]
            result = self.collection.get(ids=batch_ids, include=[])
            existing.update(result["ids"])

        new_chunks = [(id_, chunk) for id_, chunk in deduped if id_ not in existing]

        if not new_chunks:
            console.print("[dim]  ↳ Todos los chunks ya estaban indexados[/dim]")
            return 0

        console.print(f"[dim]  ↳ Generando embeddings para {len(new_chunks)} chunks...[/dim]")

        # Procesar en batches
        inserted = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [b[0] for b in batch]
            batch_chunks = [b[1] for b in batch]

            texts = [c.text for c in batch_chunks]
            embeddings = embed_texts(texts)
            documents = texts
            metadatas = [c.metadata for c in batch_chunks]

            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            inserted += len(batch)

        return inserted

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_url: Optional[str] = None,
    ) -> list[dict]:
        """
        Busca chunks relevantes para una query.

        Args:
            query: Pregunta o texto de búsqueda
            top_k: Número de resultados a retornar
            filter_url: Filtrar por URL específica (opcional)

        Returns:
            Lista de resultados con texto, metadata y score de relevancia
        """
        top_k = top_k or config.TOP_K_RESULTS
        query_embedding = embed_query(query)

        where = {"source_url": filter_url} if filter_url else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convertir distancia cosine a score de similitud (0-1)
                similarity = 1 - dist
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                    "source_url": meta.get("source_url", ""),
                    "source_title": meta.get("source_title", ""),
                })

        # Ordenar por similitud descendente
        output.sort(key=lambda x: x["similarity"], reverse=True)
        return output

    def stats(self) -> dict:
        """Retorna estadísticas de la colección."""
        count = self.collection.count()
        return {
            "domain": self.domain,
            "collection": self.collection_name,
            "total_chunks": count,
        }

    def clear(self) -> None:
        """Borra todos los chunks de la colección."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"domain": self.domain, "hnsw:space": "cosine"},
        )
        console.print(f"[yellow]⚠ Colección '{self.collection_name}' limpiada[/yellow]")


def list_indexed_sites() -> list[dict]:
    """Lista todos los sitios indexados en ChromaDB."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collections = client.list_collections()
    return [
        {
            "name": col.name,
            "domain": col.metadata.get("domain", col.name),
            "chunks": col.count(),
        }
        for col in collections
    ]
