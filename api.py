"""
API REST — Microservicio de crawling e indexación para agentes de IA.

Diseñado para ser llamado desde un proyecto externo que necesita que su
agente aprenda del contenido de cualquier sitio web automáticamente.

Flujo de uso desde el agente externo:
    1. POST /index        → inicia crawling + indexación, devuelve job_id
    2. GET  /job/{job_id} → polling hasta status == "done"
    3. POST /search       → busca chunks relevantes para responder preguntas
    4. GET  /sites        → lista sitios ya indexados
    5. DELETE /site/{domain} → elimina un sitio del vector store

Extras:
    - POST /index acepta webhook_url para notificación push al terminar
    - GET  /site/{domain}/stats → estadísticas detalladas de un dominio
    - POST /index con re_index=true → fuerza re-crawl aunque ya esté indexado
"""
import asyncio
import hashlib
import logging
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator

from knowledge_base import KnowledgeBase, get_knowledge_base
from vector_store import VectorStore, list_indexed_sites
from config import config

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crawler.api")


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DE REQUEST / RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """Solicitud para indexar un sitio web."""

    url: str = Field(
        ...,
        description="URL del sitio a indexar. Puede ser el dominio raíz o un path específico.",
        examples=["https://miempresa.com", "https://miempresa.com/blog"],
    )
    re_index: bool = Field(
        default=False,
        description="Si es True, borra el índice existente y re-indexa desde cero.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description=(
            "URL a la que se hará un POST cuando el job termine. "
            "Recibirá el mismo payload que GET /job/{job_id}."
        ),
    )

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v


class SearchRequest(BaseModel):
    """Solicitud de búsqueda semántica sobre un dominio indexado."""

    domain: str = Field(
        ...,
        description=(
            "Dominio del sitio indexado. Puede ser el nombre de dominio "
            "(ej: 'miempresa.com') o la URL completa (se extrae el dominio)."
        ),
        examples=["miempresa.com", "https://miempresa.com"],
    )
    query: str = Field(
        ...,
        min_length=2,
        description="Pregunta o texto de búsqueda.",
        examples=["¿Cuál es el horario de atención?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de fragmentos relevantes a retornar.",
    )
    min_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Score mínimo de similitud (0-1). Fragmentos por debajo se descartan.",
    )

    @field_validator("domain")
    @classmethod
    def extract_domain(cls, v: str) -> str:
        v = v.strip()
        if v.startswith(("http://", "https://")):
            v = urlparse(v).netloc
        return v.lower()


# ── Modelos de respuesta ──────────────────────────────────────────────────────

class JobStatus(str, Enum):
    pending    = "pending"
    processing = "processing"
    done       = "done"
    error      = "error"


class IndexingStats(BaseModel):
    pages_crawled: int
    pages_failed: int
    total_chunks: int
    new_chunks: int
    total_words: int


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    domain: Optional[str] = None
    base_url: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    stats: Optional[IndexingStats] = None
    error: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    source_url: str
    source_title: str
    similarity: float


class SearchResponse(BaseModel):
    domain: str
    query: str
    results: list[SearchResult]
    total_results: int


class SiteInfo(BaseModel):
    domain: str
    collection: str
    total_chunks: int


class SitesResponse(BaseModel):
    total: int
    sites: list[SiteInfo]


class SiteStatsResponse(BaseModel):
    domain: str
    collection: str
    total_chunks: int


class DeleteResponse(BaseModel):
    domain: str
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# STORE DE JOBS EN MEMORIA
# Para producción con múltiples workers, reemplazar con Redis.
# ─────────────────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE INDEXACIÓN EN THREAD PROPIO
# Playwright (usado por Crawl4AI) requiere su propio event loop en Windows.
# ─────────────────────────────────────────────────────────────────────────────

def _run_index_job(job_id: str, url: str, re_index: bool, webhook_url: Optional[str]) -> None:
    """
    Ejecuta el pipeline completo: crawl → chunk → embed → ChromaDB.
    Corre en un thread separado con su propio event loop.
    """
    job = _jobs[job_id]
    job["status"] = JobStatus.processing
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(f"[{job_id}] Iniciando indexación de {url}")

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        domain = urlparse(url).netloc

        # Si re_index, limpiar el vector store existente antes de crawlear
        if re_index:
            logger.info(f"[{job_id}] re_index=True → limpiando índice existente de '{domain}'")
            store = VectorStore(domain)
            store.clear()

        # Pipeline completo: crawl + chunk + embed + ChromaDB
        kb, result = loop.run_until_complete(KnowledgeBase.from_url(url))

        finished_at = datetime.now(timezone.utc).isoformat()
        started_dt = datetime.fromisoformat(job["started_at"])
        finished_dt = datetime.fromisoformat(finished_at)
        duration = (finished_dt - started_dt).total_seconds()

        job.update({
            "status": JobStatus.done,
            "domain": result.domain,
            "base_url": result.base_url,
            "finished_at": finished_at,
            "duration_seconds": round(duration, 2),
            "stats": {
                "pages_crawled": result.pages_crawled,
                "pages_failed": result.pages_failed,
                "total_chunks": result.total_chunks,
                "new_chunks": result.new_chunks,
                "total_words": result.total_words,
            },
            "error": None,
        })

        logger.info(
            f"[{job_id}] Indexación completada en {duration:.1f}s — "
            f"{result.pages_crawled} páginas, {result.new_chunks} chunks nuevos"
        )

    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        started_dt = datetime.fromisoformat(job.get("started_at", finished_at))
        finished_dt = datetime.fromisoformat(finished_at)
        duration = (finished_dt - started_dt).total_seconds()

        job.update({
            "status": JobStatus.error,
            "finished_at": finished_at,
            "duration_seconds": round(duration, 2),
            "error": str(exc),
        })
        logger.error(f"[{job_id}] Error en indexación: {exc}", exc_info=True)

    finally:
        loop.close()
        # Notificar via webhook si fue configurado
        if webhook_url:
            _fire_webhook(webhook_url, job_id, job)


def _fire_webhook(webhook_url: str, job_id: str, job: dict) -> None:
    """
    Envía notificación POST al webhook cuando el job termina.
    Se ejecuta en el mismo thread del job, al final.
    """
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "domain": job.get("domain"),
        "base_url": job.get("base_url"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "duration_seconds": job.get("duration_seconds"),
        "stats": job.get("stats"),
        "error": job.get("error"),
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(webhook_url, json=payload)
            logger.info(f"Webhook enviado a {webhook_url} → HTTP {resp.status_code}")
    except Exception as exc:
        logger.warning(f"Webhook falló ({webhook_url}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Crawler API iniciada y lista.")
    yield
    logger.info("Crawler API detenida.")


app = FastAPI(
    title="Crawler & Knowledge Base API",
    description=(
        "Microservicio de crawling e indexación semántica. "
        "Pega el link de un sitio web y el agente aprende de todo su contenido automáticamente."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite llamadas desde cualquier origen (ajustar en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health():
    """Verifica que el servicio está operativo."""
    return {
        "status": "ok",
        "service": "Crawler & Knowledge Base API",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Indexación ────────────────────────────────────────────────────────────────

@app.post(
    "/index",
    status_code=202,
    response_model=JobResponse,
    tags=["Indexación"],
    summary="Indexar un sitio web",
    description=(
        "Inicia el pipeline completo: crawl del sitio → extracción de texto → "
        "chunking semántico → embeddings → almacenamiento en vector store. "
        "Devuelve inmediatamente un `job_id`. Usa `GET /job/{job_id}` para "
        "consultar el estado."
    ),
)
async def start_index(request: IndexRequest):
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "status": JobStatus.pending,
        "domain": None,
        "base_url": request.url,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "stats": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_index_job,
        args=(job_id, request.url, request.re_index, request.webhook_url),
        daemon=True,
        name=f"index-{job_id[:8]}",
    )
    thread.start()

    logger.info(f"Job {job_id} creado para URL: {request.url}")

    return JobResponse(
        job_id=job_id,
        status=JobStatus.pending,
        base_url=request.url,
    )


@app.get(
    "/job/{job_id}",
    response_model=JobResponse,
    tags=["Indexación"],
    summary="Estado de un job de indexación",
    description=(
        "Consulta el estado de un job. Haz polling cada pocos segundos hasta "
        "que el status sea `done` o `error`."
    ),
)
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")

    job = _jobs[job_id]

    stats = None
    if job.get("stats"):
        stats = IndexingStats(**job["stats"])

    return JobResponse(
        job_id=job_id,
        status=job["status"],
        domain=job.get("domain"),
        base_url=job.get("base_url"),
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        duration_seconds=job.get("duration_seconds"),
        stats=stats,
        error=job.get("error"),
    )


# ── Búsqueda semántica ────────────────────────────────────────────────────────

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Búsqueda"],
    summary="Buscar contenido en un dominio indexado",
    description=(
        "Realiza búsqueda semántica sobre el contenido de un sitio ya indexado. "
        "Devuelve los fragmentos más relevantes ordenados por similitud. "
        "Úsalos como contexto para tu LLM/agente."
    ),
)
async def search(request: SearchRequest):
    try:
        store = VectorStore(request.domain)

        # Verificar que el dominio tiene chunks
        stats = store.stats()
        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"El dominio '{request.domain}' no está indexado o no tiene contenido. "
                    "Usa POST /index primero."
                ),
            )

        raw_results = store.search(request.query, top_k=request.top_k)

        # Filtrar por similitud mínima
        filtered = [
            r for r in raw_results
            if r["similarity"] >= request.min_similarity
        ]

        results = [
            SearchResult(
                text=r["text"],
                source_url=r["source_url"],
                source_title=r["source_title"],
                similarity=r["similarity"],
            )
            for r in filtered
        ]

        return SearchResponse(
            domain=request.domain,
            query=request.query,
            results=results,
            total_results=len(results),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error en búsqueda para '{request.domain}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Gestión de sitios ─────────────────────────────────────────────────────────

@app.get(
    "/sites",
    response_model=SitesResponse,
    tags=["Sitios"],
    summary="Listar sitios indexados",
    description="Devuelve todos los sitios web que han sido indexados, con su número de chunks.",
)
async def list_sites():
    try:
        sites_raw = list_indexed_sites()
        sites = [
            SiteInfo(
                domain=s["domain"],
                collection=s["name"],
                total_chunks=s["chunks"],
            )
            for s in sites_raw
        ]
        return SitesResponse(total=len(sites), sites=sites)
    except Exception as exc:
        logger.error(f"Error listando sitios: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/site/{domain}/stats",
    response_model=SiteStatsResponse,
    tags=["Sitios"],
    summary="Estadísticas de un dominio",
    description="Devuelve estadísticas detalladas del vector store de un dominio específico.",
)
async def site_stats(domain: str):
    try:
        store = VectorStore(domain)
        stats = store.stats()

        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"El dominio '{domain}' no está indexado.",
            )

        return SiteStatsResponse(
            domain=stats["domain"],
            collection=stats["collection"],
            total_chunks=stats["total_chunks"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error obteniendo stats de '{domain}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/site/{domain}",
    response_model=DeleteResponse,
    tags=["Sitios"],
    summary="Eliminar un dominio del índice",
    description=(
        "Borra completamente el vector store de un dominio. "
        "Útil antes de re-indexar o para limpiar datos obsoletos. "
        "También puedes usar `re_index=true` en POST /index para hacer esto automáticamente."
    ),
)
async def delete_site(domain: str):
    try:
        store = VectorStore(domain)
        stats = store.stats()

        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"El dominio '{domain}' no está indexado.",
            )

        store.clear()
        logger.info(f"Dominio '{domain}' eliminado del vector store.")

        return DeleteResponse(
            domain=domain,
            message=f"Dominio '{domain}' eliminado correctamente del índice.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error eliminando dominio '{domain}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Manejador global de errores ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor.",
            "detail": str(exc),
        },
    )
