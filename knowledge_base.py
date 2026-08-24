"""
Pipeline principal: conecta crawler → chunker → vector store.
Es el módulo central que orquesta el proceso de entrenamiento.
"""
import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crawler import crawl_website, CrawlResult
from chunker import chunk_text, Chunk
from vector_store import VectorStore, list_indexed_sites
from config import config

console = Console()


@dataclass
class IndexingResult:
    """Resultado del proceso completo de indexación."""
    domain: str
    base_url: str
    pages_crawled: int
    pages_failed: int
    total_chunks: int
    new_chunks: int
    total_words: int
    collection_name: str


class KnowledgeBase:
    """
    Base de conocimiento de un sitio web.
    Combina crawling + chunking + vectorización en un solo pipeline.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.store = VectorStore(domain)

    @classmethod
    async def from_url(cls, url: str) -> tuple["KnowledgeBase", IndexingResult]:
        """
        Crea una KnowledgeBase desde una URL, ejecutando el pipeline completo.

        Args:
            url: URL del sitio a indexar

        Returns:
            Tupla (KnowledgeBase, IndexingResult)
        """
        # Paso 1: Crawlear el sitio
        crawl_result = await crawl_website(url)
        domain = crawl_result.domain

        kb = cls(domain)

        # Paso 2: Chunkear e indexar
        result = await kb._index_crawl_result(crawl_result)

        return kb, result

    async def _index_crawl_result(self, crawl_result: CrawlResult) -> IndexingResult:
        """Indexa el resultado del crawling en el vector store."""
        all_chunks: list[Chunk] = []

        console.print("\n[bold blue]📝 Procesando y dividiendo contenido...[/bold blue]")

        for page in crawl_result.pages:
            chunks = chunk_text(
                text=page.markdown,
                source_url=page.url,
                source_title=page.title,
            )
            all_chunks.extend(chunks)
            console.print(
                f"  [dim]↳ {page.title[:60]:<60}[/dim] "
                f"[cyan]{len(chunks)} chunks[/cyan] "
                f"[dim]({page.word_count:,} palabras)[/dim]"
            )

        console.print(f"\n[bold blue]🔢 Total chunks a indexar: {len(all_chunks)}[/bold blue]")

        # Paso 3: Insertar en ChromaDB
        console.print("\n[bold blue]💾 Indexando en vector store...[/bold blue]")
        new_chunks = self.store.add_chunks(all_chunks)

        stats = self.store.stats()

        result = IndexingResult(
            domain=crawl_result.domain,
            base_url=crawl_result.base_url,
            pages_crawled=crawl_result.success_count(),
            pages_failed=len(crawl_result.failed_urls),
            total_chunks=stats["total_chunks"],
            new_chunks=new_chunks,
            total_words=crawl_result.total_words,
            collection_name=stats["collection"],
        )

        _print_indexing_summary(result)
        return result

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Busca contenido relevante en la base de conocimiento."""
        return self.store.search(query, top_k=top_k)

    def stats(self) -> dict:
        """Retorna estadísticas de la base de conocimiento."""
        return self.store.stats()

    def clear(self) -> None:
        """Limpia la base de conocimiento."""
        self.store.clear()


def get_knowledge_base(domain: str) -> KnowledgeBase:
    """Obtiene una KnowledgeBase existente por dominio."""
    return KnowledgeBase(domain)


def _print_indexing_summary(result: IndexingResult) -> None:
    """Imprime un resumen visual del proceso de indexación."""
    table = Table(title="📊 Resumen de Indexación", border_style="blue")
    table.add_column("Métrica", style="cyan", width=25)
    table.add_column("Valor", style="bold white")

    table.add_row("Dominio", result.domain)
    table.add_row("URL base", result.base_url)
    table.add_row("Páginas procesadas", str(result.pages_crawled))
    table.add_row("Páginas fallidas", str(result.pages_failed))
    table.add_row("Total palabras", f"{result.total_words:,}")
    table.add_row("Chunks creados", str(result.new_chunks))
    table.add_row("Total en store", str(result.total_chunks))
    table.add_row("Colección", result.collection_name)

    console.print()
    console.print(table)
    console.print()


async def index_url(url: str) -> tuple[KnowledgeBase, IndexingResult]:
    """Función de conveniencia para indexar una URL."""
    return await KnowledgeBase.from_url(url)


def show_indexed_sites() -> None:
    """Muestra todos los sitios indexados en la base de datos."""
    sites = list_indexed_sites()

    if not sites:
        console.print("[yellow]No hay sitios indexados aún.[/yellow]")
        return

    table = Table(title="🗄  Sitios Indexados", border_style="green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Dominio", style="cyan")
    table.add_column("Colección", style="dim")
    table.add_column("Chunks", justify="right", style="bold")

    for i, site in enumerate(sites, 1):
        table.add_row(
            str(i),
            site["domain"],
            site["name"],
            str(site["chunks"]),
        )

    console.print()
    console.print(table)
    console.print()
