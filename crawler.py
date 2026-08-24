"""
Motor de crawling usando Crawl4AI.
Extrae contenido limpio de un sitio web completo de forma asíncrona.

Estrategias de descubrimiento de URLs (en orden de preferencia):
  1. Sitemap XML  — más completo, especialmente para SPAs
  2. BFS sobre links HTML — fallback para sitios sin sitemap
"""
import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import config

console = Console()


@dataclass
class PageResult:
    """Resultado de scraping de una página individual."""
    url: str
    title: str
    markdown: str
    word_count: int
    status_code: int = 200
    error: Optional[str] = None


@dataclass
class CrawlResult:
    """Resultado completo del crawling de un sitio."""
    base_url: str
    domain: str
    pages: list[PageResult] = field(default_factory=list)
    total_words: int = 0
    failed_urls: list[str] = field(default_factory=list)

    def success_count(self) -> int:
        return len([p for p in self.pages if p.error is None])


def extract_title(markdown: str, url: str) -> str:
    """Extrae el título de la página desde el markdown."""
    lines = markdown.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()
    # fallback: usar el path de la URL
    path = urlparse(url).path.strip("/").replace("-", " ").replace("/", " › ")
    return path.title() if path else urlparse(url).netloc


def clean_markdown(markdown: str) -> str:
    """
    Limpia el markdown de ruido innecesario:
    - Elimina líneas con solo símbolos
    - Colapsa espacios en blanco excesivos
    - Elimina fragmentos muy cortos
    """
    if not markdown:
        return ""

    lines = markdown.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Ignorar líneas vacías consecutivas (dejar máximo 1)
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # Ignorar líneas que son solo caracteres especiales (separadores, etc.)
        if re.match(r'^[\-\=\*\_\#\~\`\|\s]{3,}$', stripped):
            continue
        # Ignorar líneas muy cortas que no aportan contenido
        if len(stripped) < 3 and not stripped.startswith("#"):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # Colapsar más de 2 saltos de línea consecutivos
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def is_useful_content(markdown: str, min_words: int = 10) -> bool:
    """Verifica si la página tiene contenido útil suficiente."""
    words = len(markdown.split())
    return words >= min_words


async def discover_urls_from_sitemap(base_url: str, limit: int = 200) -> list[str]:
    """
    Intenta descubrir URLs del sitio via sitemap.xml y robots.txt.
    Retorna lista de URLs del mismo dominio, vacía si no hay sitemap.

    Busca en:
      - /sitemap.xml
      - /sitemap_index.xml
      - /robots.txt  (para encontrar la ubicación del sitemap)
    """
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    domain = parsed.netloc
    found_urls: list[str] = []

    sitemap_candidates = [
        f"{origin}/sitemap.xml",
        f"{origin}/sitemap_index.xml",
        f"{origin}/sitemap-index.xml",
        f"{origin}/sitemaps/sitemap.xml",
    ]

    # Intentar extraer sitemap de robots.txt
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(f"{origin}/robots.txt",
                                 headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                for line in r.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        if sitemap_url not in sitemap_candidates:
                            sitemap_candidates.insert(0, sitemap_url)
    except Exception:
        pass

    async def fetch_sitemap(url: str) -> list[str]:
        """Descarga y parsea un sitemap XML, soporta sitemapindex."""
        urls: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    return []
                root = ET.fromstring(r.text)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                # Sitemap index: contiene otros sitemaps
                for sitemap_tag in root.findall("sm:sitemap/sm:loc", ns):
                    sub_urls = await fetch_sitemap(sitemap_tag.text.strip())
                    urls.extend(sub_urls)

                # Sitemap normal: contiene URLs de páginas
                # Aceptar URLs con o sin www (zifro.app == www.zifro.app)
                bare_domain = domain[4:] if domain.startswith("www.") else domain
                for url_tag in root.findall("sm:url/sm:loc", ns):
                    loc = url_tag.text.strip()
                    if bare_domain in loc:
                        urls.append(loc)

        except Exception:
            pass
        return urls

    for candidate in sitemap_candidates:
        urls = await fetch_sitemap(candidate)
        if urls:
            console.print(
                f"[dim green]🗺  Sitemap encontrado:[/dim green] [dim]{candidate}[/dim] "
                f"[dim]→ {len(urls)} URLs[/dim]"
            )
            found_urls = urls
            break

    # Normalizar: aceptar URLs con o sin www como equivalentes
    # Ej: usuario puso www.zifro.app pero sitemap tiene zifro.app → igual
    netloc = parsed.netloc
    if netloc.startswith("www."):
        alt_netloc = netloc[4:]          # www.zifro.app → zifro.app
    else:
        alt_netloc = "www." + netloc     # zifro.app     → www.zifro.app

    base_prefix       = parsed.scheme + "://" + netloc    + parsed.path.rstrip("/")
    base_prefix_alt   = parsed.scheme + "://" + alt_netloc + parsed.path.rstrip("/")

    def _matches_prefix(u: str) -> bool:
        return u.startswith(base_prefix) or u.startswith(base_prefix_alt)

    def _normalize_to_input(u: str) -> str:
        """Convierte URLs del sitemap al mismo origen que el usuario indicó."""
        if u.startswith(base_prefix_alt):
            return parsed.scheme + "://" + netloc + u[len(base_prefix_alt):]
        return u

    filtered = [
        _normalize_to_input(u)
        for u in found_urls
        if _matches_prefix(u) and not _is_binary_url(u)
    ]

    # Deduplicar y limitar
    seen = set()
    result = []
    for u in filtered:
        if u not in seen:
            seen.add(u)
            result.append(u)
        if len(result) >= limit:
            break

    return result


def _is_binary_url(url: str) -> bool:
    """Retorna True si la URL apunta a un archivo binario/descargable."""
    SKIP_EXTENSIONS = (
        '.pdf', '.zip', '.rar', '.xlsx', '.xls', '.doc', '.docx',
        '.ppt', '.pptx', '.csv', '.xml', '.json', '.mp4', '.mp3',
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
        '.txt', '.rtf',
    )
    path_only = url.lower().split('?')[0]
    return any(path_only.endswith(ext) for ext in SKIP_EXTENSIONS)


class WebsiteCrawler:
    """
    Crawler de sitios web completos.
    Usa BFS para recorrer páginas del mismo dominio.
    """

    def __init__(self):
        # Umbral bajo para no descartar SPAs (React/Next.js) donde
        # el contenido llega después del render JS
        self.md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.35,
                threshold_type="dynamic",
                min_word_threshold=10,
            )
        )

    async def crawl(self, url: str) -> CrawlResult:
        """
        Crawlea un sitio web completo a partir de una URL raíz.

        Args:
            url: URL de entrada del sitio

        Returns:
            CrawlResult con todas las páginas extraídas
        """
        # Normalizar URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        domain = urlparse(url).netloc
        result = CrawlResult(base_url=url, domain=domain)

        console.print(f"\n[bold cyan]🕷  Iniciando crawling de:[/bold cyan] [yellow]{url}[/yellow]")
        console.print(f"[dim]Profundidad máx: {config.MAX_DEPTH} | Páginas máx: {config.MAX_PAGES_PER_SITE}[/dim]\n")

        # Extensiones a ignorar (archivos binarios/descargables)
        SKIP_EXTENSIONS = (
            '.pdf', '.zip', '.rar', '.xlsx', '.xls', '.doc', '.docx',
            '.ppt', '.pptx', '.csv', '.xml', '.json', '.mp4', '.mp3',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
            '.txt', '.rtf',
        )

        def should_skip_url(u: str) -> bool:
            path_only = u.lower().split('?')[0]
            return any(path_only.endswith(ext) for ext in SKIP_EXTENSIONS)

        # Prefijo base para filtrar solo la sección indicada
        parsed = urlparse(url)
        base_prefix = parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip("/")

        # Filtro de prefijo compatible con Crawl4AI 0.9.x
        # FilterChain soporta tanto filtros sync como async
        from crawl4ai.deep_crawling.filters import URLFilter, FilterChain

        class PrefixFilter(URLFilter):
            def __init__(self, prefix: str):
                super().__init__()
                self._prefix = prefix

            def apply(self, url: str) -> bool:
                result = url.startswith(self._prefix)
                self._update_stats(result)
                return result

        filter_chain = FilterChain([PrefixFilter(base_prefix)])

        # Estrategia de crawling BFS
        deep_strategy = BFSDeepCrawlStrategy(
            max_depth=config.MAX_DEPTH,
            max_pages=config.MAX_PAGES_PER_SITE,
            include_external=False,
            filter_chain=filter_chain,
        )

        # JavaScript que hace scroll completo para activar lazy loading
        # y espera a que todos los componentes React/Vue terminen de renderizar
        scroll_js = """
        async () => {
            const sleep = ms => new Promise(r => setTimeout(r, ms));
            // Scroll progresivo para disparar lazy loading
            let last = 0;
            while (true) {
                window.scrollTo(0, document.body.scrollHeight);
                await sleep(600);
                if (document.body.scrollHeight === last) break;
                last = document.body.scrollHeight;
            }
            window.scrollTo(0, 0);
            await sleep(500);
        }
        """

        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_strategy,
            markdown_generator=self.md_generator,
            # BYPASS para no usar caché stale de intentos anteriores
            cache_mode=CacheMode.BYPASS,
            wait_for_images=False,
            exclude_external_images=True,
            exclude_social_media_links=True,
            process_iframes=False,
            remove_overlay_elements=True,
            # Más tiempo para SPAs (React/Next.js/Vue) que renderizan con JS
            page_timeout=60000,
            # Espera a que el body tenga contenido real antes de extraer
            wait_for="css:body",
            # Scroll completo para activar lazy loading en SPAs
            js_code=scroll_js,
            # Pausa extra tras el render JS para contenido dinámico
            delay_before_return_html=2.5,
            simulate_user=True,
            magic=True,
        )

        # ── Estrategia 1: Sitemap XML ─────────────────────────────────────────
        # Para SPAs (React/Next.js/Vue) el BFS no puede seguir rutas JS.
        # El sitemap.xml lista todas las URLs directamente.
        sitemap_urls = await discover_urls_from_sitemap(url, limit=config.MAX_PAGES_PER_SITE)
        use_sitemap = len(sitemap_urls) > 0

        if use_sitemap:
            console.print(
                f"[bold green]🗺  Usando sitemap:[/bold green] "
                f"[cyan]{len(sitemap_urls)} URLs descubiertas[/cyan]\n"
            )
        else:
            console.print(
                "[dim yellow]⚠  Sin sitemap — usando BFS sobre links HTML[/dim yellow]\n"
            )

        pages_found = 0

        async def _process_page(page, seen_urls: set) -> Optional[PageResult]:
            """Procesa una página crawleada y retorna PageResult o None."""
            nonlocal pages_found
            pages_found += 1

            if should_skip_url(page.url):
                return None
            if not page.url.startswith(base_prefix):
                return None
            if not page.success:
                result.failed_urls.append(page.url)
                return None
            if page.url in seen_urls:
                return None
            seen_urls.add(page.url)

            # Extraer markdown — para SPAs preferir raw si fit está vacío
            raw_md = ""
            if page.markdown:
                fit_md = ""
                raw_md_full = ""
                if hasattr(page.markdown, 'fit_markdown') and page.markdown.fit_markdown:
                    fit_md = page.markdown.fit_markdown
                if hasattr(page.markdown, 'raw_markdown') and page.markdown.raw_markdown:
                    raw_md_full = page.markdown.raw_markdown
                if isinstance(page.markdown, str):
                    raw_md_full = page.markdown
                raw_md = fit_md if len(fit_md.split()) >= 30 else (raw_md_full or fit_md)

            cleaned = clean_markdown(raw_md)
            if not is_useful_content(cleaned):
                return None

            return PageResult(
                url=page.url,
                title=extract_title(cleaned, page.url),
                markdown=cleaned,
                word_count=len(cleaned.split()),
                status_code=page.status_code or 200,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            total_expected = len(sitemap_urls) if use_sitemap else config.MAX_PAGES_PER_SITE
            task = progress.add_task("[cyan]Crawleando páginas...", total=total_expected)
            seen_urls: set[str] = set()

            async with AsyncWebCrawler() as crawler:

                if use_sitemap:
                    # Crawlear cada URL del sitemap individualmente
                    # Configuración sin deep crawl (ya tenemos las URLs)
                    single_config = CrawlerRunConfig(
                        markdown_generator=self.md_generator,
                        cache_mode=CacheMode.BYPASS,
                        wait_for_images=False,
                        exclude_external_images=True,
                        exclude_social_media_links=True,
                        process_iframes=False,
                        remove_overlay_elements=True,
                        page_timeout=60000,
                        wait_for="css:body",
                        js_code=scroll_js,
                        delay_before_return_html=2.5,
                        simulate_user=True,
                        magic=True,
                    )
                    for page_url in sitemap_urls[:config.MAX_PAGES_PER_SITE]:
                        try:
                            page = await crawler.arun(url=page_url, config=single_config)
                            progress.update(
                                task, advance=1,
                                description=f"[cyan]Procesando {pages_found + 1}/{total_expected}..."
                            )
                            page_result = await _process_page(page, seen_urls)
                            if page_result:
                                result.pages.append(page_result)
                                result.total_words += page_result.word_count
                        except Exception as e:
                            result.failed_urls.append(page_url)
                            pages_found += 1

                else:
                    # Fallback: BFS sobre links HTML
                    pages = await crawler.arun(url=url, config=run_config)
                    for page in pages:
                        progress.update(
                            task, advance=1,
                            description=f"[cyan]Procesando {pages_found + 1} páginas..."
                        )
                        page_result = await _process_page(page, seen_urls)
                        if page_result:
                            result.pages.append(page_result)
                            result.total_words += page_result.word_count

        console.print(f"\n[bold green]✅ Crawling completado[/bold green]")
        console.print(f"   Estrategia: [cyan]{'Sitemap XML' if use_sitemap else 'BFS links'}[/cyan]")
        console.print(f"   Páginas útiles: [bold]{result.success_count()}[/bold]")
        console.print(f"   Páginas fallidas: [bold red]{len(result.failed_urls)}[/bold red]")
        console.print(f"   Total palabras: [bold]{result.total_words:,}[/bold]\n")

        return result


async def crawl_website(url: str) -> CrawlResult:
    """Función de conveniencia para crawlear desde fuera del módulo."""
    crawler = WebsiteCrawler()
    return await crawler.crawl(url)
