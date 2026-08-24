# 🤖 Web Knowledge Agent

Sistema que convierte cualquier sitio web en una base de conocimiento lista para entrenar un agente de WhatsApp (o cualquier otro chatbot).

## Arquitectura

```
URL del cliente
     ↓
Crawl4AI (crawling profundo + limpieza)
     ↓
Chunker inteligente (respeta secciones, overlap semántico)
     ↓
sentence-transformers (embeddings locales, sin costo)
     ↓
ChromaDB (vector store persistente)
     ↓
OpenRouter LLM (RAG para respuestas)
```

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar Playwright (requerido por Crawl4AI)

```bash
crawl4ai-setup
# o manualmente:
playwright install chromium
```

### 3. Configurar API key

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key de OpenRouter:
```
OPENROUTER_API_KEY=sk-or-...
```

Obtén tu key en: https://openrouter.ai/keys

### 4. Verificar instalación

```bash
python setup.py
```

---

## Uso

### Indexar un sitio web

```bash
python main.py index https://miempresa.com
```

Esto:
- Crawlea hasta 50 páginas del sitio (configurable en `.env`)
- Extrae el contenido limpio en Markdown
- Lo divide en chunks semánticos
- Genera embeddings localmente
- Los guarda en ChromaDB

### Chatear con la base de conocimiento

```bash
python main.py ask miempresa.com
```

Comandos dentro del chat:
- `/fuentes` — ver las fuentes del último resultado
- `/limpiar` — limpiar historial de conversación
- `/salir` — terminar

Usar un modelo específico:
```bash
python main.py ask miempresa.com --model google/gemini-flash-1.5
```

### Ver sitios indexados

```bash
python main.py list
```

### Ver estadísticas de un sitio

```bash
python main.py stats miempresa.com
```

### Exportar la base de conocimiento

```bash
# Exportar todos los formatos
python main.py export miempresa.com

# Solo un formato
python main.py export miempresa.com --format markdown
python main.py export miempresa.com --format jsonl
python main.py export miempresa.com --format finetune  # formato OpenAI fine-tuning
python main.py export miempresa.com --format csv
```

### Limpiar y re-indexar

```bash
python main.py clear miempresa.com
python main.py index https://miempresa.com
```

---

## Configuración (.env)

| Variable | Default | Descripción |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Tu API key de OpenRouter (requerida) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Modelo LLM a usar |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modelo de embeddings local |
| `MAX_PAGES_PER_SITE` | `50` | Máximo de páginas a crawlear |
| `MAX_DEPTH` | `3` | Profundidad máxima del crawling |
| `CONCURRENT_REQUESTS` | `5` | Peticiones paralelas |
| `CHUNK_SIZE` | `800` | Tokens por chunk |
| `CHUNK_OVERLAP` | `100` | Overlap entre chunks |
| `TOP_K_RESULTS` | `5` | Resultados RAG por query |

## Modelos recomendados (OpenRouter)

| Modelo | Costo | Velocidad | Calidad |
|---|---|---|---|
| `openai/gpt-4o-mini` | $0.15/1M | Rápido | ⭐⭐⭐⭐ |
| `google/gemini-flash-1.5` | $0.075/1M | Muy rápido | ⭐⭐⭐⭐ |
| `anthropic/claude-3-haiku` | $0.25/1M | Rápido | ⭐⭐⭐⭐ |
| `meta-llama/llama-3.1-8b-instruct:free` | Gratis | Moderado | ⭐⭐⭐ |

## Integración con WhatsApp

El sistema expone una `KnowledgeBase` y un `RAGAgent` que puedes importar directamente en tu bot de WhatsApp:

```python
from knowledge_base import KnowledgeBase
from agent import RAGAgent

# Cargar knowledge base existente
kb = KnowledgeBase("miempresa.com")
agent = RAGAgent(kb)

# En el handler de mensajes de WhatsApp:
def handle_whatsapp_message(message: str) -> str:
    response = agent.ask(message, stream=False)
    return response.answer
```

## Estructura del proyecto

```
crawler/
├── main.py           # CLI principal
├── crawler.py        # Motor de crawling (Crawl4AI)
├── chunker.py        # División semántica de texto
├── embeddings.py     # Embeddings locales (sentence-transformers)
├── vector_store.py   # ChromaDB wrapper
├── knowledge_base.py # Pipeline principal
├── agent.py          # Agente RAG
├── ai_client.py      # Cliente OpenRouter
├── exporter.py       # Exportación de datos
├── config.py         # Configuración centralizada
├── setup.py          # Verificación del entorno
├── requirements.txt
├── .env.example
└── chroma_db/        # Base de datos vectorial (auto-generado)
```
