#!/usr/bin/env python3
"""Общая конфигурация конвейера подготовки данных и загрузки в Qdrant.

Все секреты и настройки читаются из переменных окружения и/или файла `.env`
(формат `КЛЮЧ=ЗНАЧЕНИЕ`). Парсер .env написан вручную, чтобы не тянуть
лишние зависимости. Файл `.env` загружается ПЕРЕД чтением настроек.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path | None = None) -> None:
    """Минимальный загрузчик .env: ключ=значение (без разбора кавычек)."""
    env_file = path or (Path(__file__).resolve().parent.parent / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ----------------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRANSCRIPT_LOG = ROOT / "transcript.log"      # лог с [start -> end] строками
TRANSCRIPT_TXT = ROOT / "audio.small.txt"     # текст без таймкодов (запасной)

SEGMENTS_PATH = DATA_DIR / "segments.jsonl"   # {start, end, text}
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"       # чанки с метаданными
VOCAB_PATH = DATA_DIR / "bm25_stats.json"     # vocab + idf + avgdl

# ----------------------------------------------------------------------------
# Эмбеддинги (dense)
# ----------------------------------------------------------------------------
# multilingual-e5 требует префиксы "query: " / "passage: " — это важно!
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768
EMBEDDING_MAX_TOKENS = 512
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# ----------------------------------------------------------------------------
# Чанкинг
# ----------------------------------------------------------------------------
CHUNK_TARGET_TOKENS = 350            # целевой размер чанка (токены)
CHUNK_MAX_TOKENS = 460               # жёсткий потолок (запас под префикс)
OVERLAP_SENTENCES = 1                # перенос последних предложений в след. чанк
MIN_SENTENCE_LEN = 2                 # отбрасывать «обрывки» короче N символов

# ----------------------------------------------------------------------------
# Qdrant
# ----------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "transcript_skin")
QDRANT_UPSERT_BATCH = 64
QDRANT_VECTOR_NAME_DENSE = "dense"
QDRANT_VECTOR_NAME_SPARSE = "sparse"

# ----------------------------------------------------------------------------
# BM25 (sparse)
# ----------------------------------------------------------------------------
BM25_K1 = 1.2
BM25_B = 0.75

# ----------------------------------------------------------------------------
# Источник видео (метаданные чанков)
# ----------------------------------------------------------------------------
VIDEO_URL = os.getenv("VIDEO_URL", "https://www.youtube.com/watch?v=ovfvFJbXiAk")
VIDEO_ID = os.getenv("VIDEO_ID", "ovfvFJbXiAk")
VIDEO_TITLE = os.getenv("VIDEO_TITLE", "")

# ----------------------------------------------------------------------------
# Hybrid-поиск
# ----------------------------------------------------------------------------
HYBRID_TOP_K = 10                   # сколько кандидатов в каждом префетче
HYBRID_FINAL_K = 5                   # сколько вернуть после RRF-слияния

# ----------------------------------------------------------------------------
# Реранкер (cross-encoder) — повышает точность выдачи перед LLM
# ----------------------------------------------------------------------------
RERANK_ENABLED = False   # выключен для Telegram-бота: на CPU реранкер ~76s на запрос
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"    # мультиязычный, устойчив к вопросам
RERANK_CANDIDATES = 20              # сколько кандидатов из hybrid подавать реранкеру
RERANK_TOP_K = 5                    # сколько вернуть после реранка

# ----------------------------------------------------------------------------
# HTTP-сервис (для интеграции с n8n / Telegram-ботом)
# ----------------------------------------------------------------------------
SERVE_HOST = os.getenv("SERVE_HOST", "127.0.0.1")
SERVE_PORT = int(os.getenv("SERVE_PORT", "8000"))

# ----------------------------------------------------------------------------
# PostgreSQL (локальный каталог роликов)
# ----------------------------------------------------------------------------
PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "bot_db")
PG_USER = os.getenv("PG_USER", "bot")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

# ----------------------------------------------------------------------------
# LLM для ответов (через OpenRouter — OpenAI-совместимый шлюз)
# По умолчанию DeepSeek. Любой другой провайдер — просто смените URL и модель.
# ----------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
# Варианты DeepSeek на OpenRouter:
#   deepseek/deepseek-chat            — DeepSeek V3 (дёшево, быстро)
#   deepseek/deepseek-r1             — reasoning-модель
#   deepseek/deepseek-v4-pro-0813    — флагман V4 Pro (дороже)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
# Ограничение длины ответа (токены) — чем меньше, тем быстрее генерация.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
# Показывать ли в контексте метки «[Фрагмент N | таймкод]» (модель цитирует их в ответе).
CONTEXT_SHOW_TIMECODES = os.getenv("CONTEXT_SHOW_TIMECODES", "false").lower() in ("1", "true", "yes")
