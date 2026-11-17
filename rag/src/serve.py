#!/usr/bin/env python3
"""HTTP-сервис RAG — точка входа для n8n / Telegram-бота.

Позволяет внешним сценариям (n8n) задавать вопросы по транскрипту, не вникая
в детали гибридного поиска, реранкера и LLM.

Эндпоинты:
  GET  /health            — проверка живости.
  POST /search            — гибридный поиск (+реранк) без LLM.
  POST /ask               — полный RAG-ответ (поиск → реранк → LLM).

Запуск:
  python serve.py            # 127.0.0.1:8000
  SERVE_PORT=8080 python serve.py

Пример запроса к /ask:
  {"question": "Что помогает восстановить кожу?", "top_k": 5}

Ответ /ask:
  {"answer": "...", "sources": [{"start": 6341.4, "end": 6442.9,
     "time": "105:41–107:23", "text": "...", "score": 0.87}, ...]}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import rag
import search as search_mod

from fastapi import FastAPI
from pydantic import BaseModel, Field


def _fmt(seconds) -> str:
    if seconds is None:
        return "??:??"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _video_link(url: str, start) -> str:
    """Ссылка на конкретный момент видео (YouTube поддерживает &t=сек).

    Возвращает пустую строку, если нет URL или таймкода.
    """
    if not url or start is None:
        return ""
    s = int(start)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={s}s"


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=config.HYBRID_FINAL_K, ge=1, le=50)
    use_rerank: bool | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=config.HYBRID_FINAL_K, ge=1, le=50)
    use_rerank: bool | None = None
    telegram_id: int | None = None


app = FastAPI(title="Transcrypt RAG API", version="1.0.0")


@app.on_event("startup")
def _warmup():
    """Прогреваем модели, чтобы первый запрос был быстрым."""
    try:
        # Эмбеддинги и реранкер грузятся лениво; прогреем их заранее.
        import embeddings
        _ = embeddings.embed_query("")
        if config.RERANK_ENABLED:
            import reranker
            _ = reranker.get_reranker()
        print("Модели прогружены.")
    except Exception as e:
        print(f"[warn] прогрев моделей: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "collection": config.QDRANT_COLLECTION}


@app.get("/videos")
def videos():
    """Список уникальных видео (название + ссылка), лежащих в коллекции."""
    return {"videos": search_mod.list_videos()}


@app.post("/search")
def search(req: SearchRequest):
    results = rag.retrieve(req.question, top_k=req.top_k, use_rerank=req.use_rerank)
    return {
        "query": req.question,
        "results": [
            {
                "start": r.get("start"),
                "end": r.get("end"),
                "time": f"{_fmt(r.get('start'))}–{_fmt(r.get('end'))}",
                "text": r.get("text"),
                "score": r.get("rerank_score", r.get("score")),
                "url": r.get("url", ""),
                "link": _video_link(r.get("url", ""), r.get("start")),
            }
            for r in results
        ],
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not config.LLM_API_KEY:
        return {"error": "LLM_API_KEY не задан в .env"}, 500
    answer_text, results = rag.answer(req.question, top_k=req.top_k,
                                      use_rerank=req.use_rerank,
                                      telegram_id=req.telegram_id)
    return {
        "query": req.question,
        "answer": answer_text,
        "sources": [
            {
                "start": r.get("start"),
                "end": r.get("end"),
                "time": f"{_fmt(r.get('start'))}–{_fmt(r.get('end'))}",
                "text": r.get("text"),
                "score": r.get("rerank_score", r.get("score")),
                "url": r.get("url", ""),
                "link": _video_link(r.get("url", ""), r.get("start")),
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host=config.SERVE_HOST, port=config.SERVE_PORT,
                reload=False)
