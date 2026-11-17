#!/usr/bin/env python3
"""Гибридный поиск по коллекции: dense (e5) + sparse (BM25) через RRF.

Примеры:
  python search.py "что влияет на здоровье кожи"
  python search.py "витамин А печень" --top 5
  python search.py "йод" --from 0 --to 1800     (фильтр по таймкодам, сек)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import embeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector,
    Filter, FieldCondition, Range,
)


def get_client() -> QdrantClient:
    if not config.QDRANT_URL:
        print("Ошибка: не задан QDRANT_URL (см. .env.example).")
        sys.exit(1)
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None,
                        timeout=300)


def hybrid_search(query: str, top_k: int = None, start: float = None,
                  end: float = None) -> list[dict]:
    """Возвращает список [{score, text, start, end, chunk_index, ...}]."""
    top_k = top_k or config.HYBRID_FINAL_K
    client = get_client()

    dense_q = embeddings.embed_query(query)
    sparse_q = embeddings.query_sparse(query)
    sp_q = SparseVector(indices=sparse_q["indices"], values=sparse_q["values"])

    prefetch = [
        Prefetch(
            query=dense_q,
            using=config.QDRANT_VECTOR_NAME_DENSE,
            limit=max(top_k, config.HYBRID_TOP_K),
        ),
        Prefetch(
            query=sp_q,
            using=config.QDRANT_VECTOR_NAME_SPARSE,
            limit=max(top_k, config.HYBRID_TOP_K),
        ),
    ]

    # Опциональный фильтр по таймкодам.
    must = []
    if start is not None or end is not None:
        rng = Range(gte=start, lte=end)
        if start is None:
            rng = Range(lte=end)
        if end is None:
            rng = Range(gte=start)
        must.append(FieldCondition(key="start", range=rng))

    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
        query_filter=Filter(must=must) if must else None,
    )

    out = []
    for p in result.points:
        pl = p.payload or {}
        out.append({
            "id": p.id,
            "score": p.score,
            "text": pl.get("text", ""),
            "start": pl.get("start"),
            "end": pl.get("end"),
            "chunk_index": pl.get("chunk_index"),
            "url": pl.get("url", ""),
            "video_id": pl.get("video_id", ""),
        })
    return out


def _scroll_page(client, offset):
    """Одна страница scroll с retry (Qdrant Cloud иногда тормозит)."""
    import time as _time
    last = None
    for attempt in range(3):
        try:
            return client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=500,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 2:
                _time.sleep(3)
    raise last


def scan_collection() -> tuple[list[dict], dict]:
    """Один полный scroll по коллекции.

    Возвращает (videos, grouped):
      videos  — список уникальных видео [{title, url, video_id}], по названию;
      grouped — {video_id: [{text, chunk_index}, ...]}, отсортировано.
    """
    client = get_client()
    videos = {}
    grouped = {}
    offset = None
    while True:
        raw = _scroll_page(client, offset)
        # qdrant-client возвращает кортеж (points, next_page_offset) или объект.
        if isinstance(raw, tuple):
            points, offset = raw
        else:
            points, offset = raw.points, raw.next_page_offset

        if not points:
            break
        for p in points:
            pl = p.payload or {}
            vid = pl.get("video_id") or pl.get("url") or ""
            if not vid:
                continue
            if vid not in videos:
                videos[vid] = {
                    "title": pl.get("title") or "",
                    "url": pl.get("url") or "",
                    "video_id": vid,
                }
            grouped.setdefault(vid, []).append({
                "text": pl.get("text", ""),
                "chunk_index": pl.get("chunk_index", 0),
            })
        if offset is None:
            break

    for vid in grouped:
        grouped[vid].sort(key=lambda c: c["chunk_index"] if c["chunk_index"] is not None else 0)

    vlist = sorted(videos.values(), key=lambda v: (v["title"] or "").lower())
    return vlist, grouped


def _videos_cache_reset():
    return {"data": None, "ts": 0.0}


_videos_cache = _videos_cache_reset()


def list_videos(ttl: float = 60.0) -> list[dict]:
    """Список уникальных видео [{title, url, video_id}], с кэшем."""
    import time as _time
    now = _time.time()
    if _videos_cache["data"] is not None and now - _videos_cache["ts"] < ttl:
        return _videos_cache["data"]
    vlist, _ = scan_collection()
    _videos_cache["data"] = vlist
    _videos_cache["ts"] = now
    return vlist


def _fmt(seconds) -> str:
    if seconds is None:
        return "??:??"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=config.HYBRID_FINAL_K)
    ap.add_argument("--from", dest="start", type=float, help="с секунды")
    ap.add_argument("--to", dest="end", type=float, help="до секунды")
    ap.add_argument("--rerank", action="store_true",
                    help="применить cross-encoder реранкер")
    args = ap.parse_args()

    if args.rerank:
        import reranker
        cands = hybrid_search(args.query, top_k=config.RERANK_CANDIDATES,
                              start=args.start, end=args.end)
        results = reranker.rerank(args.query, cands)[:args.top]
        print(f"Гибридный поиск + реранкер ({len(results)}):\n")
    else:
        results = hybrid_search(args.query, top_k=args.top,
                                start=args.start, end=args.end)
        print(f"Результаты гибридного поиска ({len(results)}):\n")

    if not results:
        print("Ничего не найдено.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score"))
        tag = "rerank" if "rerank_score" in r else "hybrid"
        print(f"[{i}] [{_fmt(r['start'])}-{_fmt(r['end'])}] "
              f"chunk={r['chunk_index']} {tag}={score:.4f}")
        print(f"    {r['text']}\n")


if __name__ == "__main__":
    main()
