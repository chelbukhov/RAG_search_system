#!/usr/bin/env python3
"""Индексация чанков в Qdrant (dense + sparse для гибридного поиска).

Шаги:
  1. Читаем data/chunks.jsonl.
  2. Считаем dense-эмбеддинги (multilingual-e5, prefix "passage: ").
  3. Строим BM25 (sparse) и сохраняем статистику на диск.
  4. Создаём коллекцию (векторы: "dense" 768d Cosine + "sparse" BM25).
  5. Загружаем точки батчами с метаданными (таймкоды, индексы, источник).
  6. Создаём payload-индексы для фильтрации.

Требует QDRANT_URL (и обычно QDRANT_API_KEY) — см. .env.example.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import embeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier, PointStruct, SparseVector,
    PayloadSchemaType,
)


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_video_title(url: str) -> str:
    """Опционально получаем название видео; при ошибке возвращаем ''."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title") or ""
    except Exception:
        return ""


def build_payload(chunk: dict, meta: dict) -> dict:
    return {
        "text": chunk["text"],
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "chunk_index": chunk.get("chunk_index"),
        "token_count": chunk.get("token_count"),
        "char_count": chunk.get("char_count"),
        "sentence_count": chunk.get("sentence_count"),
        "video_id": meta.get("video_id"),
        "title": meta.get("title"),
        "url": meta.get("url"),
        "model": meta.get("model"),
        "language": meta.get("language", "ru"),
        "created_at": meta.get("created_at"),
    }


def main():
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else str(config.CHUNKS_PATH)
    chunks = load_chunks(chunks_path)
    if not chunks:
        print("Нет чанков. Запустите chunk.py сначала.")
        sys.exit(1)
    print(f"Чанков: {len(chunks)}")

    if not config.QDRANT_URL:
        print("Ошибка: не задан QDRANT_URL. Заполните .env (см. .env.example).")
        sys.exit(1)

    # Метаданные источника.
    meta = {
        "video_id": os.getenv("VIDEO_ID", ""),
        "url": os.getenv("VIDEO_URL", config.VIDEO_URL),
        "title": os.getenv("VIDEO_TITLE", ""),
        "model": config.EMBEDDING_MODEL,
        "language": "ru",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not meta["title"] and meta["url"]:
        print("  Получаю название видео...")
        meta["title"] = fetch_video_title(meta["url"]) or ""

    # 1) Dense.
    print("Считаю dense-эмбеддинги...")
    texts = [c["text"] for c in chunks]
    dense = embeddings.embed_passages(texts)

    # 2) Sparse.
    print("Строю BM25 (sparse)...")
    bm = embeddings.build_sparse(texts)
    sparse_list = [bm.doc_sparse(i) for i in range(len(texts))]

    # 3) Qdrant.
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            config.QDRANT_VECTOR_NAME_DENSE: VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            config.QDRANT_VECTOR_NAME_SPARSE: SparseVectorParams(modifier=Modifier.NONE),
        },
    )
    print(f"Коллекция '{config.QDRANT_COLLECTION}' создана.")

    # 4) Upsert батчами.
    for start in range(0, len(chunks), config.QDRANT_UPSERT_BATCH):
        batch = []
        for i in range(start, min(start + config.QDRANT_UPSERT_BATCH, len(chunks))):
            sp = sparse_list[i]
            batch.append(PointStruct(
                id=i,
                vector={
                    config.QDRANT_VECTOR_NAME_DENSE: dense[i],
                    config.QDRANT_VECTOR_NAME_SPARSE: SparseVector(
                        indices=sp["indices"], values=sp["values"]),
                },
                payload=build_payload(chunks[i], meta),
            ))
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=batch, wait=True)
        print(f"  загружено {min(start + config.QDRANT_UPSERT_BATCH, len(chunks))}/{len(chunks)}")

    # 5) Payload-индексы для фильтрации.
    client.create_payload_index(config.QDRANT_COLLECTION, "chunk_index", PayloadSchemaType.INTEGER)
    client.create_payload_index(config.QDRANT_COLLECTION, "start", PayloadSchemaType.FLOAT)
    client.create_payload_index(config.QDRANT_COLLECTION, "end", PayloadSchemaType.FLOAT)
    client.create_payload_index(config.QDRANT_COLLECTION, "video_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "title", PayloadSchemaType.KEYWORD)
    print("Payload-индексы созданы.")

    info = client.get_collection(config.QDRANT_COLLECTION)
    print(f"Готово. Точек в коллекции: {info.points_count}")


if __name__ == "__main__":
    main()
