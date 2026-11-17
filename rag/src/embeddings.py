#!/usr/bin/env python3
"""Эмбеддинги: dense (multilingual-e5) + sparse (BM25).

Важные нюансы:
  * e5 требует префиксы "query: " / "passage: " — без них качество резко падает.
  * Dense-векторы L2-нормализованы (для метрики Cosine в Qdrant).
  * Sparse-словарь (term -> id) фиксируется по корпусу и переиспользуется
    при запросе (иначе id не совпадут).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import bm25


_dense_model = None


def get_dense_model():
    """Ленивая загрузка dense-модели (кешируется в процессе)."""
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Загружаю модель эмбеддингов: {config.EMBEDDING_MODEL}")
        _dense_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _dense_model


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = get_dense_model()
    prefixed = [config.PASSAGE_PREFIX + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, batch_size=32,
                        show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    model = get_dense_model()
    return model.encode([config.QUERY_PREFIX + query],
                        normalize_embeddings=True).tolist()[0]


def build_sparse(corpus: list[str]) -> bm25.BM25:
    """Строим BM25 по корпусу и сохраняем статистику (vocab + idf)."""
    bm = bm25.BM25(corpus)
    bm.save_stats()
    print(f"  BM25: словарь {len(bm.vocab)} терминов")
    return bm


def query_sparse(query: str) -> dict:
    """Разреженный вектор запроса; те же vocab/idf, что и при индексации."""
    stats = bm25.load_stats()
    if not stats:
        return {"indices": [], "values": []}
    bm = bm25.BM25([], vocab=stats["vocab"], idf=stats["idf"])
    return bm.query_sparse(query)
