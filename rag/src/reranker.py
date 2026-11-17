#!/usr/bin/env python3
"""Реранкер (cross-encoder) для уточнения порядка выдачи.

После того как гибридный поиск (dense+sparse, RRF) отобрал N кандидатов,
cross-encoder оценивает релевантность каждой пары (вопрос, чанк) напрямую —
это заметно повышает точность финального топ-К.

Модель: BAAI/bge-reranker-v2-m3 (мультиязычная, устойчива к естественным
вопросам, в отличие от bge-reranker-base, который «схлопывает» скор на
вопросах вида «Что автор говорит про…»).
Загружается лениво и кешируется в процессе.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"  Загружаю реранкер: {config.RERANK_MODEL}")
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def rerank(query: str, docs: list[dict]) -> list[dict]:
    """Возвращает docs, отсортированные по релевантности, с полем rerank_score."""
    if not docs:
        return docs
    model = get_reranker()
    pairs = [(query, d["text"]) for d in docs]
    scores = model.predict(pairs)
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)
    docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return docs
