#!/usr/bin/env python3
"""BM25 sparse-векторы для гибридного поиска в Qdrant.

Реализовано самостоятельно, чтобы полностью контролировать соотношение
term -> index и совместимость с Qdrant.

Требование Qdrant: sparse-вектор = {"indices": [int], "values": [float]},
где indices лежат в одном пространстве целочисленных id для всех точек.
Поэтому словарь term -> id, а также IDF и средняя длина документа,
вычисляются один раз по корпусу и сохраняются на диск, чтобы при запросе
использовать ровно те же id и веса IDF.
"""

import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Токен: буквы (латиница/кириллица/ё) и цифры, регистр не важен.
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1: float = None, b: float = None,
                 vocab: dict | None = None, idf: dict | None = None):
        self.k1 = k1 if k1 is not None else config.BM25_K1
        self.b = b if b is not None else config.BM25_B
        self.corpus = corpus
        self.doc_tokens = [tokenize(t) for t in corpus]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))

        self.vocab = vocab if vocab is not None else self._build_vocab()
        self.idf = idf if idf is not None else self._compute_idf()

    # -- построение ---------------------------------------------------------
    def _build_vocab(self) -> dict:
        vocab = {}
        for toks in self.doc_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        return vocab

    def _compute_idf(self) -> dict:
        n = len(self.doc_tokens)
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        return {
            t: math.log((n - c + 0.5) / (c + 0.5) + 1.0)
            for t, c in df.items()
        }

    # -- sparse-вектор документа (для upsert) --------------------------------
    def doc_sparse(self, doc_idx: int) -> dict:
        toks = self.doc_tokens[doc_idx]
        tf = Counter(toks)
        dl = self.doc_len[doc_idx]
        indices, values = [], []
        for t, f in tf.items():
            if t not in self.idf or t not in self.vocab:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score = self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                indices.append(self.vocab[t])
                values.append(float(score))
        return {"indices": indices, "values": values}

    # -- sparse-вектор запроса -------------------------------------------------
    def query_sparse(self, query: str) -> dict:
        tf = Counter(tokenize(query))
        indices, values = [], []
        for t, f in tf.items():
            if t not in self.idf or t not in self.vocab:
                continue  # термин отсутствует в корпусе — пропускаем
            denom = f + self.k1
            score = self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                indices.append(self.vocab[t])
                values.append(float(score))
        return {"indices": indices, "values": values}

    # -- сохранение/загрузка статистики --------------------------------------
    def save_stats(self, path: str = None):
        p = path or str(config.VOCAB_PATH)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return p


def load_stats(path: str = None) -> dict:
    p = path or str(config.VOCAB_PATH)
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
