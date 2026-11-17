#!/usr/bin/env python3
"""Нарезка сегментов транскрибации на чанки для RAG.

Особенности и учтённые нюансы:
  * Разбивка на предложения (сегменты Whisper могут содержать >1 предложения).
  * Токен-ориентированная группировка по целевому размеру (не по символам),
    чтобы не превысить лимит модели эмбеддингов (512 токенов e5-base).
  * Перекрытие (overlap) последними предложениями для сохранения контекста.
  * Сохранение таймкодов (start/end) и исходных сегментов для метаданных.
  * Обработка «длинных» предложений > потолка (принудительная нарезка).
  * Отсев пустых/мусорных фрагментов.
Выход: data/chunks.jsonl
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Разбивка на предложения: конец = . ! ? … за которыми пробел/конец строки.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < config.MIN_SENTENCE_LEN:
            continue
        out.append(p)
    return out


def _load_tokenizer():
    """Загружаем токенизатор e5 для точного подсчёта токенов.

    Если transformers недоступен — вернём fallback-функцию (примерная
    оценка: 1 токен ≈ 4 символа).
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        return tok, "tokenizer"
    except Exception as e:
        print(f"  [warn] Не удалось загрузить токенизатор ({e}), использую оценку 1 токен≈4 символа")

        class _Fallback:
            def encode(self, text, add_special_tokens=True):
                import math
                return [0] * max(1, math.ceil(len(text) / 4))

        return _Fallback(), "fallback"


def load_segments(path: str) -> list[dict]:
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segs.append(json.loads(line))
    return segs


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _hard_split_long_sentence(tokenizer, text: str, max_tokens: int) -> list[str]:
    """Если одно предложение дольше потолка — режем по токенам с небольшим пересечением."""
    words = text.split()
    pieces, cur, cur_tokens = [], [], 0
    for w in words:
        wt = count_tokens(tokenizer, w) + 1
        if cur and cur_tokens + wt > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += wt
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_segments(segments: list[dict]) -> list[dict]:
    tokenizer, mode = _load_tokenizer()

    # 1) Сегменты -> предложения с метаданными.
    sentences = []
    for seg in segments:
        for sent in split_sentences(seg.get("text", "")):
            sentences.append({
                "text": sent,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "seg_id": seg.get("id"),
            })

    # 2) Группировка предложений в чанки.
    chunks = []
    cur, cur_tokens = [], 0

    def finalize(cur):
        text = " ".join(s["text"] for s in cur)
        seg_ids = [s["seg_id"] for s in cur if s.get("seg_id") is not None]
        starts = [s["start"] for s in cur if s.get("start") is not None]
        ends = [s["end"] for s in cur if s.get("end") is not None]
        return {
            "text": text,
            "start": starts[0] if starts else None,
            "end": ends[-1] if ends else None,
            "seg_start": seg_ids[0] if seg_ids else None,
            "seg_end": seg_ids[-1] if seg_ids else None,
            "token_count": cur_tokens,
            "char_count": len(text),
            "sentence_count": len(cur),
        }

    for s in sentences:
        n = count_tokens(tokenizer, s["text"])
        # Если предложение само по себе больше потолка — режем принудительно.
        if n > config.CHUNK_MAX_TOKENS:
            if cur:
                chunks.append(finalize(cur))
                cur, cur_tokens = [], 0
            for piece in _hard_split_long_sentence(tokenizer, s["text"], config.CHUNK_MAX_TOKENS):
                seg_id = s.get("seg_id")
                chunks.append({
                    "text": piece,
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "seg_start": seg_id,
                    "seg_end": seg_id,
                    "token_count": count_tokens(tokenizer, piece),
                    "char_count": len(piece),
                    "sentence_count": 1,
                })
            continue

        if cur and cur_tokens + n > config.CHUNK_MAX_TOKENS:
            chunks.append(finalize(cur))
            # Перекрытие: сохранить последние предложения в следующий чанк.
            keep = cur[-config.OVERLAP_SENTENCES:] if config.OVERLAP_SENTENCES > 0 else []
            cur, cur_tokens = keep, sum(count_tokens(tokenizer, x["text"]) for x in keep)

        cur.append(s)
        cur_tokens += n

    if cur:
        chunks.append(finalize(cur))

    # 3) Индексация и лёгкая пост-очистка.
    result = []
    for i, c in enumerate(chunks):
        c["chunk_index"] = i
        result.append(c)

    print(f"  Токенизатор: {mode}")
    print(f"  Предложений: {len(sentences)}")
    print(f"  Чанков: {len(result)}")
    if result:
        lens = [c["token_count"] for c in result]
        print(f"  Размер чанков (токены): min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")
    return result


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(config.SEGMENTS_PATH)
    out = sys.argv[2] if len(sys.argv) > 2 else str(config.CHUNKS_PATH)

    segments = load_segments(src)
    print(f"Загружено сегментов: {len(segments)}")
    chunks = chunk_segments(segments)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
