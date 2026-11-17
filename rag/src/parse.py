#!/usr/bin/env python3
"""Разбор лога транскрибации в структурированные сегменты.

Вход:  текстовый файл со строками вида
         [  start ->   end]  текст
Выход: data/segments.jsonl  (по одному JSON-объекту на строку)
         {"id": 0, "start": 3.9, "end": 8.3, "text": "..."}

Также допускается вход без таймкодов (просто текст построчно) — тогда
таймкоды проставляются как null.
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Строки вида:  [ 6341.4s ->  6346.3s]  текст
TS_RE = re.compile(r"^\s*\[\s*([\d.]+)s?\s*->\s*([\d.]+)s?\s*\]\s*(.*)$")


def parse_log(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Определяем режим: лог с таймкодами или простой текст.
    has_ts = any(TS_RE.match(ln) for ln in lines)

    segments = []
    for line in lines:
        line = line.rstrip()
        m = TS_RE.match(line)
        if m:
            start = float(m.group(1))
            end = float(m.group(2))
            text = m.group(3).strip()
        elif has_ts:
            # В логе с таймкодами несовпадающие строки — служебные, пропускаем.
            continue
        else:
            text = line.strip()
            if not text:
                continue
            start = end = None
        if text:
            segments.append({
                "id": len(segments),
                "start": start,
                "end": end,
                "text": text,
            })
    return segments


def save_segments(segments: list[dict], out_path: str) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(config.TRANSCRIPT_LOG)
    out = sys.argv[2] if len(sys.argv) > 2 else str(config.SEGMENTS_PATH)

    if not os.path.isfile(src):
        print(f"Файл не найден: {src}")
        sys.exit(1)

    segments = parse_log(src)
    save_segments(segments, out)

    with_ts = sum(1 for s in segments if s["start"] is not None)
    print(f"Сегментов: {len(segments)} (с таймкодами: {with_ts})")
    if segments:
        print(f"Диапазон:  {segments[0]['start']}s .. {segments[-1]['end']}s")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
