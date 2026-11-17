#!/usr/bin/env python3
"""CLI: ответ на вопрос по видеотранскрипту (RAG: search + rerank + LLM).

Примеры:
  python ask.py "Что автор говорит про витамин А и печень?"
  python ask.py "йод" --top 3 --no-rerank
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import rag


def _fmt(seconds) -> str:
    if seconds is None:
        return "??:??"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--top", type=int, default=config.HYBRID_FINAL_K)
    ap.add_argument("--no-rerank", action="store_true", help="отключить реранкер")
    args = ap.parse_args()

    if not config.LLM_API_KEY:
        print("Ошибка: не задан LLM_API_KEY (см. .env.example).")
        sys.exit(1)

    question = " ".join(args.question)
    use_rerank = not args.no_rerank
    answer, results = rag.answer(question, top_k=args.top, use_rerank=use_rerank)

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("\nИсточники:")
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score"))
        print(f"  [{i}] {_fmt(r['start'])}–{_fmt(r['end'])} "
              f"(score={score:.3f}) — {r['text'][:120]}...")


if __name__ == "__main__":
    main()
