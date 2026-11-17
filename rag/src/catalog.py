#!/usr/bin/env python3
"""Локальный каталог роликов (PostgreSQL).

Таблица `videos` хранит кэш: video_id, title, description, url, chunk_count.
Пополняется командой /update_content (LLM-суммаризация), читается при
каталоговых запросах бота — без обращения к Qdrant и LLM.

Каждая операция открывает отдельное подключение и закрывает его — так не
бывает «протухших» соединений после рестарта Postgres.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import psycopg2
import psycopg2.extras

_TABLE = "videos"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    video_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _connect():
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        connect_timeout=10,
    )


def ensure_table():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        conn.commit()
    finally:
        conn.close()


def upsert_videos(rows: list[dict]):
    """Вставляет/обновляет ролики. rows: [{video_id, title, description, url, chunk_count}]."""
    if not rows:
        return
    ensure_table()
    sql = f"""
        INSERT INTO {_TABLE} (video_id, title, description, url, chunk_count, updated_at)
        VALUES (%(video_id)s, %(title)s, %(description)s, %(url)s, %(chunk_count)s, now())
        ON CONFLICT (video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            url = EXCLUDED.url,
            chunk_count = EXCLUDED.chunk_count,
            updated_at = now()
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)
        conn.commit()
    finally:
        conn.close()


def list_videos() -> list[dict]:
    """Все ролики из каталога, отсортированные по названию."""
    ensure_table()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT video_id, title, description, url, chunk_count FROM {_TABLE} ORDER BY title"
            )
            return cur.fetchall()
    finally:
        conn.close()


def count_videos() -> int:
    ensure_table()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {_TABLE}")
            return cur.fetchone()[0]
    finally:
        conn.close()


def find_video(hint: str) -> list[dict]:
    """Ищет ролики по подстроке в названии (ILIKE)."""
    ensure_table()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT video_id, title, description, url, chunk_count "
                f"FROM {_TABLE} WHERE title ILIKE %s ORDER BY title",
                (f"%{hint}%",),
            )
            return cur.fetchall()
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# История диалога (messages)
# ----------------------------------------------------------------------------
_MESSAGES_TABLE = "messages"
_HISTORY_LIMIT = 10

_CREATE_MESSAGES = f"""
CREATE TABLE IF NOT EXISTS {_MESSAGES_TABLE} (
    id          BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL,
    role        TEXT NOT NULL,           -- 'user' | 'assistant'
    text        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def ensure_messages_table():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_MESSAGES)
        conn.commit()
    finally:
        conn.close()


def save_message(telegram_id: int, role: str, text: str):
    """Сохраняет сообщение и оставляет только последние N на пользователя."""
    if telegram_id is None or not text:
        return
    ensure_messages_table()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {_MESSAGES_TABLE} (telegram_id, role, text) VALUES (%s, %s, %s)",
                (telegram_id, role, text),
            )
            # удаляем всё, что выходит за последние N
            cur.execute(
                f"DELETE FROM {_MESSAGES_TABLE} WHERE id IN ("
                f"  SELECT id FROM {_MESSAGES_TABLE} WHERE telegram_id=%s "
                f"  ORDER BY created_at DESC, id DESC OFFSET %s"
                f")",
                (telegram_id, _HISTORY_LIMIT),
            )
        conn.commit()
    finally:
        conn.close()


def load_history(telegram_id: int, limit: int = _HISTORY_LIMIT) -> list[dict]:
    """Последние `limit` сообщений пользователя в хронологическом порядке."""
    if telegram_id is None:
        return []
    ensure_messages_table()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT role, text FROM ("
                f"  SELECT role, text, created_at, id FROM {_MESSAGES_TABLE} "
                f"  WHERE telegram_id=%s ORDER BY created_at DESC, id DESC LIMIT %s"
                f") t ORDER BY created_at ASC, id ASC",
                (telegram_id, limit),
            )
            return cur.fetchall()
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Статистика для команды /users
# ----------------------------------------------------------------------------
def count_users() -> int:
    """Количество зарегистрированных пользователей."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM users")
            return cur.fetchone()[0]
    finally:
        conn.close()


def get_last_messages(limit: int = 2) -> list[dict]:
    """Последние `limit` сообщений (по всем пользователям), хронологически."""
    ensure_messages_table()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT telegram_id, role, text FROM ("
                f"  SELECT telegram_id, role, text, id FROM {_MESSAGES_TABLE} "
                f"  ORDER BY id DESC LIMIT %s"
                f") t ORDER BY id ASC",
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def get_user_info(telegram_id: int) -> dict | None:
    """Данные пользователя: username, first_name, last_name."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT telegram_id, username, first_name, last_name FROM users WHERE telegram_id=%s",
                (telegram_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()
