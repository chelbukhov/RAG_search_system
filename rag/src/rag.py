#!/usr/bin/env python3
"""Высокоуровневый RAG-слой: поиск (+реранк) и генерация ответа.

Это общий слой для CLI (`ask.py`) и HTTP-сервиса (`serve.py`), а значит —
и для будущего n8n-сценария (Telegram-бот будет дёргать HTTP-эндпоинт).

Функции:
  * retrieve(query) — гибридный поиск + реранкер → топ-чанки.
  * answer(query)  — retrieve + промпт + LLM → (answer, sources).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import search as search_mod
import reranker as reranker_mod
import catalog


def retrieve(query: str, top_k: int = None, use_rerank: bool = None) -> list[dict]:
    """Гибридный поиск (+ опциональный реранк) → список чанков.

    Возвращает [{text, start, end, chunk_index, score, rerank_score?}].
    """
    top_k = top_k or config.HYBRID_FINAL_K
    use_rerank = config.RERANK_ENABLED if use_rerank is None else use_rerank

    if use_rerank:
        candidates = search_mod.hybrid_search(
            query, top_k=config.RERANK_CANDIDATES)
        candidates = reranker_mod.rerank(query, candidates)
        return candidates[:top_k]

    return search_mod.hybrid_search(query, top_k=top_k)


# ----------------------------------------------------------------------------
# Идентичность бота и быстрые «мета»-ответы
# ----------------------------------------------------------------------------
BOT_IDENTITY = (
    "Я — бот-ассистент Школы осознанного здоровья. "
    "Помогаю отвечать на вопросы по содержанию видеороликов канала: "
    "задайте вопрос по теме видео, и я найду ответ и приложу ссылки на "
    "нужные моменты ролика."
)

# Вопросы про самого бота и приветствия — отвечаем сразу, без RAG/LLM.
_META_PATTERNS = [
    (r"\bкто (ты|вы)\b|ты кто|\bчто ты такое\b", BOT_IDENTITY),
    (r"что (ты|вы) (умеешь|умеете|можешь|можете|делаешь|делаете)[?!.]*$", BOT_IDENTITY),
    (r"чем (ты|вы) (занимаешься|занимаетесь|полезен|полезны)[?!.]*$", BOT_IDENTITY),
    (r"^(привет|здравствуй|здравствуйте|добрый день|добрый вечер|доброе утро|хай|hello|hi|салют)\b",
     "Здравствуйте! " + BOT_IDENTITY),
    (r"^(пока|до свидания|до встречи|прощай|прощайте|всего доброго|всего хорошего|спокойной ночи|благодарю|спасибо)\b",
     "Всего доброго! Рад был помочь. Обращайтесь, когда появятся вопросы. 👋"),
    (r"^(помощь|help|команды|меню)\b", BOT_IDENTITY),
]


def detect_meta(query: str):
    """Возвращает готовый ответ на «мета»-вопрос (приветствие/про бота) или None."""
    q = (query or "").strip().lower()
    for pattern, text in _META_PATTERNS:
        if re.search(pattern, q):
            return text
    return None


## Фразы, по которым бот отдаёт список доступных видео.
#_VIDEO_LIST_PATTERNS = [
#    r"какие (видео|ролики|выпуски|вебинары)",
#    r"список (видео|роликов|выпусков|вебинаров)",
#    r"(твои|ваши) (видео|ролики|выпуски)",
#    r"покажи (видео|ролики|список)",
#    r"что (у тебя|у вас) есть[?!.]*$",
#    r"что есть на канале[?!.]*$",
#    r"что можно (посмотреть|почитать)[?!.]*$",
#]

_VIDEO_LIST_PATTERNS = [
    # "какие (у тебя/у вас) (есть) видео/ролики/выпуски/вебинары/записи"
    r"какие\s+(?:\w+\s+){0,3}(?:видео|ролик|выпуск|вебинар|запис)\w*",
    # "список (твоих/всех/...) видео/роликов/выпусков/вебинаров/записей"
    r"список\s+(?:\w+\s+){0,3}(?:видео|ролик|выпуск|вебинар|запис)\w*",
    # "список" в одиночку — тоже просьба показать перечень
    r"^\s*список\s*[?!.]*$",
    # "(твои|ваши|твоих|ваших) (видео|ролики|выпуски|вебинары|записи)"
    r"(?:твои|ваши|твоих|ваших)\s+(?:видео|ролик|выпуск|вебинар|запис)\w*",
    # "покажи (мне) (все) (список) (твоих) видео/роликов/..."
    r"покаж\w*\s+(?:\w+\s+){0,3}(?:видео|ролик|выпуск|вебинар|запис)\w*",
    # "покажи (мне) список ..."
    r"покаж\w*\s+(?:\w+\s+){0,3}список\b",
    # "перечисли/выведи/дай/выдай (список) видео/роликов/..."
    r"(?:перечисли|выведи|дай|выдай)\w*\s+(?:\w+\s+){0,3}(?:список|перечень|видео|ролик|выпуск|вебинар|запис)\w*",
    # "что (у тебя|у вас) есть"
    r"что\s+(?:у тебя|у вас)\s+есть[?!.]*$",
    # "что есть на канале"
    r"что\s+есть\s+на\s+канале[?!.]*$",
    # "что можно (посмотреть|почитать)"
    r"что\s+можно\s+(?:посмотреть|почитать)[?!.]*$",
]

def _is_video_list_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(re.search(p, q) for p in _VIDEO_LIST_PATTERNS)


def format_video_list(videos: list[dict]) -> str:
    """Формирует текст списка видео: названия + ссылки."""
    if not videos:
        return "В базе пока нет видео."
    lines = ["Вот доступные видео:"]
    for i, v in enumerate(videos, 1):
        title = v.get("title") or "(без названия)"
        url = v.get("url") or ""
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   {url}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Локальный каталог роликов (PostgreSQL)
# ----------------------------------------------------------------------------
def _catalog_videos() -> list[dict]:
    """Ролики из локального каталога; если пусто/недоступно — фолбэк на Qdrant."""
    try:
        rows = catalog.list_videos()
        if rows:
            return rows
    except Exception:
        pass  # БД каталога недоступна — отвечаем из Qdrant
    return search_mod.list_videos()


_UPDATE_PATTERNS = [
    r"^/update_content\b",
    r"^\s*обнови\s+(каталог|ролики|видео)\b",
    r"^\s*обновить\s+(каталог|ролики|видео)\b",
]


def _is_update_command(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(re.search(p, q) for p in _UPDATE_PATTERNS)


_USERS_PATTERNS = [
    r"^/users\b",
    r"^\s*(сколько\s+)?пользовател\w*\b",
    r"^\s*статистика\b",
]


def _is_users_command(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(re.search(p, q) for p in _USERS_PATTERNS)


def _format_username(telegram_id: int) -> str:
    """Человекочитаемое имя пользователя для отчёта."""
    try:
        u = catalog.get_user_info(telegram_id)
    except Exception:
        u = None
    if u:
        if u.get("username"):
            return f"@{u['username']}"
        name = " ".join(x for x in [u.get("first_name"), u.get("last_name")] if x)
        if name.strip():
            return name.strip()
    return f"id {telegram_id}"


def _users_report() -> str:
    """Отчёт по пользователям и последней активности."""
    try:
        total = catalog.count_users()
        last = catalog.get_last_messages(2)
    except Exception:
        return "Не удалось получить данные из БД."

    lines = [f"👥 Пользователей: {total}"]
    if not last:
        return "\n".join(lines) + "\n\nПока нет сообщений."

    # определяем, какой пользователь задал вопрос (по user-строке)
    user_msg = next((m for m in last if m.get("role") == "user"), last[0])
    name = _format_username(user_msg["telegram_id"])

    lines.append(f"\nПоследняя активность ({name}):")
    for m in last:
        prefix = "👤 Вопрос" if m.get("role") == "user" else "🤖 Бот"
        text = (m.get("text") or "").strip()
        if len(text) > 100:
            text = text[:100] + "…"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


_VIDEO_DESCRIBE_PATTERNS = [
    r"о\s+ч[ёе]м\s+(?:видео|ролик)",
    r"про\s+что\s+(?:видео|ролик)",
    r"расскажи\s+(?:о|про)\s+(?:видео|ролик)",
    r"что\s+за\s+(?:видео|ролик)",
]


def _is_describe_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(re.search(p, q) for p in _VIDEO_DESCRIBE_PATTERNS)


def _extract_video_hint(query: str) -> str:
    """Извлекает «название» из вопроса вида «о чём видео X». Пусто, если нет."""
    q = query.lower()
    m = re.search(
        r"(?:о\s+ч[ёе]м|про\s+что|расскажи\s+(?:о|про)|что\s+за)\s+(?:видео|ролик)",
        q,
    )
    if not m:
        return ""
    hint = q[m.end():].strip()
    hint = re.sub(r"^(с\s+названием|под\s+названием|называется|название)\s*", "", hint)
    return hint.strip(" \t\n:;.,!?«»\"'()")


def _describe_video(query: str):
    """Возвращает готовый текст описания видео по названию, или None."""
    hint = _extract_video_hint(query)
    if not hint:
        return None
    try:
        matches = catalog.find_video(hint)
    except Exception:
        matches = []
    if not matches:
        return ("Не нашёл видео по запросу «{hint}».\n\n".format(hint=hint)
                + format_video_list(_catalog_videos()))
    if len(matches) == 1:
        v = matches[0]
        desc = v.get("description") or "Описание пока не сгенерировано."
        url = v.get("url") or ""
        text = f"{v['title']}\n\n{desc}"
        if url:
            text += f"\n\n{url}"
        return text
    # несколько совпадений — показываем список
    lines = ["Нашёл несколько подходящих видео:"]
    for v in matches:
        lines.append(f"• {v['title']}")
    return "\n".join(lines)


def _extract_video_number(query: str):
    """Извлекает номер ролика из «видео номер N» / «N-е видео». None, если нет."""
    q = query.lower()
    m = re.search(r"(?:видео|ролик)\s*(?:номер|№|#)?\s*(\d+)", q)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*[-–]?\s*(?:е|й|ое|ый|го|я)\s*(?:видео|ролик)", q)
    if m:
        return int(m.group(1))
    return None


def _describe_video_by_number(query: str):
    """Описание ролика по его порядковому номеру в списке каталога."""
    n = _extract_video_number(query)
    if n is None:
        return None
    videos = _catalog_videos()
    if n < 1 or n > len(videos):
        return f"У меня всего {len(videos)} видео. Назови номер от 1 до {len(videos)}."
    v = videos[n - 1]
    desc = v.get("description") or "Описание пока не сгенерировано."
    url = v.get("url") or ""
    text = f"{v['title']}\n\n{desc}"
    if url:
        text += f"\n\n{url}"
    return text


SUMMARIZE_PROMPT = (
    "Ты кратко резюмируешь видеоролик по его транскрипту. "
    "Напиши 2–3 предложения: о чём это видео, какие ключевые темы и советы. "
    "Пиши на русском языке, обычным текстом, без Markdown, сносок и таймкодов."
)


def _summarize_video(title: str, text: str) -> str:
    """Генерирует краткое описание ролика через LLM."""
    text = (text or "").strip()[:12000]
    if not text:
        return ""
    messages = [
        {"role": "system", "content": SUMMARIZE_PROMPT},
        {"role": "user", "content": f"Название: {title}\n\nТранскрипт:\n{text}"},
    ]
    return _unescape_md(call_llm(messages)).strip()


def _update_catalog() -> str:
    """Сканирует Qdrant, генерирует описания и сохраняет каталог в Postgres."""
    videos, grouped = search_mod.scan_collection()
    if not videos:
        return "В Qdrant нет видео."
    rows = []
    for v in videos:
        chunks = grouped.get(v["video_id"], [])
        text = "\n".join(c["text"] for c in chunks)
        desc = _summarize_video(v["title"], text)
        rows.append({
            "video_id": v["video_id"],
            "title": v["title"],
            "description": desc,
            "url": v["url"],
            "chunk_count": len(chunks),
        })
    catalog.upsert_videos(rows)
    return f"Каталог обновлён: {len(rows)} роликов."


# ----------------------------------------------------------------------------
# Промпт и LLM
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Ты — бот-ассистент Школы осознанного здоровья, отвечающий на вопросы "
    "по содержанию видеороликов канала.\n"
    "Сначала определи, о чём вопрос:\n"
    "1) Если вопрос про тебя самого (например, «кто ты», «что ты умеешь»), "
    "приветствие, прощание, благодарность или вопрос не по содержанию видео — ответь как бот-ассистент "
    "о себе, кратко, и НЕ используй предоставленные фрагменты.\n"
    "2) Если вопрос про содержание видео — отвечай ТОЛЬКО на основании "
    "предоставленных фрагментов, от первого лица, как автор видео («я считаю», "
    "«я рекомендую»), кратко и по существу.\n"
    "Если в запросе есть раздел «История диалога» — учитывай её: отвечай с учётом "
    "предыдущих вопросов и ответов (например, «а в чём минусы?» относится к "
    "предыдущей теме).\n"
    "Общие правила: не выдумывай факты, которых нет в контексте; если в контексте "
    "нет ответа — честно скажи об этом; не добавляй сносок, номеров фрагментов, "
    "таймкодов, Markdown-разметки и обратных слэшей; пиши на русском языке."
)


def _fmt(seconds) -> str:
    if seconds is None:
        return "??:??"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        if config.CONTEXT_SHOW_TIMECODES:
            parts.append(
                f"[Фрагмент {i} | {_fmt(r['start'])}–{_fmt(r['end'])}]\n{r['text']}"
            )
        else:
            parts.append(r['text'])
    return "\n\n".join(parts)


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    import requests
    base = config.LLM_BASE_URL.rstrip("/")
    url = base + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_tokens": config.LLM_MAX_TOKENS,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _unescape_md(text: str) -> str:
    """Чистит ответ модели от Markdown-артефактов.

    DeepSeek иногда возвращает текст с обратными слэшами перед спецсимволами
    (\. \- \( \), MarkdownV2-экранирование), а также выделяет слова звёздочками
    (*слово*) или подчёркиваниями (_слово_). Для «чистого» текстового ответа
    боту всё это убираем.
    """
    # \X → X для спецсимволов Markdown и обычных знаков пунктуации.
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!>|])", r"\1", text)
    # *текст* или _текст_ → текст (снимаем простые выделения).
    text = re.sub(r"(?<!\w)([*_])(.*?)\1(?!\w)", r"\2", text)
    return text


def _load_history_text(telegram_id) -> str:
    """Последние сообщения диалога пользователя в виде текста (или '')."""
    if telegram_id is None:
        return ""
    try:
        rows = catalog.load_history(telegram_id)
    except Exception:
        return ""
    if not rows:
        return ""
    lines = []
    for r in rows:
        who = "Пользователь" if r["role"] == "user" else "Бот"
        lines.append(f"{who}: {r['text']}")
    return "\n".join(lines)


def answer(query: str, top_k: int = None, use_rerank: bool = None,
           telegram_id: int = None) -> tuple[str, list[dict]]:
    """Полный цикл: ответ + сохранение истории диалога."""
    text, sources = _answer_impl(query, top_k, use_rerank, telegram_id)

    # Сохраняем пару «вопрос → ответ» в историю (все сообщения),
    # кроме служебных команд — чтобы они не попадали в «последнюю активность».
    if telegram_id is not None and not (_is_update_command(query) or _is_users_command(query)):
        try:
            catalog.save_message(telegram_id, "user", query)
            catalog.save_message(telegram_id, "assistant", text)
        except Exception:
            pass  # не роняем бота, если БД недоступна

    return text, sources


def _answer_impl(query: str, top_k, use_rerank, telegram_id):
    """Логика ответа без сохранения истории."""
    # «Мета»-вопросы (про бота, приветствия) — мгновенный ответ без RAG/LLM.
    meta = detect_meta(query)
    if meta is not None:
        return meta, []

    # Команда обновления каталога (пока без ограничений по пользователю).
    if _is_update_command(query):
        return _update_catalog(), []

    # Отчёт по пользователям и активности.
    if _is_users_command(query):
        return _users_report(), []

    # «видео номер N» — ролик по порядковому номеру в каталоге.
    if _extract_video_number(query) is not None:
        return _describe_video_by_number(query), []

    # Описание конкретного видео (из локального каталога, без LLM).
    if _is_describe_query(query):
        desc = _describe_video(query)
        if desc is not None:
            return desc, []

    # Запрос списка видео — из локального каталога (фолбэк на Qdrant).
    if _is_video_list_query(query):
        return format_video_list(_catalog_videos()), []

    results = retrieve(query, top_k=top_k, use_rerank=use_rerank)
    context = build_context(results)

    history_text = _load_history_text(telegram_id)
    user_content = f"Вопрос: {query}\n\nКонтекст:\n{context}"
    if history_text:
        user_content = f"История диалога:\n{history_text}\n\n{user_content}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    answer_text = call_llm(messages)
    return _unescape_md(answer_text), results
