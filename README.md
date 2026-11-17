# Telegram RAG Bot — сценарий n8n

Чат-бот Telegram → RAG (Qdrant) → готовый ответ. Сценарий принимает сообщение,
отправляет его в RAG-сервис проекта **transcrypt** и возвращает пользователю
LLM-ответ. Настроен на короткий, быстрый ответ (~4–6 с) без источников.

## Архитектура (продакшн, Linux-сервер)

```
Telegram → n8n (Docker, https://n8n.chelbukhov.ru)
               │  HTTP POST http://rag:8000/ask   ← внутренняя Docker-сеть
               ▼
          rag (Docker: serve.py / FastAPI)
               │  Qdrant Cloud (гибридный поиск) + DeepSeek (LLM)
               ▼
          { answer } → Telegram Send Message
```

Важно: n8n **не** ходит в Qdrant напрямую. Весь RAG (dense+sparse поиск,
реранкер, LLM) обёрнут в `serve.py`, который работает отдельным Docker-контейнером
`rag` в той же compose-сети, что и n8n. n8n обращается к нему по имени `rag`.

## Развёртывание на сервере

Сервер: `217.69.174.112` (Ubuntu 22.04), всё в Docker в `/opt/n8n/`.

Стек (`docker-compose.yml`):
- `caddy` — HTTPS (n8n.chelbukhov.ru, Let's Encrypt)
- `n8n` — порт 5678 только внутри сети
- `rag` — `serve.py`, порт 8000 только внутри сети
- `stt` — распознавание голоса (faster-whisper), порт 9000 только внутри сети
- `postgres` — PostgreSQL 16, порт 5432 только внутри сети (база пользователей)

Каталог `/opt/n8n/rag/` содержит:
- `Dockerfile`, `requirements.txt`
- `src/` — код RAG (serve, rag, search, reranker, embeddings, bm25, config)
- `data/bm25_stats.json` — словарь/IDF для sparse-поиска
- `.env` — ключи Qdrant Cloud и LLM (подключается через `env_file`)
- кэш HF-моделей — в Docker volume `rag_hf_cache`

Каталог `/opt/n8n/stt/` содержит:
- `Dockerfile`, `requirements.txt`
- `app/main.py` — FastAPI-сервис `POST /transcribe`
- `.env` — `TELEGRAM_BOT_TOKEN`, настройки модели Whisper
- кэш модели Whisper — в Docker volume `stt_whisper_cache`

### Полезные команды (на сервере)

```bash
cd /opt/n8n
docker compose ps                                      # статус контейнеров
docker compose logs -f rag                             # логи RAG (видно прогрузку моделей)
docker compose build rag && docker compose up -d       # пересобрать/перезапустить rag
docker exec rag curl -s localhost:8000/health          # проверка внутри контейнера
```

## Импорт workflow в n8n

1. Откройте https://n8n.chelbukhov.ru → **Workflows** → **Import from File**.
2. Выберите `telegram-rag-bot.json`.
3. Создайте credential **Telegram** (токен бота от `@BotFather`) и подключите
   его в обоих Telegram-узлах.
4. Включите **Active**.

Узел **Ask RAG** уже указывает на `http://rag:8000/ask` — менять ничего не нужно,
если n8n и rag в одной compose-сети.

## Как устроен workflow

| Узел | Что делает |
|------|------------|
| `Telegram Trigger` | ловит входящее сообщение (`$json.message.text`) |
| `Typing` (Telegram) | `sendChatAction` с `action: typing` — пользователь видит «печатает…» |
| `Is Start?` (IF) | если текст `== /start` → ветка приветствия, иначе → RAG |
| `Prepare User` (Code) | извлекает `telegram_id`/`username`/`first_name`/`last_name`/`language_code`/`chatId` из сообщения |
| `Upsert User` (Postgres) | upsert в таблицу `users` по `telegram_id` |
| `Send Start` | приветствие и краткий рассказ о боте |
| `Ask RAG` | HTTP `POST http://rag:8000/ask`, тело `{"question": ..., "use_rerank": false}` |
| `Format Reply` (Code) | берёт `answer`, добавляет пронумерованный список кликабельных ссылок на источники, формирует `text` + `chatId` |
| `Send Answer` | отправляет ответ обратно в чат |

## Ссылки на источники (переход к моменту видео)

К каждому источнику `/ask` дополнительно добавляет поля `url` и `link`:

```json
{
  "time": "47:25–50:09",
  "url": "https://www.youtube.com/watch?v=ovfvFJbXiAk",
  "link": "https://www.youtube.com/watch?v=ovfvFJbXiAk&t=2845s"
}
```

- `url` — берётся из метаданных чанка в Qdrant (payload `url`, записывается
  при индексации из `VIDEO_URL`).
- `link` — готовая ссылка на конкретный момент: `url&t=<start>s`
  (YouTube открывает видео с нужной секунды). Формируется в `serve.py`
  функцией `_video_link()`.

Узел **Format Reply** собирает из этих полей строку «📽 Источники: …» —
пронумерованный список (1, 2, 3…) кликабельных ссылок, каждая ведёт на свой
таймкод. Это удобно, когда в RAG Store хранятся чанки из нескольких видео —
ссылка всегда указывает на правильный ролик.

> Чтобы ссылки появлялись в ответе, `url` должен быть записан в payload чанков
> при индексации (поле `VIDEO_URL` в `.env` проекта transcrypt).

## База данных пользователей (PostgreSQL)

Локальная БД на сервере (контейнер `postgres` в том же стеке), доступна только
внутри Docker-сети. Секреты — в `/opt/n8n/.env`:

```
POSTGRES_DB=bot_db
POSTGRES_USER=bot
POSTGRES_PASSWORD=<сгенерированный пароль>
```

Схема задаётся `postgres/init.sql` при первом старте (volume `postgres_data`):

| Таблица | Назначение |
|---------|-----------|
| `users` | пользователи бота: `telegram_id` (unique), `username`, `first_name`, `last_name`, `language_code` |
| `user_profiles` | персональные данные (полное имя, дата рождения, пол, рост/вес, заметки) — задел на будущее |
| `user_metrics` | результаты анализов/измерений: `metric_name`, `value`, `unit`, `measured_at`, `source` — задел на будущее |
| `videos` | каталог роликов: `video_id`, `title`, `description`, `url`, `chunk_count` |
| `messages` | история диалога: `telegram_id`, `role`, `text`, `created_at` (последние 10 на пользователя) |

Подключение из n8n: host `postgres`, порт `5432`, БД `bot_db`, юзер `bot`.

### Доступ из DBeaver (SSH-туннель + read-only)

PostgreSQL привязан к `127.0.0.1:5432` на сервере — наружу порт **не** открыт.
Подключение из DBeaver — через SSH-туннель:

1. В DBeaver создай подключение PostgreSQL.
2. Вкладка **SSH** → «Use SSH Tunnel» → host `217.69.174.112`, порт `22`,
   user `root`, пароль (или ключ).
3. Основное подключение → host `localhost`, порт `5432`, БД `bot_db`,
   user `dbeaver`, пароль `<см. /opt/n8n/.env на сервере>`.

Пользователь `dbeaver` — **read-only** (только `SELECT`, запись запрещена).
Для изменений используй основного `bot` (но это и есть боевой юзер n8n/rag).

### Полезные команды (Postgres)

```bash
cd /opt/n8n
docker compose logs -f postgres                                   # логи БД
docker exec postgres psql -U bot -d bot_db -c '\dt'              # список таблиц
docker exec postgres psql -U bot -d bot_db -c 'select * from users;'  # пользователи
```

### Регистрация пользователя при `/start`

Ветка `/start` теперь: **Prepare User** (Code) → **Upsert User** (Postgres) →
**Send Start**.

- `Prepare User` достаёт из `$json.message.from` поля `telegram_id`, `username`,
  `first_name`, `last_name`, `language_code` и `chatId`.
- `Upsert User` (Postgres v2.1, операция **Insert or Update**) пишет в таблицу
  `users`, матчит по уникальному `telegram_id`: если пользователь уже есть —
  обновляет, иначе создаёт (логика upsert).

Подключение Postgres в n8n:
- credential **Postgres**: host `postgres`, порт `5432`, БД `bot_db`,
  пользователь `bot`, пароль из `/opt/n8n/.env` (`POSTGRES_PASSWORD`).

## «Мета»-вопросы и идентичность бота

Бот различает вопросы про **себя** и вопросы про **содержание видео**:

- Вопросы про бота («кто ты», «что ты умеешь»), приветствия и прочее вне темы
  видео — обрабатываются **мгновенно** в `rag.py` (функция `detect_meta()`),
  без обращения к Qdrant и LLM. Готовые фразы заданы константами
  `BOT_IDENTITY` и `_META_PATTERNS`.
- Остальные вопросы идут через RAG. В `SYSTEM_PROMPT` добавлено правило
  маршрутизации: если вопрос не про содержимое видео — отвечать как бот-ассистент,
  а не «за автора видео».

Идентичность бота (текст «кто ты») задаётся в `rag.py` в `BOT_IDENTITY`.
По умолчанию: «Я — бот-ассистент Школы осознанного здоровья…».

> Приветствие `/start` по-прежнему обрабатывается в n8n (ветка `Is Start?`),
> а не в `detect_meta` — но `detect_meta` ловит повторные «привет» и т.п.

## Каталог роликов (PostgreSQL)

Бот умеет перечислять ролики и отвечать «о чём видео X» — из локального
кэша в PostgreSQL (таблица `videos`), без Qdrant и LLM.

Распознаются фразы:
- список: «какие видео», «список видео», «твои видео», «что у тебя есть»,
  «что есть на канале», «покажи видео», «что можно посмотреть?»;
- описание: «о чём видео …», «про что ролик …», «расскажи о видео …» (поиск
  по названию через `ILIKE`).

### Обновление каталога

Команда `/update_content` (пока доступна всем) сканирует Qdrant, для каждого
ролика генерирует краткое описание через LLM (суммаризация транскрипта) и
сохраняет в таблицу `videos`. Ответ: «Каталог обновлён: N роликов».

> Ограничение по пользователям (только админ) планируется добавить позже
> на уровне workflow n8n.

### Ответ бота на «покажи ролики» (пример)

```
Вот доступные видео:
1. Карнивор-диета от А до Я с Мариной: ...
   https://www.youtube.com/watch?v=iKkgxq2SnBM
2. ...
```

Фолбэк: если каталог в Postgres пуст (ещё не запускали `/update_content`),
список берётся напрямую из Qdrant (`search.list_videos()`).

Также доступен endpoint `GET /videos` (в `serve.py`) — возвращает список
видео из Qdrant для проверки:
```bash
docker exec rag curl -s localhost:8000/videos
```

## Команда /users (статистика и активность)

Служебная команда `/users` (по аналогии с `/update_content`) показывает:
- количество зарегистрированных пользователей (таблица `users`);
- последнюю активность: последний вопрос и ответ бота (последние 2 строки
  таблицы `messages`) с указанием, какой пользователь задал вопрос.

Реализовано в `rag.py` (`_users_report`, `_format_username`, `_is_users_command`).
Ответ не сохраняется в историю (как и `/update_content`), чтобы не засорять
«последнюю активность».

Пример ответа:
```
👥 Пользователей: 3

Последняя активность (@username):
👤 Вопрос: что такое карнивор
🤖 Бот: Карнивор — это ...
```

### «Расскажи про видео номер N»

После «покажи ролики» можно сослаться на номер: «расскажи про видео номер 14»
(или «14-е видео»). Бот по порядковому номеру в списке находит ролик и отдаёт
его описание. Реализовано в `rag.py` (`_extract_video_number`,
`_describe_video_by_number`), без LLM.

## История диалога (контекст беседы)

Бот хранит **последние 10 сообщений** на пользователя в таблице `messages`
(`telegram_id`, `role`, `text`, `created_at`). Это даёт многоходовые диалоги:

```
> что такое карнивор
→ ответ про карнивор
> а в чём минусы?
→ бот понимает, что речь про карнивор (из истории)
```

Как это устроено:
- n8n передаёт `telegram_id` в `/ask` (в узле **Ask RAG** поле
  `telegram_id: $('Telegram Trigger').item.json.message.from.id`).
- `rag.py` перед RAG-ответом подгружает историю и добавляет её в промпт
  (раздел «История диалога»), а после ответа сохраняет пару «вопрос → ответ».
- При записи оставляет только последние 10 сообщений (таблица не разрастается).

## Распознавание голоса (STT)

Голосовые сообщения Telegram распознаются локальным сервисом `stt`
(faster-whisper, модель `small`), после чего текст идёт по тому же пути,
что и обычный текст (включая «мета»-вопросы «кто ты» и список видео).

Поток голоса:
```
Telegram (voice) → n8n If voice → HTTP POST http://stt:9000/transcribe {file_id}
   → stt: getFile → скачивание .ogg → ffmpeg (ogg→wav) → whisper → {text}
   → n8n: Prepare Voice Question ({question: text}) → Ask RAG → …
```

Сервис `stt` сам забирает файл из Telegram по `TELEGRAM_BOT_TOKEN` (в
`stt/.env`), поэтому n8n передаёт ему только `file_id`.

Эндпоинты STT:
- `GET /health` — проверка живости.
- `POST /transcribe` — тело `{"file_id": "..."}` → ответ `{"text": "..."}`.

Проверка вручную (изнутри `stt`):
```bash
docker exec stt curl -s localhost:9000/health
```

Настройки модели (в `stt/.env`): `WHISPER_MODEL` (по умолч. `small`),
`WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, `STT_LANGUAGE` (пусто =
автоопределение). Первый запуск скачивает модель (~460 МБ) в volume
`stt_whisper_cache`.

## Настройки RAG-сервиса (флаги конфигурации)

Все настраиваются через переменные окружения (в `rag/.env` или
`docker-compose.yml` → `environment`), после изменения — перезапуск
`docker compose up -d rag` (пересборка кода **не** требуется):

| Переменная | Значение по умолчанию | Назначение |
|------------|----------------------|------------|
| `LLM_MAX_TOKENS` | `500` | Предел длины ответа. Меньше → быстрее. ~800–1000 → развёрнутый ответ |
| `CONTEXT_SHOW_TIMECODES` | `false` | Включать ли метки `[Фрагмент N \| таймкод]` в контекст → в ответ |
| `LLM_MODEL` | `deepseek/deepseek-chat` | Модель LLM (через OpenRouter) |

Флаги, заданные в коде (`config.py`), а не через env:
- `RERANK_ENABLED = False` — реранкер выключен (на CPU он ~76 с на запрос).

### Как вернуть источники/таймкоды в ответ (в будущем)

1. Включить метки в контексте: `CONTEXT_SHOW_TIMECODES=true`.
2. Убрать из `SYSTEM_PROMPT` (файл `rag.py`) строку
   «Не добавляй сносок, номеров фрагментов или таймкодов» — чтобы модель
   снова цитировала фрагменты.

### Как вернуть реранкер (в будущем, напр. на сервере с GPU)

В `config.py` поднять `RERANK_ENABLED` в `True`, затем:
```bash
cd /opt/n8n && docker compose build rag && docker compose up -d
```

## Производительность (замеры)

Прогон полного цикла `/ask` (2 vCPU, 11 ГБ RAM VPS):

| Этап | Время |
|------|-------|
| dense-эмбеддинг (e5, прогрет) + sparse (BM25) | < 1 с |
| гибридный поиск Qdrant | ~0.3 с |
| реранкер (bge-reranker-v2-m3) | ~~76 с~~ **выключен** |
| генерация DeepSeek (`max_tokens=500`) | ~3–4 с |
| **Итого** | **~4–6 с** |

Раньше реранкер и развёрнутые ответы давали ~90 с — поэтому они отключены.

## Проверка

Напишите боту вопрос по содержимому видео (например, «Что помогает восстановить
кожу?»). В ответ придёт краткий сгенерированный ответ и список «📽 Источники»
с кликабельными ссылками на конкретные моменты видео.