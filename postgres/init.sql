-- Схема БД Telegram-бота (RAG) — создаётся при первом старте Postgres.
-- Движок: PostgreSQL 16 (docker), БД указывется через POSTGRES_DB.

-- 1) Пользователи. Один telegram_id = один пользователь бота.
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    telegram_id   BIGINT      NOT NULL UNIQUE,          -- id пользователя в Telegram
    username      TEXT,                                 -- @username (может отсутствовать)
    first_name    TEXT,                                 -- имя из Telegram
    last_name     TEXT,                                 -- фамилия из Telegram
    language_code TEXT,                                 -- язык клиента (ru/en/...)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2) Персональные данные пользователя (профиль).
-- В будущем сюда добавим дату рождения, пол, рост/вес и т.п.
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id        BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    full_name      TEXT,                                 -- полное имя (если != first/last)
    birth_date     DATE,                                 -- дата рождения
    gender         TEXT,                                 -- 'male' / 'female' / NULL
    height_cm      NUMERIC(5,2),                         -- рост, см
    weight_kg      NUMERIC(5,2),                         -- вес, кг
    notes          TEXT,                                 -- произвольные заметки
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id)
);

-- 3) Результаты анализов (лабораторные и прочие измерения).
-- Каждая строка — один показатель пользователя на определённую дату.
CREATE TABLE IF NOT EXISTS user_metrics (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_name TEXT        NOT NULL,                    -- напр. 'vitamin_d', 'cholesterol', 'glucose'
    value       NUMERIC,                                 -- числовое значение (если есть)
    unit        TEXT,                                    -- единица измерения (нг/мл, ммоль/л, ...)
    text_value  TEXT,                                    -- текстовое значение (для нечисловых анализов)
    measured_at DATE,                                    -- дата измерения/сдачи анализа
    source      TEXT,                                    -- напр. 'ручной ввод', 'файл', 'лаборатория'
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_metrics_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 4) Каталог видеороликов (кэш, пополняется командой /update_content).
-- Хранит название + краткое описание + ссылку, чтобы отвечать на
-- «покажи ролики» и «о чём видео X» без обращения к Qdrant и LLM.
CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Индексы для быстрого поиска.
CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id);
CREATE INDEX IF NOT EXISTS idx_metrics_user_name ON user_metrics(user_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_user_date ON user_metrics(user_id, measured_at);

-- 5) История диалога (последние 10 сообщений на пользователя).
-- Используется для контекста беседы (многоходовые вопросы).
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL,
    role        TEXT NOT NULL,           -- 'user' | 'assistant'
    text        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_telegram ON messages(telegram_id, created_at DESC);

-- 6) Пользователь для DBeaver (read-only) — для ручного просмотра БД.
-- Создаётся отдельно (см. README), т.к. init.sql выполняется только при
-- первом старте volume. Пример:
--   CREATE ROLE dbeaver LOGIN PASSWORD '<пароль>';
--   GRANT CONNECT ON DATABASE bot_db TO dbeaver;
--   GRANT USAGE ON SCHEMA public TO dbeaver;
--   GRANT SELECT ON ALL TABLES IN SCHEMA public TO dbeaver;
--   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dbeaver;