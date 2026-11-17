#!/usr/bin/env python3
"""STT-сервис: голосовое Telegram → текст (faster-whisper).

Эндпоинт:
  POST /transcribe   {"file_id": "<telegram file_id>"}  →  {"text": "..."}

Сервис сам:
  1) получает file_path через Telegram API getFile (нужен bot-token);
  2) скачивает файл (ogg/opus);
  3) конвертирует ffmpeg ogg → wav (16k mono);
  4) распознаёт faster-whisper (модель small);
  5) возвращает текст.

Запуск:
  python main.py            # http://0.0.0.0:9000
"""

import io
import os
import subprocess
import tempfile

import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

from faster_whisper import WhisperModel

# ----------------------------------------------------------------------------
# Конфигурация (env)
# ----------------------------------------------------------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# RU по умолчанию; Whisper сам подхватит прочие языки.
LANGUAGE = os.getenv("STT_LANGUAGE", "")  # пусто = автоопределение
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "1"))  # 1 = greedy (быстро), 5 = beam

app = FastAPI(title="Telegram STT", version="1.0.0")

# Ленивая загрузка модели (кешируется в процессе + volume на диске).
_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"Загружаю faster-whisper: {MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE})")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Модель загружена.")
    return _model


# ----------------------------------------------------------------------------
# Telegram helpers
# ----------------------------------------------------------------------------
def tg_get_file_path(file_id: str) -> str:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
    r = requests.get(url, params={"file_id": file_id}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getFile: {data}")
    return data["result"]["file_path"]


def tg_download(file_path: str) -> bytes:
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


# ----------------------------------------------------------------------------
# Аудио-конвертация и распознавание
# ----------------------------------------------------------------------------
def ogg_to_wav(ogg_bytes: bytes) -> bytes:
    """Конвертирует ogg/opus в WAV 16k mono через ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as fin:
        fin.write(ogg_bytes)
        in_path = fin.name

    out_path = in_path + ".wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def transcribe_bytes(audio_wav: bytes) -> str:
    model = get_model()
    segments, info = model.transcribe(
        io.BytesIO(audio_wav),
        language=LANGUAGE or None,
        beam_size=BEAM_SIZE,   # greedy(1) = максимальная скорость
        vad_filter=True,       # отсекает паузы
        condition_on_previous_text=False,  # не тратим время на подстройку контекста
    )
    parts = [s.text.strip() for s in segments]
    return " ".join(p for p in parts if p)


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    file_id: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    file_path = tg_get_file_path(req.file_id)
    ogg = tg_download(file_path)
    wav = ogg_to_wav(ogg)
    text = transcribe_bytes(wav)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=False)