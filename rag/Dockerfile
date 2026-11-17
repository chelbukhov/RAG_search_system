FROM python:3.11-slim

# libgomp1 нужен torch/scikit-learn, curl — для самопроверки /health
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости (torch на Linux с PyPI ставится CPU-only — легче и не требует GPU)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код RAG-сервиса + BM25-словарь (sparse-поиск)
COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
