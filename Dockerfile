FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

# catboost требует libgomp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY config ./config
COPY src ./src
COPY scripts ./scripts

# Модель и признаки не запекаются в образ: при старте сервис скачивает их
# из S3 (параметры — через переменные окружения / env_file) либо берёт
# из смонтированных каталогов models/ и data/prepared/.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data/raw data/prepared models logs \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"APP_PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn src.main:app --host \"$APP_HOST\" --port \"$APP_PORT\""]
