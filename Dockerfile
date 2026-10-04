FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    GEMINI_REFRESH_INTERVAL=600 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
# Gemini 网页端逆向通过 HTTP 协议工作，不需要下载 Chromium 浏览器运行时。
ARG UV_HTTP_TIMEOUT=180
RUN UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY api ./api
COPY core ./core
COPY providers ./providers
COPY utils ./utils

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--access-log"]
