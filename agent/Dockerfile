FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

# uv via official slim binary (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Copy the sibling contracts package first so the path dep resolves at sync time.
COPY contracts /contracts
COPY agent/pyproject.toml /app/pyproject.toml
COPY agent/uv.lock /app/uv.lock

RUN uv sync --frozen --no-dev

COPY agent/app /app/app
# include Alembic config + migrations so `alembic upgrade head` works inside the container
COPY agent/alembic.ini /app/alembic.ini
COPY agent/alembic /app/alembic

EXPOSE 8000
CMD ["bash", "-c", "uv run --no-sync alembic upgrade head && uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000"]
