# File summary:
# - Builds the Docker image for the platform FastAPI service.
# - Installs dependencies with `uv` from `platform/pyproject.toml` and `platform/uv.lock`.
# - Copies the contracts package, Alembic files, and platform application code into the image.
# - Starts Uvicorn on port 8000 inside the container.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

# uv via official slim binary (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Copy the sibling contracts package first so the path dep resolves at sync time.
COPY contracts /contracts
COPY platform/pyproject.toml /app/pyproject.toml
COPY platform/uv.lock /app/uv.lock

RUN uv sync --frozen --no-dev

COPY platform/alembic.ini /app/alembic.ini
COPY platform/alembic /app/alembic
COPY platform/artifacts /app/platform/artifacts
COPY platform/data /app/platform/data
COPY platform/app /app/app

EXPOSE 8000
CMD ["bash", "-c", "uv run --no-sync alembic upgrade head && uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000"]
