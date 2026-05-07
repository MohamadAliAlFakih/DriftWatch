FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

# install uv binary into /usr/local/bin
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# install Python deps from worker/pyproject.toml (frozen lock) — done before COPY of app code so dep cache survives source edits
COPY worker/pyproject.toml /app/pyproject.toml
COPY worker/uv.lock /app/uv.lock
RUN uv sync --frozen --no-dev

# D-01: build-time COPY of platform's ML modules so retrain can re-fit the pipeline.
# Lands as /app/ml/{preprocessing,train,data,...}.py — accessible via `from ml.preprocessing import ...`.
COPY platform/app/ml /app/ml

# D-07: COPY raw UCI dataset to a separate path so it doesn't shadow the ml package.
# The raw dataset is at /app/data/bank-full.csv; SHE's data.py computes the deterministic split.
COPY platform/data /app/data

# COPY worker source last so source edits don't bust the dep cache layer above
COPY worker /app/worker

# expose worker package as top-level imports (`from tools.replay import replay`) plus /app for `import ml.*`
ENV PYTHONPATH=/app/worker:/app

# arq entrypoint reads main.WorkerSettings from /app/worker (on PYTHONPATH)
CMD ["uv", "run", "arq", "main.WorkerSettings"]