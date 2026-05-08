# RUNBOOK

How to bring DriftWatch up from a clean clone, run the Friday demo, and recover from the most common failure modes.

---

## 1. Prerequisites

- Docker Desktop running (Windows/macOS) or Docker Engine (Linux)
- `uv` installed on host (`pip install uv` or [astral.sh/uv](https://astral.sh/uv))
- A Groq API key (https://console.groq.com/keys) for the agent's LLM

---

## 2. First-time setup (clean clone)

```powershell
# 1. Copy the env template and fill it in
cp .env.example .env

# 2. Open .env and set:
#    - GROQ_API_KEY=gsk_...                    (your key)
#    - POSTGRES_PASSWORD=...                   (any random string)
#    - WEBHOOK_HMAC_SECRET=...                 (32+ random bytes; see comment in file)
#    - PROMOTION_BEARER_TOKEN=...              (any random string)
#    Update PLATFORM_DATABASE_URL / AGENT_DATABASE_URL / MLFLOW_BACKEND_URI
#    to use the same POSTGRES_PASSWORD.

# 3. Build images and bring the full stack up
docker compose up -d --build

# 4. Apply database migrations (one-time per fresh postgres volume)
docker compose exec platform uv run alembic upgrade head
docker compose exec agent uv run alembic upgrade head
```

After ~30 seconds:

| Service | URL | Purpose |
|---|---|---|
| Dashboard (Streamlit) | http://localhost:8501 | Reviewer console |
| Platform API + docs | http://localhost:8001/docs | Predict, registry, drift, promote |
| Agent API + docs | http://localhost:8002/docs | Webhooks, investigations, HIL |
| MLflow UI | http://localhost:5000 | Experiment & model registry |

Verify with `docker compose ps` — all 7 services should be `Up` (postgres + redis show `(healthy)`).

---

## 3. Train the model (one-time per fresh stack)

The platform serves predictions from artifacts in `platform/artifacts/model_v1/`. If you wiped postgres or are on a fresh build, train first:

```powershell
docker compose exec platform uv run python -m app.ml.train
```

This:
- Loads the bank marketing CSV
- Trains 3 candidate pipelines, picks the best by CV AUC under recall ≥ 0.75
- Writes `model.pkl`, `schema.json`, `threshold.json`, `metrics.json`, `card.md` to `platform/artifacts/model_v1/`
- Logs the run to MLflow + registers as `driftwatch-bank-marketing` v1

Confirm in the dashboard's Registry tab — should show v1 with test AUC/F1.

---

## 4. The Friday demo (live drift)

**Order matters — open the dashboard FIRST so the audience sees the baseline.**

### Step 1 — Open the dashboard on screen
```
http://localhost:8501
```
All four tabs (Registry, Investigations, Queue & DLQ, HIL Inbox) should show calm/empty state. Registry shows the trained v1.

### Step 2 — Run the traffic script in a terminal
```powershell
uv run --project scripts scripts/send_traffic.py
```

What it does:
- **Phase 1:** sends 50 normal predictions → POSTs `/api/v1/drift/check` → severity should be `none`
- Pauses for `Enter` (lets you narrate the moment)
- **Phase 2:** sends 200 *shifted* predictions (mutates one numeric + one categorical column) → POSTs `/api/v1/drift/check` → severity should jump to `medium`/`high` → webhook fires to agent

### Step 3 — Watch the dashboard react (auto-refreshes every 5s)
1. **Investigations tab** populates with a new row
2. **Queue & DLQ tab** shows the action job the agent enqueued
3. **HIL Inbox tab** shows a pending approval

### Step 4 — Click Approve in the HIL Inbox
The agent calls platform's `/api/v1/promote` with the bearer token → registry updates → Registry tab shows the new model version is now Production.

### Customizing the script
```powershell
# point at a different platform host
uv run --project scripts scripts/send_traffic.py --base-url http://localhost:8001

# change phase sizes
uv run --project scripts scripts/send_traffic.py --normal 60 --shifted 250

# skip the pause between phases (for automated runs)
uv run --project scripts scripts/send_traffic.py --no-pause
```

The shift configuration lives at the top of [scripts/send_traffic.py](scripts/send_traffic.py):
```python
SHIFT_NUMERIC_COL = "balance"          # or "euribor3m" if dataset has it
SHIFT_NUMERIC_DELTA = 50_000.0         # tune to trip PSI
SHIFT_CATEGORICAL_COL = "job"
SHIFT_CATEGORICAL_VALUE = "student"
```

---

## 5. Stopping & restarting

```powershell
docker compose stop                  # stop containers, keep volumes
docker compose start                 # restart with same data

docker compose down                  # stop + remove containers, keep volumes
docker compose up -d                 # back up

docker compose down -v               # NUKE volumes (predictions, registry, MLflow runs)
```

After `down -v` you must re-run migrations + retrain (sections 2.4 and 3).

---

## 6. Common failures & fixes

### "password authentication failed for user postgres"
Postgres volume was created with the *old* `POSTGRES_PASSWORD`. The volume persists after rebuilds, so changing `.env` alone doesn't help.

```powershell
docker compose down
docker volume rm driftwatch_postgres_data
docker compose up -d --build
docker compose exec platform uv run alembic upgrade head
docker compose exec agent uv run alembic upgrade head
```

### `relation "investigations" does not exist` (or `model_registry_records`)
Migrations weren't run on the current postgres volume.
```powershell
docker compose exec platform uv run alembic upgrade head
docker compose exec agent uv run alembic upgrade head
```

### `/predict` returns 500: `FileNotFoundError: ... threshold.json`
Model artifacts aren't inside the platform container. Either train in-container (section 3) or copy host artifacts in:
```powershell
docker cp platform/artifacts driftwatch-platform:/app/platform/
docker cp platform/data driftwatch-platform:/app/platform/
```

### Agent rejects webhook with 400 / `webhook_body_invalid`
Platform's webhook payload doesn't match `DriftEventV1` (severity labels, field names, or types). Check `docker compose logs platform agent` for the validation error and align the contract.

### Dashboard tab shows "Could not reach platform/agent: timed out"
The named service is down or restarting. `docker compose ps` to confirm; `docker compose logs <service>` for the cause.

### Agent crash-loops on startup
Almost always one of:
- Postgres password mismatch (see above)
- `GROQ_API_KEY` missing or invalid in `.env`
- Migrations not run yet on a fresh volume

---

## 7. Quick health check

```powershell
docker compose ps                        # all Up
curl http://localhost:8001/api/v1/health # platform OK
curl http://localhost:8002/health        # agent OK
```

If all three pass, the stack is ready for the demo.