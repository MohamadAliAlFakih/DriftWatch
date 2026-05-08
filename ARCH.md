# ARCH

System architecture for DriftWatch — a self-healing MLOps stack for the UCI Bank Marketing classifier.

---

## 1. Top-level diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          docker compose network                            │
│                                                                            │
│  ┌──────────┐   webhook   ┌────────┐   enqueue    ┌────────┐               │
│  │ platform │ ──────────► │ agent  │ ───────────► │ redis  │ ◄──┐          │
│  │ (FastAPI)│ ◄────────── │(LangGr)│              └────────┘    │ pull     │
│  └────┬─────┘  promote    └───┬────┘                            │          │
│       │                       │                            ┌────┴───┐      │
│       │ uses                  │ checkpoints                │ worker │      │
│       ▼                       ▼                            └────┬───┘      │
│  ┌──────────┐ ◄───────────────┴────────────────────────────────┘           │
│  │ postgres │   (3 logical DBs: platform / agent / mlflow)                 │
│  └────┬─────┘                                                              │
│       │ backend store                                                      │
│       ▼                                                                    │
│  ┌──────────┐                                                              │
│  │ mlflow   │ — experiment tracking + model registry                       │
│  └──────────┘                                                              │
│                                                                            │
│  ┌────────────────────────────────────────────────────────┐                │
│  │ dashboard  (Streamlit, polls platform + agent every 5s)│                │
│  └────────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────┘

  External:
   - Browser (reviewer) → http://localhost:8501
   - send_traffic.py    → http://localhost:8001/api/v1/predict
```

---

## 2. The 7 services

| Service | Image / source | Internal port | Host port | Role |
|---|---|---|---|---|
| **platform** | `./platform/Dockerfile` (FastAPI + uv) | 8000 | 8001 | Model serving, drift detection, registry, promotion gate |
| **agent** | `./agent/Dockerfile` (FastAPI + LangGraph) | 8000 | 8002 | Drift webhook receiver, supervisor agent, HIL |
| **worker** | `./worker/Dockerfile` (arq) | — | — | Background job consumer (replay/retrain/rollback) |
| **dashboard** | `./dashboard/Dockerfile` (Streamlit) | 8501 | 8501 | Reviewer UI: registry, investigations, queue/DLQ, HIL |
| **postgres** | `postgres:16-alpine` | 5432 | 5432 | Shared DB — three logical DBs (`platform`, `agent`, `mlflow`) |
| **redis** | `redis:7-alpine` | 6379 | 6379 | LangGraph checkpoint store + arq job queue |
| **mlflow** | `ghcr.io/mlflow/mlflow:v2.16.2` | 5000 | 5000 | Experiment tracking + model registry, backed by postgres |

**Why 7 containers (not fewer):** the requirements spec says *"the platform and the agent each survive the other crashing."* That requires per-service process isolation. Stateful infra (postgres, redis, mlflow) is split out for independent lifecycle/upgrade. See DECISIONS.md for the full reasoning.

---

## 3. End-to-end flow (the Friday demo)

```
1. send_traffic.py POSTs 50 normal /predict requests
       │
       ▼
   platform.PredictionService stores each prediction in `predictions` table
       │
       ▼
2. send_traffic.py POSTs /api/v1/drift/check
       │
       ▼
   platform.DriftService reads last 200 predictions, computes PSI/chi², severity = "none"
       │
       ▼
3. send_traffic.py POSTs 200 SHIFTED /predict requests (mutates 1 numeric + 1 categorical column)
       │
       ▼
4. send_traffic.py POSTs /api/v1/drift/check
       │
       ▼
   platform.DriftService recomputes — severity now "medium"/"high"
   Severity changed AND not insufficient_data → fire webhook
       │
       ▼
   platform.WebhookService POSTs HMAC-signed payload to http://agent:8000/webhooks/drift
       │
       ▼
5. agent.api.webhooks verifies HMAC → 202 Accepted → background task
       │
       ▼
6. agent inserts row in `investigations` table → schedules LangGraph run
       │
       ▼
   LangGraph supervisor: triage (LLM) → action (LLM + tools) → comms (LLM)
       │
       ▼
7. action node enqueues a job (e.g. retrain) on redis via arq
       │
       ▼
   worker pulls the job, runs it, writes result back to postgres
       │
       ▼
8. agent pauses for human approval → posts to HIL inbox
       │
       ▼
9. Reviewer clicks Approve in dashboard
       │
       ▼
10. agent calls platform's POST /api/v1/promote with bearer token
       │
       ▼
   platform.PromotionService updates `model_registry_records` → new version is Production
       │
       ▼
11. Dashboard auto-refreshes → Registry tab reflects the promotion
```

Total Groq token spend per investigation: ~6k–13k (3–5 LLM calls across triage/action/comms).

---

## 4. Data ownership

| Data | Lives where | Owner |
|---|---|---|
| `predictions` table | postgres / `platform` DB | platform |
| `model_registry_records` table | postgres / `platform` DB | platform |
| `drift_reports`, `drift_alerts`, `reference_statistics` | postgres / `platform` DB | platform |
| `investigations` table | postgres / `agent` DB | agent |
| LangGraph checkpoints | postgres / `agent` DB | agent (LangGraph PostgresSaver) |
| arq job queue + DLQ | redis | worker |
| MLflow experiments + registered models | postgres / `mlflow` DB + `mlflow_artifacts` volume | mlflow service |
| Trained model artifacts (model.pkl, schema.json, etc.) | `platform/artifacts/model_v1/` (mounted into platform container) | platform |

---

## 5. Service-to-service contracts

**platform → agent (drift webhook)**
- `POST http://agent:8000/webhooks/drift`
- Headers: `X-Driftwatch-Signature: sha256=<hmac>` (HMAC-SHA256 of body using `WEBHOOK_HMAC_SECRET`)
- Body: `DriftEventV1` JSON
- Agent returns `202 Accepted` immediately, processes async

**agent → platform (promotion)**
- `POST http://platform:8000/api/v1/promote`
- Headers: `Authorization: Bearer <PROMOTION_BEARER_TOKEN>`
- Body: `{model_name, model_version}`
- Platform validates the day-4 promotion gate before updating registry

**agent → platform (backfill on boot)**
- `GET http://platform:8000/api/v1/drift/recent?since=<iso8601>`
- Used after agent restart to catch any drift events emitted while it was down

**dashboard → platform / agent (read-only)**
- Polls `/api/v1/registry/state`, `/api/v1/registry/history` on platform
- Polls `/investigations`, `/queue/dlq`, `/hil/pending` on agent
- Auto-refresh every 5s

---

## 6. Persistence & restart behavior

- **postgres volume** (`postgres_data`) — survives `docker compose down`, lost only on `down -v`
- **redis volume** (`redis_data`) — same
- **MLflow artifacts** (`mlflow_artifacts`) — same
- **Agent restart mid-investigation** — LangGraph PostgresSaver checkpoints state per node, so a kill-and-restart resumes from the last completed node, not from scratch (per spec requirement)
- **Platform restart** — predictions and drift reports persist; serving model is reloaded from `model.pkl` on first `/predict` call

---

## 7. Out of scope / not built

- Authentication for end users (only service-to-service auth via HMAC + bearer)
- Multi-tenant model registry (single registered model name)
- Real-time training (training is on-demand via worker queue, not streaming)
- Drift alerting on a schedule (drift checks are triggered on-demand; production would add cron/APScheduler — see DECISIONS.md)