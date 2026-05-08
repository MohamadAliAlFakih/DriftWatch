# DriftWatch

A self-healing MLOps stack for the UCI Bank Marketing classifier: a drift-aware model service, a LangGraph supervisor agent that investigates and acts on drift, a Redis-backed job queue, and a Streamlit dashboard with human-in-the-loop approvals.

Built for AIE Program — Week 5 (Drift Triage Co-Pilot).

---

## At a glance

```
predictions ──► platform ──drift webhook──► agent ──► redis ──► worker
                  ▲                          │
                  └──── promote ─────────────┘
                                              ↓ (HIL)
                  dashboard ◄─────────────────┘
```

- **Platform** (FastAPI + MLflow) — serves `/predict`, computes drift (PSI / chi²), gates promotion
- **Agent** (FastAPI + LangGraph + Postgres checkpoints) — investigates each drift event; supervisor topology with triage / action / comms sub-agents
- **Worker** (arq) — runs slow tools (replay test, retrain, rollback) from a Redis queue with idempotency keys and a DLQ
- **Dashboard** (Streamlit) — Registry, Investigations, Queue/DLQ, HIL Inbox tabs

---

## Quick start

```powershell
cp .env.example .env                  # then fill in GROQ_API_KEY + secrets
docker compose up -d --build
docker compose exec platform uv run alembic upgrade head
docker compose exec agent uv run alembic upgrade head
docker compose exec platform uv run python -m app.ml.train
```

Then open:

| | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| Platform API docs | http://localhost:8001/docs |
| Agent API docs | http://localhost:8002/docs |
| MLflow UI | http://localhost:5000 |

To run the live drift demo:

```powershell
uv run --project scripts scripts/send_traffic.py
```

Full instructions: see [RUNBOOK.md](RUNBOOK.md).

---

## Documentation

- **[ARCH.md](ARCH.md)** — system architecture, the 7 containers, end-to-end data flow, service-to-service contracts
- **[DECISIONS.md](DECISIONS.md)** — design decisions, trade-offs, and acknowledged spec deviations
- **[RUNBOOK.md](RUNBOOK.md)** — how to bring the stack up, run the demo, and recover from common failures

---

## Stack

| Layer | Tech |
|---|---|
| Model | scikit-learn (LogReg / RandomForest / HistGradientBoosting) + MLflow registry |
| Platform API | FastAPI, SQLAlchemy, Alembic, psycopg |
| Agent | FastAPI, LangGraph, langchain-groq |
| LLM | Groq `llama-3.3-70b-versatile` |
| Queue | arq on Redis 7 |
| Storage | Postgres 16 (3 logical DBs), Redis 7, MLflow 2.16 |
| Dashboard | Streamlit, streamlit-autorefresh, httpx |
| Packaging | uv (per-service `pyproject.toml`) |
| Containerization | Docker Compose (7 services) |

---

## Repo layout

```
.
├── platform/      # FastAPI model service (training, serving, drift, registry)
├── agent/         # FastAPI LangGraph supervisor + webhook receiver
├── worker/        # arq job consumer (slow tools)
├── dashboard/     # Streamlit reviewer console
├── contracts/     # shared Pydantic models for service-to-service contracts
├── scripts/       # demo helpers (send_traffic.py)
├── infra/         # postgres init.sql
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── README.md
├── ARCH.md
├── DECISIONS.md
└── RUNBOOK.md
```

---

## Authors

- **[ME]** — Mohamad Ali Al Fakih — agent / worker / dashboard / demo scripts
- **[SHE]** — Hasan — platform / training / model registry / drift detection

Both partners are responsible for the whole system at presentation time.

---

## Submission

- Tag: `v0.1.0-week5`
- Dataset: UCI Bank Marketing (`bank-full.csv` — see DECISIONS.md SD-03 if dataset swap is in flight)
- LLM: Groq `llama-3.3-70b-versatile` — chosen for low latency on agent supervisor calls