# DECISIONS

Architectural and process decisions made during DriftWatch development. Each entry covers what we chose, why, and what the alternative would have cost.

---

## D-01 — 7 separate containers (not bundled)

**Chosen:** platform, agent, worker, dashboard, postgres, redis, mlflow each in their own container.

**Why:** The spec requires *"the platform and the agent each survive the other crashing"*. Process isolation is the only way to guarantee that. Bundling would also block independent scaling (`docker compose up --scale worker=4` matters for backfills) and cleanly map to the pair-project ownership split (one half per partner).

**Cost of alternative:** A single bundled container would have lower RAM overhead but fails the spec requirement outright.

---

## D-02 — Single shared Postgres with three logical DBs

**Chosen:** One `postgres:16-alpine` container hosting `platform`, `agent`, `mlflow` databases.

**Why:** Smaller image footprint, single backup target, simpler `.env` for local dev. The three services own disjoint schemas, so logical separation is enough.

**Cost of alternative:** Three separate Postgres containers would have full isolation but triple the memory and complicate compose health checks.

---

## D-03 — Redis serves two roles

**Chosen:** Single `redis:7-alpine` container is both the LangGraph checkpoint store and the arq job queue.

**Why:** Redis handles both workloads natively; the keyspaces don't collide. One fewer dependency to operate.

**Cost of alternative:** Separate Redis instances would be cleaner but materially overkill for a 5-minute demo workload.

---

## D-04 — MLflow registry, not custom

**Chosen:** Use MLflow's built-in model registry (`registered_model_name="driftwatch-bank-marketing"`).

**Why:** Spec says *"Register the fitted pipeline in MLflow"*. Reusing MLflow's registry table avoids reinventing version tracking, stage transitions, and the artifact triple (binary + schema + card).

**Cost of alternative:** A custom registry table would be one less moving part but also one more thing to test and reason about.

---

## D-05 — On-demand drift checks (no scheduler)

**Chosen:** `POST /api/v1/drift/check` only runs when called. The demo script triggers it explicitly after Phase 1 and Phase 2.

**Why:** A scheduler (cron / APScheduler) is the production-correct answer but adds a moving part with no graded benefit for a demo. Triggering from the script keeps the Friday demo deterministic and observable.

**Cost of alternative:** Production deployment will need a scheduler. Documented here so that's an explicit known gap, not an oversight.

---

## D-06 — HMAC-signed webhooks (platform → agent)

**Chosen:** `WEBHOOK_HMAC_SECRET` shared via `.env`. Platform signs every drift webhook with HMAC-SHA256; agent rejects unsigned or mis-signed payloads.

**Why:** Service-to-service trust without exposing the agent's webhook endpoint to the public internet. Cheap, correct, no extra infra.

**Cost of alternative:** mTLS or a real auth proxy would be stronger but disproportionate for a single-host docker-compose project.

---

## D-07 — Bearer token for promotion endpoint

**Chosen:** Agent calls `POST /api/v1/promote` with `Authorization: Bearer ${PROMOTION_BEARER_TOKEN}`.

**Why:** Promotion is the most consequential action in the system (changes which model serves real predictions). A separate token makes audit logs and revocation cleaner than reusing the HMAC secret.

**Cost of alternative:** Reusing the HMAC secret would be one less thing in `.env` but conflates "drift signal" auth with "production change" auth.

---

## D-08 — Streamlit dashboard polls every 5s

**Chosen:** Dashboard auto-refreshes via `streamlit-autorefresh` at 5-second intervals. Each tab independently calls platform/agent read endpoints.

**Why:** Simple, no websockets, no extra infra. 5s is fast enough that the Friday demo feels live without hammering the APIs.

**Cost of alternative:** WebSockets / SSE would be lower latency but require additional plumbing in both dashboard and platform/agent.

---

## D-09 — Each dashboard tab owns its own try/except

**Chosen:** A dead service in one tab does not crash the whole dashboard page (panels in `dashboard/lib/panels.py` each catch their own errors).

**Why:** The reviewer must always be able to reach the HIL inbox even if MLflow or the agent is misbehaving. Page-level crashes would block approvals.

**Cost of alternative:** A single shared error handler is shorter but creates correlated failure modes.

---

## D-10 — Single `.env` shared by all services

**Chosen:** One `.env` at repo root, `env_file: .env` in compose for every service. Dashboard's `pydantic-settings` uses `extra="ignore"` so unknown vars don't crash boot.

**Why:** One file to copy from `.env.example`, one place to rotate secrets. Each service's settings model only declares the keys it actually uses.

**Cost of alternative:** Per-service env files would be cleaner separation but require copying multiple templates and risk drift between them.

---

## D-11 — Demo-time prediction traffic comes from `scripts/send_traffic.py`

**Chosen:** A standalone uv project under `scripts/` with two-phase logic (50 normal → drift check → 200 shifted → drift check). Mutates one numeric column and one categorical column in-flight.

**Why:** Spec demands *"Shift one numeric and one categorical live during the demo"*. A script is reproducible, scriptable, and pauses for `Enter` between phases so the presenter controls the moment of drift.

**Cost of alternative:** Manually clicking Swagger 200+ times is not viable; randomized natural drift would be unpredictable for a 5-minute demo.

---

## D-12 — Phase 1 baseline traffic is non-negotiable for the demo

**Chosen:** Always send 50 normal requests + drift check before sending shifted traffic.

**Why:** [drift_service.py:147–155](platform/app/services/drift_service.py#L147-L155) refuses to fire a webhook if `previous_severity == "insufficient_data"`. Without Phase 1, the first severity computation skips straight to `medium+` but no webhook fires, killing the demo.

**Cost of alternative:** None — this is a hard requirement of the existing code path.

---

## D-13 — `dashboard/lib/` exception in `.gitignore`

**Chosen:** `.gitignore` line 17 ignores any `lib/` (Python build artifact convention). We added an explicit `!dashboard/lib/` and `!dashboard/lib/**` exception.

**Why:** Phase 5 dashboard ships its panel renderers as source modules under `dashboard/lib/`. Without the exception, those files would silently never reach the repo, breaking the dashboard image build on PR.

**Cost of alternative:** Renaming the directory would also work but breaks established `dashboard/lib/` import paths.

---

## D-14 — Root `.dockerignore` covers all four service builds

**Chosen:** A single `.dockerignore` at repo root. All four service Dockerfiles use `context: .`, so this file controls what every image build sees.

**Why:** Keeps build context KB instead of GB (`.venv/` directories alone were hundreds of MB). Also prevents secrets — `.env`, planning docs in `docs - DO NOT COMMIT/`, and `.git/` — from ever entering an image layer.

**Cost of alternative:** Per-service `.dockerignore` files would be more granular but harder to keep consistent.

---

## D-15 — User does git operations manually; tooling never commits

**Chosen:** All `git add` / `commit` / `push` / `tag` operations are run by the human, never by automation.

**Why:** Fewer surprises, no risk of an automated commit including the wrong file (`.env`, planning docs). Also matches the partner's working style.

**Cost of alternative:** Faster iteration but higher blast radius if something automated commits a secret.

---

## D-16 — One PR for the entire Phase 5 dashboard

**Chosen:** `feat: Phase 5 dashboard (registry, investigations, queue, HIL) + .dockerignore` shipped as a single PR.

**Why:** The dashboard pieces are tightly coupled — `app.py` imports from `lib/`, `lib/api.py` reads from `config.py`, the Dockerfile copies all of them, and the new pyproject deps are required by the new code. Splitting would have shipped broken intermediate states.

**Cost of alternative:** Smaller PRs are easier to review individually but each one would have failed CI in isolation.

---

## Spec deviations (acknowledged)

These are gaps where what we shipped does not match the requirements PDF exactly:

### SD-01 — CI is a no-op stub

The spec requires CI to *"build images, run agent snapshot trajectory tests with a mocked LLM, run a 1e-12 fidelity replay test against the model. Refuse to merge if any of these regress."*

Our `.github/workflows/ci.yml` is currently a placeholder that only validates the workflow YAML parses. Real test wiring was descoped given the timeline. Tests exist in the repo (`platform/tests/`, `agent/tests/`) and pass locally; they're just not wired into the merge gate.

### SD-02 — On-demand drift checks (see D-05)

Spec implies the platform watches its own predictions; we trigger checks from the demo script.

### SD-03 — Dataset version — RESOLVED

Original concern: the repo shipped the older `bank-full.csv` (17 columns) instead of the spec-required `bank-additional-full.csv` (20 columns including the macroeconomic features `euribor3m`, `cons.price.idx`, etc.).

**Resolved** in PR #9 (Phase 3.1). The dataset was swapped, the `pdays` sentinel handling was updated from `-1` to `999`, the derived field `pdays_was_minus_one` was renamed to `pdays_was_999`, and the model was retrained. Raw CSVs are now gitignored (only `bank-additional-names.txt` metadata is tracked); contributors download the dataset from UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing.

### SD-04 — Train/test split — RESOLVED

Original concern: the spec calls for stratified 60/20/20 (train/validation/test). The repo used 70/30 with no validation split.

**Resolved** in PR #9 (Phase 3.1). [platform/app/ml/data.py](platform/app/ml/data.py) now exposes `make_train_validation_test_split` with explicit `SPLIT_TRAIN_SIZE=0.60`, `SPLIT_VALIDATION_SIZE=0.20`, `SPLIT_TEST_SIZE=0.20` env knobs in `.env.example`. The legacy two-way split helper is preserved for callers that still need it.

### SD-05 — Webhook payload contract aligned with `DriftEventV1`

The platform's `_create_alert` originally produced a custom payload shape (nested `window.start/end`, severity labels `none/low/medium/high`, string `model_version`, `signals` object). The agent's `contracts/v1.py::DriftEventV1` requires a flat structure with `green/yellow/red` severities, integer `model_version`, and a `top_metrics: list[DriftMetric]` field. The mismatch was caught during end-to-end smoke testing of `send_traffic.py` — the webhook fired but the agent rejected with `400 webhook_body_invalid`.

[platform/app/services/drift_service.py](platform/app/services/drift_service.py) now maps internal severities through `_CONTRACT_SEVERITY` (green/yellow/red collapse), flattens the window fields, casts version strings via `_coerce_model_version`, and builds `top_metrics` from the highest-magnitude PSI/chi² signals via `_build_top_metrics`. The `DriftAlert` row still stores the old fields it cares about — only the *outbound* webhook payload changed.