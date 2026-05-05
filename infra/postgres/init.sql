-- DriftWatch Postgres init — runs once on first boot of the postgres container.
-- Mounted at /docker-entrypoint-initdb.d/init.sql:ro by docker-compose.yml.
-- Creates the three logical databases used by platform, agent, and mlflow services
-- (D-11). Schema migrations are owned by the respective service phases — Phase 1
-- creates the MLflow schema, Phase 2 creates the platform schema, Phase 3 creates
-- the agent schema (LangGraph checkpointer tables).

CREATE DATABASE platform;
CREATE DATABASE agent;
CREATE DATABASE mlflow;
