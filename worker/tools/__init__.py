"""Worker tool functions — replay, retrain, rollback (D-06..D-08).

Each tool is registered on WorkerSettings.functions and dispatched by name when the
agent's action node calls `arq_pool.enqueue_job("replay" | "retrain" | "rollback", ...)`.
"""