"""Worker-side writes to the agent's investigations.state JSONB column (D-02).

Tools call merge_result_into_state(session, investigation_id, "replay_result", {...})
which reads the row, shallow-merges {key: payload} into the state dict, and commits.
The agent's GET /investigations sees the update on next poll — no callback endpoint.
"""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Investigation


# merge {result_key: payload} into investigations.state JSONB; no-op if row missing
async def merge_result_into_state(
    session: AsyncSession,
    investigation_id: uuid.UUID,
    result_key: str,
    payload: dict[str, Any],
) -> None:
    # primary-key lookup; tools that target a non-existent investigation simply log + return
    row = await session.get(Investigation, investigation_id)
    if row is None:
        # missing investigation_id is non-fatal — caller logs the no-op and continues
        return
    # shallow-merge new key into existing state dict (preserve other fields)
    new_state = dict(row.state) if isinstance(row.state, dict) else {}
    new_state[result_key] = payload
    row.state = new_state
    # commit so the agent's GET /investigations sees the update on next poll
    await session.commit()