"""Snapshot trajectory tests (TEST-01, D-22, D-23).

For each fixture under tests/snapshots/, run the graph with FakeChatModel +
MemorySaver and assert the trajectory matches fixture["expected_trajectory"].

Run with REGEN=1 to rewrite fixtures from the actual run output.
"""

import json
import os
import pathlib
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from contracts.v1 import DriftEventV1

from app.graph.builder import build_graph
from app.testing.fakes import FakeChatModel

# absolute path to the snapshot directory; collect all .json fixtures up front
SNAPSHOT_DIR = pathlib.Path(__file__).parent / "snapshots"
FIXTURES = sorted(SNAPSHOT_DIR.glob("*.json"))


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
# load fixture -> run graph -> record trajectory -> assert (or regen)
async def test_snapshot_trajectory(fixture_path: pathlib.Path) -> None:
    fixture: dict[str, Any] = json.loads(fixture_path.read_text(encoding="utf-8"))
    # validate the input through the canonical contract before feeding the graph
    drift_event = DriftEventV1.model_validate(fixture["input"])
    # FakeChatModel pops responses indexed by call order (D-20)
    fake = FakeChatModel(responses=list(fixture["fake_llm_responses"]))
    # MemorySaver replaces AsyncPostgresSaver in tests — no live DB needed
    graph = build_graph(checkpointer=MemorySaver(), chat_model=fake)
    # config carries thread_id (D-02) AND re-binds chat_model so astream sees the test fake
    config = {
        "configurable": {
            "thread_id": fixture["scenario"],
            "chat_model": fake,
        }
    }

    # initial invoke — runs until terminal or interrupt
    initial_state = {
        "investigation_id": fixture["scenario"],
        "drift_event": drift_event.model_dump(mode="json"),
        "current_node": "triage",
        "created_at": drift_event.emitted_at.isoformat(),
        "updated_at": drift_event.emitted_at.isoformat(),
    }
    # collect (node, current_node) tuples from astream's update events
    trajectory: list[dict[str, Any]] = []
    async for chunk in graph.astream(
        initial_state, config=config, stream_mode="updates"
    ):
        # interrupt events arrive as tuples; node updates as dicts. Handle both.
        if not isinstance(chunk, dict):
            trajectory.append({"node": "__interrupt__", "current_node": "awaiting_hil"})
            continue
        for node_name, update in chunk.items():
            current = (update or {}).get("current_node", "(unchanged)") if isinstance(update, dict) else "(unchanged)"
            trajectory.append({"node": node_name, "current_node": current})

    # resume if HIL fixture supplies a resume_payload
    if "resume_payload" in fixture:
        async for chunk in graph.astream(
            Command(resume=fixture["resume_payload"]),
            config=config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                trajectory.append({"node": "__interrupt__", "current_node": "awaiting_hil"})
                continue
            for node_name, update in chunk.items():
                current = (update or {}).get("current_node", "(unchanged)") if isinstance(update, dict) else "(unchanged)"
                trajectory.append({"node": node_name, "current_node": current})

    # regen mode: rewrite fixture and pass — gated on env var so CI never silently rewrites
    if os.environ.get("REGEN") == "1":
        fixture["expected_trajectory"] = trajectory
        fixture_path.write_text(
            json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
        )
        return

    # hard equality on the recorded vs expected trajectory
    assert trajectory == fixture["expected_trajectory"], (
        f"Trajectory drift in {fixture_path.name}.\n"
        f"Expected: {json.dumps(fixture['expected_trajectory'], indent=2)}\n"
        f"Got:      {json.dumps(trajectory, indent=2)}\n"
        f"Re-record with REGEN=1 uv run pytest tests/test_snapshots.py"
    )
