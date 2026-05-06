"""Graph builder.

Compiles the supervisor topology:

    START -> triage -> [supervisor] -> action -> [supervisor] -> comms -> END
                       |                                  ^
                       v                                  |
                       comms (no_action) -----------------+
                       (rollback) -> action interrupts for HIL -> END (resumed by /hil/*)

`route_supervisor` is the conditional-edge function — pure, no LLM (D-05).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import action_node, comms_node, triage_node
from app.graph.state import InvestigationState
from app.graph.supervisor import route_supervisor


# build and compile the supervisor graph
def build_graph(
    checkpointer: BaseCheckpointSaver,
    chat_model: BaseChatModel,
) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(InvestigationState)
    graph.add_node("triage", triage_node)
    graph.add_node("action", action_node)
    graph.add_node("comms", comms_node)

    # entry: every fresh investigation starts at triage
    graph.add_edge(START, "triage")

    # after triage: route via supervisor (returns "action" or "comms")
    graph.add_conditional_edges("triage", route_supervisor, {"action": "action", "comms": "comms", END: END})
    # after action: route via supervisor (returns "comms" or END for awaiting_hil)
    graph.add_conditional_edges("action", route_supervisor, {"comms": "comms", END: END})
    # after comms: terminal
    graph.add_edge("comms", END)

    # compile the graph; chat_model is attached as an attribute so graph_runner can read it
    # and inject it into per-call config (with_config produces a RunnableBinding which loses
    # CompiledStateGraph methods like aget_state — keep the compiled object intact instead)
    compiled = graph.compile(checkpointer=checkpointer)
    compiled.chat_model = chat_model  # type: ignore[attr-defined]
    return compiled
