"""LangGraph graph definition: agent <-> tools loop, checkpointed for multi-turn memory.

    START -> agent -> (has tool_calls?) -> tools -> agent -> ... -> (no tool_calls) -> END

`agent` does both intent parsing (deciding which tool to call) and response formatting
(turning tool results back into a natural-language answer on the loop-back). `tools` runs the
real Travelpayouts / exchange-rate calls, or refines already-cached results, and never lets the LLM
fabricate data.
"""
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import agent_node, route_after_agent, tools_node
from app.agent.state import AgentState
from app.memory import get_checkpointer


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=get_checkpointer())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
