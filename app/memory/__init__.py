"""Short-term conversation memory: a process-wide LangGraph checkpointer plus thread-id helpers.

The actual state (message history, cached search results, active filters — see app/agent/state.py)
lives inside the LangGraph checkpoint itself, keyed by thread_id. One thread_id == one ongoing
conversation with its own last-search cache, so multiple Streamlit sessions/users don't bleed
into each other's context.
"""
import uuid

from langgraph.checkpoint.memory import MemorySaver

_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    return _checkpointer


def new_thread_id() -> str:
    return str(uuid.uuid4())
