"""Conversation + slot-filling state shared across the LangGraph agent's nodes.

Persisted per-thread by the LangGraph checkpointer (see app/memory), so multi-turn refinement
("only show Uzbekistan Airways", "pick the cheapest") can read the last search's cached results
and active filters without re-calling the Travelpayouts API.
"""
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class ToolLogEntry(TypedDict):
    tool: str
    params: dict[str, Any]
    result_summary: str
    served_from_cache: bool


class SearchParams(TypedDict):
    origin: str
    destination: str
    date: str
    adults: int
    children: int
    cabin_class: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # Unfiltered flight pool accumulated across every flight_search_tool call so far for the
    # current route (same origin+destination, e.g. across multiple dates) — cleared and
    # restarted when the origin/destination actually changes to a new route. This is what lets
    # compare_flights_tool genuinely rank across multiple real searches, not just the latest one.
    last_search_raw: list[dict]
    # The search params (route/date/pax/cabin) of the MOST RECENT flight_search_tool call — used
    # to detect an exact repeat (pure cache hit, no API call) vs. a same-route accumulation vs. a
    # brand new route (which resets last_search_raw).
    last_search_params: Optional[SearchParams]
    # The currently "active" view: last_search_raw narrowed by active_filters / compare_flights_tool
    # airline-or-flight-number selections. This is what follow-up turns like "pick the cheapest"
    # operate on.
    last_search_results: list[dict]
    active_filters: dict[str, Any]

    last_comparison: Optional[dict]
    tool_logs: list[ToolLogEntry]
    recent_searches: list[SearchParams]


def new_state(messages: Optional[list] = None) -> AgentState:
    return AgentState(
        messages=messages or [],
        last_search_raw=[],
        last_search_params=None,
        last_search_results=[],
        active_filters={},
        last_comparison=None,
        tool_logs=[],
        recent_searches=[],
    )
