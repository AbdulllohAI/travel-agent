"""Flight Comparison Tool: compares 2+ cached flight offers and recommends one, with reasoning.

The LLM never sees or re-types raw flight data for this tool. Its args_schema only carries the
*selection/comparison criteria* (which airlines/flight numbers, which metric); the agent's tool
execution node injects the actual cached flights (from graph state, populated by a prior
flight_search_tool call) before compare_flights() runs. This keeps price/time data flowing only
from the real API response, never from the LLM.
"""
from typing import List, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

ComparisonMetric = Literal["price", "duration", "departure_time", "arrival_time", "stops"]

_METRIC_LABELS = {
    "price": "cheapest",
    "duration": "shortest",
    "stops": "fewest stops",
    "departure_time": "earliest departure",
    "arrival_time": "earliest arrival",
}


class CompareFlightsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    airlines: Optional[List[str]] = Field(
        None, description="Only compare flights operated by these airline names (case-insensitive substring match)"
    )
    flight_numbers: Optional[List[str]] = Field(
        None, description="Only compare these specific flight numbers, e.g. ['HY123', 'TK372']"
    )
    metric: ComparisonMetric = Field(
        "price", description="Metric to recommend on: price, duration, departure_time, arrival_time, or stops"
    )


class NotEnoughFlightsError(Exception):
    """Raised when fewer than 2 flights are available to compare."""


def _duration_to_minutes(duration_str: str) -> int:
    """'3h 25m' -> 205"""
    total = 0
    for token in duration_str.split():
        if token.endswith("h"):
            total += int(token[:-1]) * 60
        elif token.endswith("m"):
            total += int(token[:-1])
    return total


def filter_flights_for_comparison(
    flights: List[dict],
    airlines: Optional[List[str]] = None,
    flight_numbers: Optional[List[str]] = None,
) -> List[dict]:
    result = flights
    if airlines:
        wanted = [a.lower() for a in airlines]
        result = [f for f in result if any(w in f["airline"].lower() for w in wanted)]
    if flight_numbers:
        wanted_fn = {fn.upper() for fn in flight_numbers}
        result = [f for f in result if f["flight_number"].upper() in wanted_fn]
    return result


def compare_flights(flights: List[dict], metric: ComparisonMetric = "price") -> dict:
    """Pure function: ranks flights by metric and returns a comparison table + recommendation."""
    if len(flights) < 2:
        raise NotEnoughFlightsError(
            f"Need at least 2 flights to compare, found {len(flights)} matching the given filters."
        )

    def sort_key(f: dict):
        if metric == "duration":
            return _duration_to_minutes(f["duration"])
        if metric in ("price", "stops"):
            return f[metric]
        return f[metric]  # departure_time / arrival_time: ISO strings sort chronologically

    ranked = sorted(flights, key=sort_key)
    best = ranked[0]

    table = [
        {
            "airline": f["airline"],
            "flight_number": f["flight_number"],
            "departure_time": f["departure_time"],
            "arrival_time": f["arrival_time"],
            "duration": f["duration"],
            "stops": f["stops"],
            "cabin_class": f.get("cabin_class", "economy"),
            "price": f["price"],
            "currency": f.get("currency", "USD"),
        }
        for f in flights
    ]

    recommendation = (
        f"{best['airline']} {best['flight_number']} is the {_METRIC_LABELS[metric]} option "
        f"({best['price']} {best.get('currency', 'USD')}, {best['duration']}, {best['stops']} stop(s))."
    )

    return {"table": table, "recommendation": recommendation, "best_flight": best}


@tool("compare_flights_tool", args_schema=CompareFlightsInput)
def compare_flights_tool(
    airlines: Optional[List[str]] = None,
    flight_numbers: Optional[List[str]] = None,
    metric: ComparisonMetric = "price",
) -> dict:
    """Compare 2+ flights already returned by a prior flight_search_tool call in this conversation.
    Never fabricates flight data — only compares real cached offers from the last search."""
    raise RuntimeError(
        "compare_flights_tool must be executed by the agent's tool-execution node (see "
        "app/agent/nodes.py), which injects cached flight results from state before calling "
        "compare_flights(). It should never be invoked directly with no data to compare."
    )
