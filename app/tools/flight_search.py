"""Flight Search Tool: real flight fares from the Travelpayouts/Aviasales price-data API
(`/v1/prices/cheap`).

The LLM only supplies search parameters. Every field in the returned flights (airline, price,
times, ...) comes straight from the Travelpayouts API response — nothing is invented here.

Honesty note: this is a free, no-approval data endpoint that returns the *cheapest cached fare*
per route/date (economy, 1 adult) rather than a full live multi-airline inventory search like a
GDS (e.g. Amadeus). That means a single search often returns only one or a handful of fares,
never fabricated ones. If the user asks for a passenger count/cabin class this endpoint can't
price, we say so explicitly rather than silently pretending the price accounts for it.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_FLIGHT_RESULTS
from app.tools.travelpayouts_client import (
    TravelpayoutsClient,
    TravelpayoutsError,
    get_travelpayouts_client,
)

logger = logging.getLogger(__name__)

TimeOfDay = Literal["morning", "afternoon", "evening"]
CabinClass = Literal["economy", "premium_economy", "business", "first"]

_airline_names_cache: Optional[dict[str, str]] = None


def _get_airline_name(code: str, client: TravelpayoutsClient) -> str:
    """Resolves a 2-letter IATA carrier code to its real airline name via Travelpayouts'
    reference data (cached process-wide) — so the LLM is never left to guess/invent a name for
    a bare code like 'HY'. Falls back to the raw code if the lookup fails or the code is unknown."""
    global _airline_names_cache
    if _airline_names_cache is None:
        try:
            data = client.get_reference_json("/en/airlines.json")
            _airline_names_cache = {a["code"]: a["name"] for a in data if a.get("code")}
        except (TravelpayoutsError, TypeError, KeyError):
            _airline_names_cache = {}
    return _airline_names_cache.get(code, code)


def reset_airline_names_cache() -> None:
    """Clears the cached airline code->name lookup. Used by tests."""
    global _airline_names_cache
    _airline_names_cache = None

SEARCH_CURRENCY = "USD"


class FlightSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(..., description="Origin airport/city IATA code, e.g. 'TAS'")
    destination: str = Field(..., description="Destination airport/city IATA code, e.g. 'IST'")
    date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    adults: int = Field(1, ge=1, le=9, description="Number of adult passengers")
    children: int = Field(0, ge=0, le=8, description="Number of child passengers (2-11 years old)")
    cabin_class: Optional[CabinClass] = Field(None, description="Preferred cabin class")
    max_stops: Optional[int] = Field(None, ge=0, le=3, description="Maximum number of stops allowed")
    time_of_day: Optional[TimeOfDay] = Field(
        None,
        description="Preferred departure time window: morning (00:00-12:00), afternoon (12:00-18:00), "
        "evening (18:00-24:00)",
    )
    direct_only: Optional[bool] = Field(None, description="If true, only return non-stop flights")

    @field_validator("origin", "destination")
    @classmethod
    def _upper_code(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        try:
            parsed = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"date must be in YYYY-MM-DD format, got '{v}'") from exc
        if parsed < date.today():
            raise ValueError(f"date '{v}' is in the past")
        return v


class NoFlightsFoundError(Exception):
    """Raised when the search succeeded but no flights matched the route/date/filters."""


def _time_bucket(iso_datetime: str) -> str:
    hour = datetime.fromisoformat(iso_datetime).hour
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _flatten_price_entries(node) -> list[dict]:
    """Travelpayouts nests price entries under varying key structures (by destination, by date,
    or both) depending on query shape. Recursively collect every leaf dict that looks like an
    actual fare (has a 'price' key), regardless of nesting."""
    if isinstance(node, dict):
        if "price" in node:
            return [node]
        entries = []
        for value in node.values():
            entries.extend(_flatten_price_entries(value))
        return entries
    if isinstance(node, list):
        entries = []
        for item in node:
            entries.extend(_flatten_price_entries(item))
        return entries
    return []


def _parse_entry(entry: dict, params: FlightSearchInput, client: TravelpayoutsClient) -> Optional[dict]:
    price = entry.get("price") or entry.get("value")
    if price is None:
        return None

    airline_code = entry.get("airline", "")
    flight_number_raw = entry.get("flight_number", "")
    flight_number = f"{airline_code}{flight_number_raw}".strip() or "N/A"
    airline_name = _get_airline_name(airline_code, client) if airline_code else "Unknown"

    departure_time = entry.get("departure_at")
    # "duration" is the round-trip total when the cached fare has a return_at; "duration_to" is
    # always the one-way outbound leg we actually want (confirmed against the live API).
    duration_minutes = entry.get("duration_to", entry.get("duration"))
    arrival_time = None
    if departure_time and duration_minutes is not None:
        try:
            arrival_time = (
                datetime.fromisoformat(departure_time) + timedelta(minutes=int(duration_minutes))
            ).isoformat()
        except ValueError:
            arrival_time = None

    duration_label = f"{int(duration_minutes) // 60}h {int(duration_minutes) % 60}m" if duration_minutes else "N/A"

    return {
        "airline": airline_name,
        "airline_code": airline_code,
        "flight_number": flight_number,
        "departure_airport": entry.get("origin_airport") or entry.get("origin") or params.origin,
        "arrival_airport": entry.get("destination_airport") or entry.get("destination") or params.destination,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "duration": duration_label,
        "stops": entry.get("transfers", entry.get("number_of_changes", 0)) or 0,
        "cabin_class": "economy",
        "price": float(price),
        "currency": SEARCH_CURRENCY,
    }


def apply_client_filters(flights: list[dict], params: FlightSearchInput) -> list[dict]:
    """Applies filters the API doesn't take as request params (max_stops, time_of_day) and sorts
    by price. Used both right after a fresh API call and to refine an already-cached raw result
    set without hitting the API again."""
    results = flights
    if params.max_stops is not None:
        results = [f for f in results if f["stops"] <= params.max_stops]
    if params.time_of_day is not None:
        results = [f for f in results if f["departure_time"] and _time_bucket(f["departure_time"]) == params.time_of_day]
    if params.direct_only:
        results = [f for f in results if f["stops"] == 0]
    return sorted(results, key=lambda f: f["price"])


def search_flights_raw(params: FlightSearchInput, client: Optional[TravelpayoutsClient] = None) -> list[dict]:
    """Calls Travelpayouts and returns the full parsed (unfiltered, untruncated) fare list.
    Callers that need to cache results for later client-side filtering should use this."""
    client = client or get_travelpayouts_client()

    query = {
        "origin": params.origin,
        "destination": params.destination,
        "depart_date": params.date,
        "currency": SEARCH_CURRENCY,
    }

    raw = client.get_prices("/v1/prices/cheap", params=query)
    entries = _flatten_price_entries(raw.get("data", {}))
    flights = [f for f in (_parse_entry(e, params, client) for e in entries) if f is not None]
    return flights


def search_flights(params: FlightSearchInput, client: Optional[TravelpayoutsClient] = None) -> list[dict]:
    """Pure function: calls Travelpayouts, returns parsed+filtered flight dicts. Raises on no results."""
    flights = apply_client_filters(search_flights_raw(params, client), params)[:MAX_FLIGHT_RESULTS]

    if not flights:
        raise NoFlightsFoundError(
            f"No flights found from {params.origin} to {params.destination} on {params.date} "
            "matching the given filters."
        )
    return flights


def _result_caveats(params: FlightSearchInput, flights: list[dict]) -> list[str]:
    notes = []
    if params.adults > 1 or params.children > 0 or params.cabin_class not in (None, "economy"):
        notes.append(
            "This free price-data API only returns economy fares priced for 1 adult. "
            "The prices below are an approximate per-adult economy reference, not an exact quote "
            f"for {params.adults} adult(s)"
            + (f", {params.children} child(ren)" if params.children else "")
            + (f", {params.cabin_class} class" if params.cabin_class not in (None, "economy") else "")
            + "."
        )
    if flights:
        notes.append(
            "This data source doesn't report stop count for these fares; 'stops' is shown as 0 "
            "(direct) by default, but the actual itinerary may include a connection — verify "
            "before booking."
        )
    return notes


@tool("flight_search_tool", args_schema=FlightSearchInput)
def flight_search_tool(
    origin: str,
    destination: str,
    date: str,
    adults: int = 1,
    children: int = 0,
    cabin_class: Optional[str] = None,
    max_stops: Optional[int] = None,
    time_of_day: Optional[str] = None,
    direct_only: Optional[bool] = None,
) -> dict:
    """Search real flight fares between two airports/cities on a given date via the Travelpayouts
    price-data API. Always returns real data from a live API call — never invents flights, prices,
    or times."""
    params = FlightSearchInput(
        origin=origin,
        destination=destination,
        date=date,
        adults=adults,
        children=children,
        cabin_class=cabin_class,
        max_stops=max_stops,
        time_of_day=time_of_day,
        direct_only=direct_only,
    )
    try:
        flights = search_flights(params)
        result = {"status": "ok", "flights": flights, "count": len(flights)}
        caveats = _result_caveats(params, flights)
        if caveats:
            result["note"] = " ".join(caveats)
        return result
    except NoFlightsFoundError as exc:
        return {"status": "no_results", "message": str(exc), "flights": []}
    except TravelpayoutsError as exc:
        return {"status": "error", "message": str(exc), "flights": []}
