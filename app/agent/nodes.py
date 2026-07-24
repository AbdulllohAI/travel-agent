"""LangGraph node functions: model invocation (intent parsing + response formatting) and
tool execution (routing, real API calls, cache-aware filtering, and error handling).

Design note: the LLM never computes flight data itself. `agent_node` only decides which tool
to call and with what arguments; `tools_node` runs the real Travelpayouts/exchange-rate calls
(or refines already-cached results) and feeds raw results back as ToolMessages. The next
`agent_node` turn then interprets those real results into a natural-language answer — that is
the "response formatting" step, done by the same node on its loop-back.
"""
import json
import logging
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from app import config
from app.agent.state import AgentState
from app.tools.airport_search import (
    AirportSearchInput,
    NoLocationsFoundError,
    airport_search_tool,
    search_airports,
)
from app.tools.compare_flights import (
    CompareFlightsInput,
    compare_flights,
    compare_flights_tool,
    filter_flights_for_comparison,
)
from app.tools.currency import (
    CurrencyConversionError,
    CurrencyConversionInput,
    convert_currency,
    currency_conversion_tool,
)
from app.tools.flight_search import (
    FlightSearchInput,
    NoFlightsFoundError,
    apply_client_filters,
    flight_search_tool,
    search_flights_raw,
)
from app.tools.travelpayouts_client import TravelpayoutsError

logger = logging.getLogger(__name__)

TOOLS = [flight_search_tool, compare_flights_tool, airport_search_tool, currency_conversion_tool]

_CORE_SEARCH_FIELDS = ("origin", "destination", "date", "adults", "children", "cabin_class")

_SYSTEM_PROMPT_TEMPLATE = """You are an AI Travel Agent that helps users search for, filter, and \
compare real flights. You understand requests in any language the user writes in.

LANGUAGE RULE: always answer in the same language and script as the user's latest message. If \
the user writes in Uzbek, answer in Uzbek using the Latin alphabet (o'zbek lotin alifbosi) — \
never Cyrillic, never Russian, never Kyrgyz/Kazakh, never mix languages.

CRITICAL RULE: you must NEVER invent flight data, prices, times, or airport codes. Every fact \
about a flight must come from a tool call result. If you don't have the data, call a tool. Only \
state facts that are literally present in the tool's result fields — never pad your answer with \
ANY extra descriptive detail not present in that result, even if the detail is true and you know \
it from general knowledge. This includes trivia/superlatives (e.g. "the busiest/largest airport"), \
history, terminals, hub status, alliances, or aircraft type. If the tool result doesn't contain a \
field for it, do not mention it at all — just list the fields the tool actually gave you (name, \
IATA code, city, country for airports; airline, flight number, times, price, etc. for flights).

TOOL-CALLING DISCIPLINE: even if you can already guess the answer from earlier messages in this \
conversation, you MUST still call the appropriate tool for every new filtering/sorting/comparison \
request (e.g. "cheapest", "only Uzbekistan Airways", "earliest", "compare X and Y") — never answer \
such requests purely from memory. Only skip a tool call when your reply needs no flight/airport/ \
currency fact at all (e.g. greetings, or asking a clarifying question).

Today's date is {today}. Resolve relative/partial dates (e.g. "15-avgust", "next Friday") to \
YYYY-MM-DD using this as the reference date, and pick the nearest future occurrence.

PASSENGER COUNT RULE: read passenger counts carefully. "1 ta kattalar va 1 ta bola" / "1 adult and \
1 child" means adults=1 AND children=1 — never leave children at 0 when the user names a child \
passenger.

Available tools:
- flight_search_tool: search real flights for a route+date+passengers. Use it for a brand new \
route/date/passenger count, or to add stop/time-of-day/cabin/direct-only constraints.
- compare_flights_tool: filter and/or rank flights ALREADY in the cached search below — by \
airline name, specific flight numbers, or a metric (price/duration/departure_time/arrival_time/ \
stops). Use this for "only show X airline", "cheapest", "fastest", "earliest departure", or a \
head-to-head comparison. It never calls the flight API again — it only works on cached results, \
so if the cache below is empty you must call flight_search_tool first.
- airport_search_tool: resolve a city/airport name to IATA code(s), or an IATA code to its name.
- currency_conversion_tool: convert a price between currencies using a live exchange rate.

compare_flights_tool's arguments are EXACTLY: airlines (list of airline name strings, optional), \
flight_numbers (list of flight number strings, optional), metric (one of "price", "duration", \
"departure_time", "arrival_time", "stops"; default "price"). Never pass full flight objects or any \
other field name — the tool reads the actual cached flights itself; you only pick the filter/metric.

Examples of phrase -> compare_flights_tool metric mapping (always call the tool, in any language):
- "cheapest" / "eng arzon" -> {{"metric": "price"}}
- "fastest" / "shortest" / "eng tez" -> {{"metric": "duration"}}
- "earlier" / "earliest" / "ertaroq" / "erta uchadigan" -> {{"metric": "departure_time"}}
- "arrives soonest" / "eng erta yetib boradigan" -> {{"metric": "arrival_time"}}
- "fewest stops" / "kam to'xtaydigan" -> {{"metric": "stops"}}
Never answer these purely from memory — always issue the matching compare_flights_tool call, even \
if you already listed these flights earlier in the conversation.

CURRENCY RULE: "$" always means USD, "€" means EUR, "so'm"/"soʻm" means UZS. When a tool result \
gives you a price with a currency code, always restate that exact currency code — never silently \
relabel it as a different currency (e.g. never turn a USD price into "so'm" without actually \
calling currency_conversion_tool first). If the user wants a price in a different currency, call \
currency_conversion_tool with from_currency set to the currency the flight price is actually in.

If a required field is missing (e.g. no travel date given) and you cannot infer it from the \
conversation, do not guess — ask the user a short clarifying question instead of calling a tool.

Current cached search context:
{cache_summary}
"""


def _cache_summary(state: AgentState) -> str:
    results = state.get("last_search_results") or []
    params = state.get("last_search_params")
    filters = state.get("active_filters") or {}
    if not results:
        return "No cached flight results yet — a new user request needs a flight_search_tool call."
    route = f"{params['origin']}->{params['destination']}" if params else "this route"
    lines = [
        f"- {len(results)} flight(s) cached for {route} (accumulated across every search on this "
        "route so far this conversation, e.g. different dates) — most recent search: "
        f"{params}" if params else f"- {len(results)} flight(s) cached",
        f"- active filters currently applied: {filters}" if filters else "- no filters currently applied",
    ]
    return "\n".join(lines)


def _build_llm():
    """Reads app.config at call time (not import time) so a UI model selector can override
    config.OLLAMA_MODEL between requests without needing to reload this module."""
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.OLLAMA_TEMPERATURE,
    ).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Intent parsing (first pass) AND response formatting (on loop-back after tool results)."""
    llm = _build_llm()
    system = SystemMessage(
        content=_SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat(), cache_summary=_cache_summary(state))
    )
    ai_msg = llm.invoke([system, *state["messages"]])
    return {"messages": [ai_msg]}


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


def _log(tool_logs: list, name: str, params: dict, result: dict, from_cache: bool) -> None:
    tool_logs.append(
        {
            "tool": name,
            "params": params,
            "result_summary": json.dumps(result, default=str)[:2000],
            "served_from_cache": from_cache,
        }
    )


_MAX_ACCUMULATED_FLIGHTS = 200


def _merge_unique_flights(existing: list[dict], new: list[dict]) -> list[dict]:
    """Appends newly fetched flights to the accumulated pool for a route, skipping any that
    are already present (same flight_number + departure_time), and caps total size so a long
    session doesn't grow the pool unbounded."""
    seen = {(f["flight_number"], f["departure_time"]) for f in existing}
    merged = list(existing)
    for f in new:
        key = (f["flight_number"], f["departure_time"])
        if key not in seen:
            merged.append(f)
            seen.add(key)
    return merged[-_MAX_ACCUMULATED_FLIGHTS:]


def _handle_flight_search(args: dict, state: AgentState, updates: dict) -> tuple[dict, bool]:
    try:
        params = FlightSearchInput(**args)
    except ValidationError as exc:
        return {"status": "error", "message": f"Invalid or missing search parameters: {exc.errors()}"}, False

    core_now = {k: getattr(params, k) for k in _CORE_SEARCH_FIELDS}
    cached_core = state.get("last_search_params")
    has_cache = bool(state.get("last_search_raw"))
    exact_repeat = cached_core is not None and cached_core == core_now and has_cache
    same_route = (
        cached_core is not None
        and has_cache
        and cached_core["origin"] == core_now["origin"]
        and cached_core["destination"] == core_now["destination"]
    )

    try:
        if exact_repeat:
            raw_pool = state["last_search_raw"]
            from_cache = True
        else:
            fetched = search_flights_raw(params)
            # Same route (origin+destination) as the last search but a different date/pax/cabin
            # -> accumulate, so compare_flights_tool can genuinely rank across multiple real
            # searches on this route instead of only ever seeing the most recent one.
            raw_pool = _merge_unique_flights(state["last_search_raw"], fetched) if same_route else fetched
            updates["last_search_raw"] = raw_pool
            updates["last_search_params"] = core_now
            recent = list(state.get("recent_searches") or [])
            recent.insert(0, core_now)
            updates["recent_searches"] = recent[:10]
            from_cache = False

        flights = apply_client_filters(raw_pool, params)

        if not flights:
            result = {
                "status": "no_results",
                "message": (
                    f"No flights found from {params.origin} to {params.destination} on "
                    f"{params.date} matching the given filters."
                ),
                "flights": [],
            }
        else:
            result = {"status": "ok", "flights": flights, "count": len(flights)}
            updates["active_filters"] = {
                k: v for k, v in {
                    "max_stops": params.max_stops,
                    "time_of_day": params.time_of_day,
                    "direct_only": params.direct_only,
                }.items() if v is not None
            }
            updates["last_search_results"] = flights

        return result, from_cache
    except NoFlightsFoundError as exc:
        return {"status": "no_results", "message": str(exc), "flights": []}, from_cache
    except TravelpayoutsError as exc:
        return {"status": "error", "message": str(exc), "flights": []}, from_cache


def _handle_compare_flights(args: dict, state: AgentState, updates: dict) -> tuple[dict, bool]:
    try:
        params = CompareFlightsInput(**args)
    except ValidationError as exc:
        return {"status": "error", "message": f"Invalid comparison parameters: {exc.errors()}"}, False

    base = state.get("last_search_results") or []
    if not base:
        return (
            {"status": "error", "message": "No cached flight results to compare. Search for flights first."},
            False,
        )

    filtered = filter_flights_for_comparison(base, params.airlines, params.flight_numbers)

    if not filtered:
        return (
            {
                "status": "no_results",
                "message": "No cached flights match that airline/flight-number filter.",
            },
            True,
        )

    updates["last_search_results"] = filtered
    if params.airlines or params.flight_numbers:
        active = dict(state.get("active_filters") or {})
        if params.airlines:
            active["airlines"] = params.airlines
        if params.flight_numbers:
            active["flight_numbers"] = params.flight_numbers
        updates["active_filters"] = active

    if len(filtered) == 1:
        f = filtered[0]
        result = {
            "status": "ok",
            "table": [f],
            "recommendation": f"Only one matching flight: {f['airline']} {f['flight_number']}.",
            "best_flight": f,
        }
    else:
        result = {"status": "ok", **compare_flights(filtered, params.metric)}

    updates["last_comparison"] = result
    return result, True


def _handle_airport_search(args: dict) -> dict:
    try:
        params = AirportSearchInput(**args)
    except ValidationError as exc:
        return {"status": "error", "message": f"Invalid airport search parameters: {exc.errors()}"}

    try:
        locations = search_airports(params)
        return {"status": "ok", "locations": locations, "count": len(locations)}
    except NoLocationsFoundError as exc:
        return {"status": "no_results", "message": str(exc), "locations": []}
    except TravelpayoutsError as exc:
        return {"status": "error", "message": str(exc), "locations": []}


def _handle_currency_conversion(args: dict) -> dict:
    try:
        params = CurrencyConversionInput(**args)
    except ValidationError as exc:
        return {"status": "error", "message": f"Invalid currency conversion parameters: {exc.errors()}"}

    try:
        result = convert_currency(params.amount, params.from_currency, params.to_currency)
        return {"status": "ok", **result}
    except CurrencyConversionError as exc:
        return {"status": "error", "message": str(exc)}


def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_messages = []
    updates: dict = {}
    tool_logs = list(state.get("tool_logs") or [])

    for call in last.tool_calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        from_cache = False

        if name == "flight_search_tool":
            result, from_cache = _handle_flight_search(args, state, updates)
        elif name == "compare_flights_tool":
            result, from_cache = _handle_compare_flights(args, state, updates)
        elif name == "airport_search_tool":
            result = _handle_airport_search(args)
        elif name == "currency_conversion_tool":
            result = _handle_currency_conversion(args)
        else:
            result = {"status": "error", "message": f"Unknown tool '{name}'."}

        _log(tool_logs, name, args, result, from_cache)
        tool_messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call_id))

    updates["messages"] = tool_messages
    updates["tool_logs"] = tool_logs
    return updates
