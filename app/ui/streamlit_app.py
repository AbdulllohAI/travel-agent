"""Streamlit UI for the AI Travel Agent.

Sidebar: search-parameter form, recent searches, and settings (model/currency/API-key status).
Main page: chat, flight results, flight comparison, and a tool-logs panel that shows exactly
which tool was called, with what parameters, and the raw result — proof that flight data comes
from real tool calls, not from the LLM's imagination.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app import config  # noqa: E402
from app.agent.graph import get_graph  # noqa: E402
from app.memory import new_thread_id  # noqa: E402
from app.tools.currency import CurrencyConversionError, convert_currency  # noqa: E402

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

CABIN_CLASSES = ["", "economy", "premium_economy", "business", "first"]
FALLBACK_MODELS = [config.OLLAMA_MODEL, *config.OLLAMA_FALLBACK_MODELS]


def _init_session_state() -> None:
    st.session_state.setdefault("thread_id", new_thread_id())
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("target_currency", config.DEFAULT_CURRENCY)
    st.session_state.setdefault("llm_provider", config.LLM_PROVIDER)


def _thread_config() -> dict:
    # llm_provider/api_key ride along per-session here (not global config
    # mutation, unlike OLLAMA_MODEL below) so agent_node's LangGraph
    # RunnableConfig picks them up per-invocation -- see nodes.py's
    # _build_llm docstring for why a public deployment needs that.
    return {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "llm_provider": st.session_state["llm_provider"],
            "api_key": st.session_state.get("gemini_api_key"),
        }
    }


def _current_state() -> dict:
    graph = get_graph()
    snapshot = graph.get_state(_thread_config())
    return snapshot.values or {}


def _list_ollama_models() -> list[str]:
    try:
        import ollama

        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        names = [m["model"] if isinstance(m, dict) else m.model for m in client.list()["models"]]
        return names or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


def _run_turn(user_text: str) -> None:
    st.session_state["chat_history"].append({"role": "user", "content": user_text})
    graph = get_graph()
    try:
        result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=_thread_config())
        final = result["messages"][-1]
        reply = final.content if isinstance(final, AIMessage) else str(final.content)
    except Exception as exc:  # Ollama down, missing/invalid Gemini key, model missing, etc. — surface, don't crash
        hint = (
            "Check that Ollama is running (`ollama serve`) and the selected model is pulled."
            if st.session_state["llm_provider"] == "ollama"
            else "Check that the Gemini API key pasted in the sidebar is valid."
        )
        reply = (
            "⚠️ The agent hit an unexpected error and couldn't finish this request.\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n{hint}"
        )
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("🔍 Search Parameters")
        with st.form("search_form"):
            origin = st.text_input("Origin (IATA code or city)", placeholder="TAS")
            destination = st.text_input("Destination (IATA code or city)", placeholder="IST")
            travel_date = st.date_input("Departure date", value=date.today() + timedelta(days=30))
            col1, col2 = st.columns(2)
            adults = col1.number_input("Adults", min_value=1, max_value=9, value=1)
            children = col2.number_input("Children", min_value=0, max_value=8, value=0)
            cabin_class = st.selectbox("Cabin class (optional)", CABIN_CLASSES)
            direct_only = st.checkbox("Direct flights only")
            submitted = st.form_submit_button("Search flights", use_container_width=True)

        if submitted:
            if not origin or not destination:
                st.sidebar.error("Origin and destination are required.")
            else:
                parts = [
                    f"Search flights from {origin.strip()} to {destination.strip()} "
                    f"on {travel_date.isoformat()} for {adults} adult(s)"
                ]
                if children:
                    parts.append(f" and {children} child(ren)")
                if cabin_class:
                    parts.append(f", {cabin_class} class")
                if direct_only:
                    parts.append(", direct flights only")
                parts.append(".")
                with st.spinner("Asking the agent..."):
                    _run_turn("".join(parts))
                st.rerun()

        st.divider()
        st.header("🕘 Recent Searches")
        recent = _current_state().get("recent_searches", [])
        if not recent:
            st.caption("No searches yet.")
        else:
            for i, s in enumerate(recent[:8]):
                label = f"{s['origin']} → {s['destination']} ({s['date']})"
                if st.button(label, key=f"recent_{i}", use_container_width=True):
                    extra = f", {s['adults']} adult(s)"
                    if s.get("children"):
                        extra += f", {s['children']} child(ren)"
                    with st.spinner("Asking the agent..."):
                        _run_turn(
                            f"Search flights from {s['origin']} to {s['destination']} on {s['date']}{extra}."
                        )
                    st.rerun()

        st.divider()
        st.header("⚙️ Settings")

        provider_labels = {"Local (Ollama)": "ollama", "Gemini (cloud)": "gemini"}
        default_label = next(
            (label for label, value in provider_labels.items()
             if value == st.session_state["llm_provider"]),
            "Local (Ollama)",
        )
        chosen_label = st.selectbox(
            "LLM Provider", list(provider_labels.keys()),
            index=list(provider_labels.keys()).index(default_label),
        )
        st.session_state["llm_provider"] = provider_labels[chosen_label]

        if st.session_state["llm_provider"] == "gemini":
            st.text_input(
                "Gemini API key",
                type="password",
                placeholder="Paste your Gemini API key",
                key="gemini_api_key",
                help="Get a free key at aistudio.google.com/apikey. Used only "
                     "for your own requests this session -- never logged, "
                     "displayed, or shared with other visitors.",
            )
            st.caption(
                "🔑 [Get a free Gemini API key](https://aistudio.google.com/apikey) "
                "— no key is stored server-side beyond this session."
            )
            if not st.session_state.get("gemini_api_key"):
                st.info("Paste a Gemini API key above to chat.")
        else:
            available_models = _list_ollama_models()
            current_model = config.OLLAMA_MODEL if config.OLLAMA_MODEL in available_models else available_models[0]
            selected_model = st.selectbox("Ollama model", available_models, index=available_models.index(current_model))
            config.OLLAMA_MODEL = selected_model

        st.session_state["target_currency"] = st.text_input(
            "Target currency for conversions", value=st.session_state["target_currency"]
        ).upper()

        st.caption("API key status")
        travelpayouts_ok = bool(config.TRAVELPAYOUTS_API_TOKEN)
        st.write(
            ("🟢" if travelpayouts_ok else "🔴")
            + " Travelpayouts token "
            + ("set" if travelpayouts_ok else "missing")
        )
        if st.session_state["llm_provider"] == "gemini":
            gemini_ok = bool(st.session_state.get("gemini_api_key"))
            st.write(("🟢" if gemini_ok else "🔴") + " Gemini key " + ("set" if gemini_ok else "missing"))
        else:
            try:
                import requests

                requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
                st.write("🟢 Ollama reachable")
            except Exception:
                st.write("🔴 Ollama unreachable")


def _render_chat_tab() -> None:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Ask about flights (e.g. \"Toshkentdan Istanbulga 15-avgust kuni reys top\")")
    if user_text:
        with st.spinner("Asking the agent..."):
            _run_turn(user_text)
        st.rerun()


def _render_results_tab() -> None:
    state = _current_state()
    flights = state.get("last_search_results", [])
    if not flights:
        st.info("No flight results yet — search for a route in the chat or the sidebar form.")
        return

    target = st.session_state["target_currency"]
    rows = []
    for f in flights:
        price_display = f"{f['price']} {f['currency']}"
        if target and target != f["currency"]:
            try:
                converted = convert_currency(f["price"], f["currency"], target)
                price_display += f" (~{converted['converted_amount']} {target})"
            except CurrencyConversionError:
                pass
        rows.append(
            {
                "Airline": f["airline"],
                "Flight": f["flight_number"],
                "From": f["departure_airport"],
                "To": f["arrival_airport"],
                "Departs": f["departure_time"],
                "Arrives": f["arrival_time"],
                "Duration": f["duration"],
                "Stops": f["stops"],
                "Cabin": f["cabin_class"],
                "Price": price_display,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_comparison_tab() -> None:
    comparison = _current_state().get("last_comparison")
    if not comparison:
        st.info('No comparison yet — try "compare X and Y" or "which is cheapest?" in the chat.')
        return

    st.dataframe(comparison.get("table", []), use_container_width=True, hide_index=True)
    if comparison.get("recommendation"):
        st.success(f"🏆 {comparison['recommendation']}")


def _render_tool_logs_tab() -> None:
    logs = _current_state().get("tool_logs", [])
    if not logs:
        st.info("No tool calls yet.")
        return

    for i, entry in enumerate(reversed(logs)):
        badge = "cached (no API call)" if entry["served_from_cache"] else "live API call"
        with st.expander(f"{len(logs) - i}. {entry['tool']} — {badge}"):
            st.write("**Parameters:**")
            st.json(entry["params"])
            st.write("**Result:**")
            st.code(entry["result_summary"], language="json")


def main() -> None:
    _init_session_state()
    st.title("✈️ AI Travel Agent")
    st.caption("Local LLM + real tool calling — flight data always comes from a live Travelpayouts API call.")

    _render_sidebar()

    tab_chat, tab_results, tab_comparison, tab_logs = st.tabs(
        ["💬 Chat", "✈️ Flight Results", "📊 Flight Comparison", "🛠️ Tool Logs"]
    )
    with tab_chat:
        _render_chat_tab()
    with tab_results:
        _render_results_tab()
    with tab_comparison:
        _render_comparison_tab()
    with tab_logs:
        _render_tool_logs_tab()


if __name__ == "__main__":
    main()
