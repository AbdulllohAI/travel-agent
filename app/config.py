"""Central configuration: model name, API keys, and constants, all sourced from env vars."""
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ── Ollama (Local LLM) ──────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_FALLBACK_MODELS = ["qwen2.5:3b-instruct", "llama3.2:3b"]
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# ── Gemini (cloud, optional) ─────────────────────────
# For a public deployment with no Ollama server available: visitors bring
# their own key via the sidebar (see ui/streamlit_app.py) rather than one
# being baked into this config. GEMINI_API_KEY here is only a local/.env
# fallback for running the "gemini" provider without the UI's key prompt.
# Uses Gemini's OpenAI-compatible endpoint (via langchain-openai's
# ChatOpenAI) so tool-calling works through the same .bind_tools() call
# agent_node already uses for Ollama -- no extra provider-specific
# LangChain package needed.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ── Travelpayouts / Aviasales Flight Data API ───────
# Free signup at https://www.travelpayouts.com, token appears immediately in your profile
# under "API token" — no approval process, unlike Amadeus/Kiwi.
TRAVELPAYOUTS_API_TOKEN = os.getenv("TRAVELPAYOUTS_API_TOKEN", "")
TRAVELPAYOUTS_MARKER = os.getenv("TRAVELPAYOUTS_MARKER", "")

# ── Currency Conversion API (bonus) ─────────────────
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY", "")
FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"

# ── App Settings ─────────────────────────────────────
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "UZS")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# ── Search defaults / limits ────────────────────────
MAX_FLIGHT_RESULTS = int(os.getenv("MAX_FLIGHT_RESULTS", "10"))
