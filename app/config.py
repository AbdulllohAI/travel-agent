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
