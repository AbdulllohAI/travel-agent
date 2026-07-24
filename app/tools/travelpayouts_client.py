"""Shared Travelpayouts/Aviasales API client.

Much simpler than the Amadeus client it replaced: no OAuth2 handshake. The autocomplete
(airport/city) endpoint needs no credentials at all; the price-data endpoint needs a single
static API token (no signature, no marker) passed as a header — issued instantly on signup at
https://www.travelpayouts.com, no partner approval required.
"""
import logging
from typing import Any, Optional

import requests

from app.config import TRAVELPAYOUTS_API_TOKEN

logger = logging.getLogger(__name__)

PRICES_BASE_URL = "https://api.travelpayouts.com"
AUTOCOMPLETE_BASE_URL = "https://autocomplete.travelpayouts.com"
DATA_BASE_URL = "https://api.travelpayouts.com/data"


class TravelpayoutsError(Exception):
    """Base error for Travelpayouts API failures."""


class TravelpayoutsAuthError(TravelpayoutsError):
    """API token missing or rejected."""


class TravelpayoutsRateLimitError(TravelpayoutsError):
    """Travelpayouts API rate limit (HTTP 429) was hit."""


class TravelpayoutsRequestError(TravelpayoutsError):
    """Travelpayouts API returned an error response, or the network call failed."""


class TravelpayoutsClient:
    """Thin GET wrapper for Travelpayouts' price-data and autocomplete endpoints."""

    def __init__(self, api_token: str = TRAVELPAYOUTS_API_TOKEN, timeout: float = 15.0):
        self.api_token = api_token
        self.timeout = timeout

    def get_prices(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Calls a token-authenticated price-data endpoint (e.g. /v1/prices/cheap)."""
        if not self.api_token:
            raise TravelpayoutsAuthError(
                "TRAVELPAYOUTS_API_TOKEN is not set. Get a free token at "
                "https://www.travelpayouts.com (Profile -> API token) and add it to your .env file."
            )
        try:
            resp = requests.get(
                f"{PRICES_BASE_URL}{path}",
                params=params or {},
                headers={"X-Access-Token": self.api_token},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TravelpayoutsRequestError(f"Could not reach the Travelpayouts API: {exc}") from exc

        if resp.status_code == 429:
            raise TravelpayoutsRateLimitError("Travelpayouts API rate limit hit. Please wait and try again.")
        if resp.status_code >= 400:
            raise TravelpayoutsRequestError(
                f"Travelpayouts API error ({resp.status_code}) on {path}: {resp.text[:300]}"
            )

        payload = resp.json()
        if isinstance(payload, dict) and payload.get("success") is False:
            raise TravelpayoutsRequestError(f"Travelpayouts API returned an error: {payload.get('error')}")
        return payload

    def get_reference_json(self, path: str) -> Any:
        """Calls an unauthenticated static reference-data endpoint (e.g. /en/airlines.json),
        used to translate IATA codes (airline, airport) into real display names — never
        guessed by the LLM."""
        try:
            resp = requests.get(f"{DATA_BASE_URL}{path}", timeout=self.timeout)
        except requests.RequestException as exc:
            raise TravelpayoutsRequestError(f"Could not reach Travelpayouts reference data: {exc}") from exc

        if resp.status_code >= 400:
            raise TravelpayoutsRequestError(
                f"Travelpayouts reference data error ({resp.status_code}) on {path}: {resp.text[:300]}"
            )
        return resp.json()

    def get_autocomplete(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Calls the unauthenticated city/airport autocomplete endpoint. Needs no token."""
        try:
            resp = requests.get(f"{AUTOCOMPLETE_BASE_URL}/places2", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TravelpayoutsRequestError(f"Could not reach the Travelpayouts autocomplete API: {exc}") from exc

        if resp.status_code == 429:
            raise TravelpayoutsRateLimitError("Travelpayouts API rate limit hit. Please wait and try again.")
        if resp.status_code >= 400:
            raise TravelpayoutsRequestError(
                f"Travelpayouts autocomplete API error ({resp.status_code}): {resp.text[:300]}"
            )
        return resp.json()


_shared_client: Optional[TravelpayoutsClient] = None


def get_travelpayouts_client() -> TravelpayoutsClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = TravelpayoutsClient()
    return _shared_client


def reset_travelpayouts_client() -> None:
    """Clears the cached client. Used by tests to force re-instantiation with mocked settings."""
    global _shared_client
    _shared_client = None
