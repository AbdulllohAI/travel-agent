"""Currency Conversion Tool (bonus): converts prices to a target currency using live exchange rates.

Uses Frankfurter (frankfurter.dev, no API key, ECB rates) by default. If EXCHANGE_RATE_API_KEY is
set, uses exchangerate-api.com instead, which additionally covers currencies Frankfurter lacks
(e.g. UZS).
"""
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.config import EXCHANGE_RATE_API_KEY, FRANKFURTER_BASE_URL


class CurrencyConversionError(Exception):
    """Raised when the exchange rate API is unreachable, errors out, or lacks the requested currency."""


class CurrencyConversionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., gt=0, description="The amount to convert")
    from_currency: str = Field(..., description="Source currency code, e.g. 'USD'")
    to_currency: str = Field(..., description="Target currency code, e.g. 'UZS'")


def _fetch_rate_frankfurter(base: str, target: str, timeout: float = 10.0) -> float:
    try:
        resp = requests.get(
            f"{FRANKFURTER_BASE_URL}/latest",
            params={"base": base, "symbols": target},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CurrencyConversionError(f"Could not reach the exchange rate API: {exc}") from exc

    if resp.status_code != 200:
        raise CurrencyConversionError(f"Exchange rate API error ({resp.status_code}): {resp.text[:200]}")

    rates = resp.json().get("rates", {})
    if target not in rates:
        raise CurrencyConversionError(
            f"Currency '{target}' isn't supported by Frankfurter (it only covers major/ECB currencies, "
            "not UZS). Set EXCHANGE_RATE_API_KEY in .env to use exchangerate-api.com instead."
        )
    return rates[target]


def _fetch_rate_exchangerate_api(base: str, target: str, timeout: float = 10.0) -> float:
    try:
        resp = requests.get(
            f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base}/{target}",
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CurrencyConversionError(f"Could not reach the exchange rate API: {exc}") from exc

    if resp.status_code != 200:
        raise CurrencyConversionError(f"Exchange rate API error ({resp.status_code}): {resp.text[:200]}")

    payload = resp.json()
    if payload.get("result") != "success":
        raise CurrencyConversionError(
            f"Exchange rate API returned an error: {payload.get('error-type', 'unknown')}"
        )
    return payload["conversion_rate"]


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Pure function: fetches a live rate and returns the converted amount."""
    from_currency, to_currency = from_currency.upper(), to_currency.upper()

    if from_currency == to_currency:
        return {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": 1.0,
            "converted_amount": round(amount, 2),
        }

    rate = (
        _fetch_rate_exchangerate_api(from_currency, to_currency)
        if EXCHANGE_RATE_API_KEY
        else _fetch_rate_frankfurter(from_currency, to_currency)
    )

    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": rate,
        "converted_amount": round(amount * rate, 2),
    }


@tool("currency_conversion_tool", args_schema=CurrencyConversionInput)
def currency_conversion_tool(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert a price from one currency to another using a live exchange rate API.
    Never invents exchange rates."""
    try:
        result = convert_currency(amount, from_currency, to_currency)
        return {"status": "ok", **result}
    except CurrencyConversionError as exc:
        return {"status": "error", "message": str(exc)}
