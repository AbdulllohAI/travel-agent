import pytest

from app.tools.currency import CurrencyConversionError, convert_currency, currency_conversion_tool

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"


def test_convert_currency_same_currency_short_circuits():
    result = convert_currency(100.0, "USD", "USD")
    assert result["rate"] == 1.0
    assert result["converted_amount"] == 100.0


def test_convert_currency_uses_frankfurter_by_default(responses):
    responses.add(
        responses.GET,
        FRANKFURTER_URL,
        json={"amount": 1.0, "base": "USD", "date": "2026-07-20", "rates": {"EUR": 0.92}},
        status=200,
    )

    result = convert_currency(100.0, "USD", "EUR")

    assert result["rate"] == 0.92
    assert result["converted_amount"] == 92.0


def test_convert_currency_frankfurter_missing_currency_raises(responses):
    responses.add(
        responses.GET,
        FRANKFURTER_URL,
        json={"amount": 1.0, "base": "USD", "date": "2026-07-20", "rates": {}},
        status=200,
    )

    with pytest.raises(CurrencyConversionError):
        convert_currency(100.0, "USD", "UZS")


def test_convert_currency_api_error_raises(responses):
    responses.add(responses.GET, FRANKFURTER_URL, json={"message": "error"}, status=500)

    with pytest.raises(CurrencyConversionError):
        convert_currency(100.0, "USD", "EUR")


def test_convert_currency_uses_exchangerate_api_when_key_set(responses, mocker):
    mocker.patch("app.tools.currency.EXCHANGE_RATE_API_KEY", "fake-key")
    responses.add(
        responses.GET,
        "https://v6.exchangerate-api.com/v6/fake-key/pair/USD/UZS",
        json={"result": "success", "conversion_rate": 12750.5},
        status=200,
    )

    result = convert_currency(100.0, "USD", "UZS")

    assert result["rate"] == 12750.5
    assert result["converted_amount"] == 1275050.0


def test_currency_conversion_tool_returns_ok_status(responses):
    responses.add(
        responses.GET,
        FRANKFURTER_URL,
        json={"amount": 1.0, "base": "USD", "date": "2026-07-20", "rates": {"EUR": 0.92}},
        status=200,
    )

    result = currency_conversion_tool.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "EUR"})

    assert result["status"] == "ok"
    assert result["converted_amount"] == 92.0


def test_currency_conversion_tool_returns_error_status_on_failure(responses):
    responses.add(responses.GET, FRANKFURTER_URL, json={"message": "error"}, status=500)

    result = currency_conversion_tool.invoke({"amount": 100.0, "from_currency": "USD", "to_currency": "EUR"})

    assert result["status"] == "error"
    assert "message" in result
