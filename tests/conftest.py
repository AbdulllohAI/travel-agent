import pytest
import responses as responses_lib

from app.tools.flight_search import reset_airline_names_cache
from app.tools.travelpayouts_client import TravelpayoutsClient


@pytest.fixture
def responses():
    """This installed version of `responses` doesn't register its pytest plugin automatically,
    so provide the same `responses` fixture API (rsps.add(...)) ourselves."""
    with responses_lib.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def travelpayouts_test_client() -> TravelpayoutsClient:
    return TravelpayoutsClient(api_token="test_token")


@pytest.fixture(autouse=True)
def _reset_airline_names_cache():
    """The airline code->name lookup is cached process-wide; reset it each test so one test's
    (mocked or unmocked) lookup result can't leak into another."""
    reset_airline_names_cache()
    yield
