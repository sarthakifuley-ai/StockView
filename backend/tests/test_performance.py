"""
Tests for the loading-performance work: bounded upstream calls, request
coalescing, stale-cache fallback with provenance, brief failure caching and
best-effort volume / company-name lookups.

yfinance is replaced with a controllable fake; nothing touches the network.
"""

import threading
import time
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import data
from errors import (
    ProviderError,
    SymbolNotFoundError,
    UpstreamBusyError,
    UpstreamRateLimitError,
    UpstreamTimeoutError,
)
from main import app
from providers import finnhub

client = TestClient(app)


def _frame(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": [10.0 + i for i in range(n)], "Volume": 1000},
        index=idx,
    )


class FakeYF:
    """Stands in for yf.Ticker. `gate` blocks history() until set."""

    def __init__(self):
        self.calls = 0
        self.gate = threading.Event()
        self.gate.set()  # not blocking by default
        self.error: Exception | None = None
        self.frame = _frame()
        self.info_gate = threading.Event()
        self.info_gate.set()
        self.info_calls = 0
        self._lock = threading.Lock()

    def Ticker(self, symbol):  # noqa: N802 - mimics yf.Ticker
        outer = self

        class _T:
            @property
            def info(self_inner):
                with outer._lock:
                    outer.info_calls += 1
                outer.info_gate.wait(10)
                return {"longName": "Fake Corp"}

            def history(self_inner, **kwargs):
                with outer._lock:
                    outer.calls += 1
                outer.gate.wait(10)
                if outer.error:
                    raise outer.error
                return outer.frame.copy()

        return _T()


@pytest.fixture
def fake(monkeypatch):
    f = FakeYF()
    monkeypatch.setattr(data.yf, "Ticker", f.Ticker)
    data._data_cache.clear()
    data._quote_cache.clear()
    data._name_cache.clear()
    data._failures.clear()
    monkeypatch.setattr(data, "YF_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(data, "BEST_EFFORT_DEADLINE_SECONDS", 0.2)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    yield f
    f.gate.set()
    f.info_gate.set()
    time.sleep(0.05)
    data._data_cache.clear()
    data._quote_cache.clear()
    data._name_cache.clear()
    data._failures.clear()


# --- bounded calls ----------------------------------------------------------

def test_hung_upstream_returns_within_deadline(fake):
    fake.gate.clear()
    start = time.time()
    with pytest.raises(UpstreamTimeoutError):
        data.download_stock_data("AAPL", period="1y", interval="1d")
    assert time.time() - start < 2.0


def test_timed_out_call_still_fills_cache_when_it_finishes(fake):
    fake.gate.clear()
    with pytest.raises(UpstreamTimeoutError):
        data.download_stock_data("AAPL", period="1y", interval="1d")
    fake.gate.set()
    deadline = time.time() + 2
    while time.time() < deadline and data._data_cache.get("AAPL:1y:1d") is None:
        time.sleep(0.02)
    assert data._data_cache.get("AAPL:1y:1d") is not None


def test_timed_out_calls_cannot_accumulate(fake, monkeypatch):
    monkeypatch.setattr(data, "_yf_slots", threading.BoundedSemaphore(2))
    fake.gate.clear()
    for sym in ("AAA", "BBB"):
        with pytest.raises(UpstreamTimeoutError):
            data.download_stock_data(sym, period="1y", interval="1d")
    # Both timed-out calls are still running and still hold their slots.
    with pytest.raises(UpstreamBusyError):
        data.download_stock_data("CCC", period="1y", interval="1d")
    # Once the stuck calls finish, capacity returns.
    fake.gate.set()
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            data.download_stock_data("DDD", period="1y", interval="1d")
            break
        except UpstreamBusyError:
            time.sleep(0.02)
    else:
        pytest.fail("slots were never released")


# --- coalescing -------------------------------------------------------------

def test_concurrent_identical_requests_share_one_upstream_call(fake, monkeypatch):
    monkeypatch.setattr(data, "YF_DEADLINE_SECONDS", 3)
    fake.gate.clear()
    results, errors = [], []

    def worker():
        try:
            results.append(data.download_stock_data("AAPL", period="1y", interval="1d"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    fake.gate.set()
    for t in threads:
        t.join(5)

    assert not errors
    assert len(results) == 6
    assert fake.calls == 1


def test_concurrent_quotes_share_one_fetch(fake, monkeypatch):
    monkeypatch.setattr(data, "YF_DEADLINE_SECONDS", 3)
    fake.gate.clear()
    out = []
    threads = [threading.Thread(target=lambda: out.append(data.get_quote("AAPL"))) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    fake.gate.set()
    for t in threads:
        t.join(5)
    assert len(out) == 4
    assert fake.calls == 1


# --- stale fallback and provenance -----------------------------------------

def test_fresh_data_is_marked_not_stale_with_timestamp(fake):
    df = data.download_stock_data("AAPL", period="1y", interval="1d")
    assert df.attrs["stale"] is False
    assert df.attrs["fetched_at"] == pytest.approx(time.time(), abs=5)


def test_stale_cache_served_and_flagged_when_upstream_fails(fake):
    seeded = _frame()
    seeded.attrs = {"stale": False, "fetched_at": 1_700_000_000}
    data._data_cache.set("AAPL:1y:1d", seeded)
    fake.error = UpstreamRateLimitError("limited")
    with patch.object(data._data_cache, "get", return_value=None):  # simulate expiry
        df = data.download_stock_data("AAPL", period="1y", interval="1d")
    assert df.attrs["stale"] is True
    assert df.attrs["fetched_at"] == 1_700_000_000  # original time, not "now"


def test_upstream_failure_without_cache_raises(fake):
    fake.error = ProviderError("down")
    with pytest.raises(ProviderError):
        data.download_stock_data("AAPL", period="1y", interval="1d")


def test_network_exception_becomes_provider_error_and_uses_stale_cache(fake):
    fake.error = ConnectionError("Could not resolve host")
    with pytest.raises(ProviderError) as info:
        data.download_stock_data("AAPL", period="1y", interval="1d")
    assert not isinstance(info.value, ValueError)  # not reported as "unknown symbol"
    data._failures.clear()

    seeded = _frame()
    seeded.attrs = {"stale": False, "fetched_at": 1_700_000_000}
    data._data_cache.set("AAPL:1y:1d", seeded)
    with patch.object(data._data_cache, "get", return_value=None):  # simulate expiry
        df = data.download_stock_data("AAPL", period="1y", interval="1d")
    assert df.attrs["stale"] is True


def test_empty_frame_after_network_failure_is_not_reported_as_unknown_symbol(fake, monkeypatch):
    fake.frame = pd.DataFrame()
    monkeypatch.setattr(data.yf_shared, "_ERRORS", {"AAPL": "Failed to perform, curl: (6) Could not resolve host"})
    with pytest.raises(ProviderError) as info:
        data.download_stock_data("AAPL", period="1y", interval="1d")
    assert not isinstance(info.value, ValueError)


def test_symbol_not_found_is_not_masked_by_stale_cache(fake):
    fake.frame = pd.DataFrame()
    with pytest.raises(SymbolNotFoundError):
        data.download_stock_data("NOPE", period="1y", interval="1d")


# --- brief failure caching --------------------------------------------------

def test_failure_is_remembered_briefly(fake):
    fake.error = ProviderError("down")
    with pytest.raises(ProviderError):
        data.download_stock_data("AAPL", period="1y", interval="1d")
    calls_after_first = fake.calls
    with pytest.raises(ProviderError):
        data.download_stock_data("AAPL", period="1y", interval="1d")
    assert fake.calls == calls_after_first  # second attempt never reached Yahoo


def test_failure_memory_expires(fake):
    fake.error = ProviderError("down")
    with pytest.raises(ProviderError):
        data.download_stock_data("AAPL", period="1y", interval="1d")
    # Age the remembered failure past its TTL, then let the provider recover.
    key = "hist:AAPL:1y:1d"
    expires, kind, msg = data._failures[key]
    data._failures[key] = (time.time() - 1, kind, msg)
    fake.error = None
    assert data.download_stock_data("AAPL", period="1y", interval="1d").attrs["stale"] is False


# --- best-effort volume and company name -----------------------------------

def test_finnhub_quote_not_blocked_by_slow_volume_lookup(fake, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    fh = {"price": 10.0, "open": 9, "high": 11, "low": 8, "previous_close": 9,
          "change": 1, "change_percent": 11.1, "timestamp": 1_700_000_123}
    fake.gate.clear()
    start = time.time()
    with patch.object(finnhub, "get_quote", return_value=fh):
        quote = data.get_quote("AAPL")
    assert time.time() - start < 1.0  # best-effort deadline, not the full 15s
    assert quote["volume"] is None
    assert quote["source"] == "finnhub"
    assert quote["stale"] is False
    assert quote["as_of"] == 1_700_000_123
    assert quote["fetched_at"] == pytest.approx(time.time(), abs=5)


def test_slow_volume_lookup_does_not_poison_history_for_others(fake, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    fh = {"price": 10.0, "open": 9, "high": 11, "low": 8, "previous_close": 9,
          "change": 1, "change_percent": 11.1, "timestamp": 1}
    fake.gate.clear()
    with patch.object(finnhub, "get_quote", return_value=fh):
        data.get_quote("AAPL")
    assert data._recent_failure("hist:AAPL:5d:1d") is None


def test_company_name_lookup_is_best_effort_and_not_retried(fake):
    fake.info_gate.clear()
    start = time.time()
    assert data.get_company_name("AAPL") == "AAPL"  # falls back to ticker
    assert time.time() - start < 1.0
    first_calls = fake.info_calls
    assert data.get_company_name("AAPL") == "AAPL"
    assert fake.info_calls == first_calls  # not retried immediately


def test_company_name_uses_finnhub_and_never_yahoo_info(fake, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    with patch.object(finnhub, "get_company_profile", return_value={"name": "Apple Inc"}):
        assert data.get_company_name("AAPL") == "Apple Inc"
    assert data.get_company_name("AAPL") == "Apple Inc"  # cached
    assert fake.info_calls == 0


def test_company_name_finnhub_miss_does_not_fall_back_to_yahoo(fake, monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    with patch.object(finnhub, "get_company_profile", return_value={}):
        assert data.get_company_name("AAPL") == "AAPL"
    assert fake.info_calls == 0


# --- quotes: stale handling -------------------------------------------------

def test_stale_quote_is_flagged_and_keeps_original_timestamp(fake):
    data._quote_cache.set("AAPL", {"price": 5.0, "open": 5, "high": 5, "low": 5, "source": "finnhub",
                                   "fetched_at": 1_700_000_000})
    fake.error = ProviderError("down")
    with patch.object(data._quote_cache, "get", return_value=None):
        quote = data.get_quote("AAPL")
    assert quote["stale"] is True
    assert quote["fetched_at"] == 1_700_000_000
    assert quote["price"] == 5.0


def test_quote_built_from_stale_candles_is_flagged_and_not_recached(fake):
    seeded = _frame()
    seeded.attrs = {"stale": False, "fetched_at": 1_700_000_000}
    data._data_cache.set("AAPL:5d:1d", seeded)
    fake.error = ProviderError("down")
    with patch.object(data._data_cache, "get", return_value=None):
        quote = data.get_quote("AAPL")
    assert quote["stale"] is True
    assert quote["fetched_at"] == 1_700_000_000
    assert data._quote_cache.get("AAPL") is None  # staleness not laundered into "fresh"


def test_quote_failure_without_cache_raises(fake):
    fake.error = ProviderError("down")
    with pytest.raises(ProviderError):
        data.get_quote("AAPL")


# --- HTTP layer -------------------------------------------------------------

def test_price_endpoint_reports_freshness_fields():
    q = {"price": 1.0, "open": 1, "high": 1, "low": 1, "source": "yfinance",
         "fetched_at": 1_700_000_000, "stale": True, "as_of": None}
    with patch("main.get_quote", return_value=q), patch("main.get_company_name", return_value="X"):
        body = client.get("/price?symbol=AAPL").json()
    assert body["stale"] is True
    assert body["fetched_at"] == 1_700_000_000


def test_history_endpoint_sets_freshness_headers():
    df = _frame(30)
    df.attrs = {"stale": True, "fetched_at": 1_700_000_000}
    with patch("main.download_stock_data", return_value=df):
        r = client.get("/history?symbol=AAPL&range=1y&interval=1d")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert r.headers["x-data-stale"] == "true"
    assert r.headers["x-data-fetched-at"] == "1700000000"


@pytest.mark.parametrize(
    "exc, status",
    [(UpstreamTimeoutError("slow"), 504), (UpstreamBusyError("busy"), 503), (ProviderError("x"), 502)],
)
def test_upstream_errors_map_to_http_status(exc, status):
    with patch("main.download_stock_data", side_effect=exc):
        assert client.get("/history?symbol=AAPL").status_code == status
