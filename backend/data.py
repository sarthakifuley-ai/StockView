"""
Market data access layer.

The rest of the app calls these functions and never talks to a provider
directly. That lets us choose the best source per data type:

  - Quotes & company names: Finnhub (official API) when FINNHUB_API_KEY is
    set, otherwise yfinance.
  - Historical candles: yfinance (not on Finnhub's free plan), cached with a
    stale-on-rate-limit fallback.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
from yfinance.exceptions import YFRateLimitError  # type: ignore

from cache import TTLCache
from config import (
    BEST_EFFORT_DEADLINE_SECONDS,
    COMPANY_NAME_CACHE_TTL_SECONDS,
    DATA_CACHE_TTL_SECONDS,
    FAILURE_CACHE_TTL_SECONDS,
    MIN_HISTORY_ROWS,
    NAME_FAILURE_TTL_SECONDS,
    QUOTE_CACHE_TTL_SECONDS,
    YF_DEADLINE_SECONDS,
    YF_MAX_PENDING,
    YF_MAX_WORKERS,
    YF_NETWORK_TIMEOUT_SECONDS,
)
from errors import (
    ProviderError,
    SymbolNotFoundError,
    UpstreamBusyError,
    UpstreamRateLimitError,
    UpstreamTimeoutError,
)
from providers import finnhub

__all__ = [
    "UpstreamRateLimitError",
    "download_stock_data",
    "get_company_name",
    "get_quote",
]

_data_cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_SECONDS, max_size=100, name="data", max_stale_seconds=86400)
_name_cache = TTLCache(ttl_seconds=COMPANY_NAME_CACHE_TTL_SECONDS, max_size=500, name="company", max_stale_seconds=86400 * 2)
_quote_cache = TTLCache(ttl_seconds=QUOTE_CACHE_TTL_SECONDS, max_size=200, name="quote", max_stale_seconds=600)


def _normalize(symbol: str) -> str:
    return symbol.upper().strip()


# ---------------------------------------------------------------------------
# Bounded, coalesced upstream calls
#
# Why this exists: a yfinance call that never returns used to pin a request
# thread forever. Python cannot kill a running thread, so a deadline alone does
# NOT cancel the underlying network call. The design therefore has three parts:
#   1. yfinance is given its own network timeout, so the HTTP call really ends.
#   2. Calls run on a small fixed pool and a request thread only waits up to a
#      deadline, so a slow call can no longer hang a request.
#   3. A semaphore caps calls that are running *or still finishing after their
#      caller gave up*. When it is full we fail fast (UpstreamBusyError)
#      instead of queueing more, so orphaned work cannot pile up.
# Identical concurrent calls share one in-flight Future (request coalescing),
# including a call whose first caller already timed out.
# ---------------------------------------------------------------------------

_yf_executor = ThreadPoolExecutor(max_workers=YF_MAX_WORKERS, thread_name_prefix="yf")
_yf_slots = threading.BoundedSemaphore(YF_MAX_PENDING)
_inflight: dict[str, Future] = {}
_inflight_lock = threading.Lock()


def _release_slot(key: str, fut: Future) -> None:
    with _inflight_lock:
        if _inflight.get(key) is fut:
            del _inflight[key]
    _yf_slots.release()


def _run_bounded(key: str, fn, deadline: float):
    """
    Run fn() on the yfinance pool, sharing any identical in-flight call.

    Raises UpstreamTimeoutError if no result within `deadline` seconds (the
    call itself may keep running and will still populate caches when done),
    or UpstreamBusyError if the pool is saturated.
    """
    with _inflight_lock:
        fut = _inflight.get(key)
        if fut is None:
            if not _yf_slots.acquire(blocking=False):
                raise UpstreamBusyError(
                    "Market data requests are backed up. Please try again shortly."
                )
            try:
                fut = _yf_executor.submit(fn)
            except Exception:
                _yf_slots.release()
                raise
            _inflight[key] = fut
            owner = True
        else:
            owner = False
    if owner:
        # Registered after releasing the lock: the callback takes it itself.
        fut.add_done_callback(lambda f, k=key: _release_slot(k, f))

    try:
        return fut.result(timeout=deadline)
    except FutureTimeout:
        raise UpstreamTimeoutError(
            "Market data provider did not respond in time. Please try again."
        ) from None


# Brief memory of recent upstream failures: (expires_at, exception class, message).
_failures: dict[str, tuple[float, type, str]] = {}
_failures_lock = threading.Lock()


def _remember_failure(key: str, exc: Exception, ttl: float = FAILURE_CACHE_TTL_SECONDS) -> None:
    # "Busy" is our own capacity limit, not evidence the provider is unhealthy;
    # remembering it would keep rejecting a key long after capacity frees up.
    if ttl <= 0 or isinstance(exc, UpstreamBusyError):
        return
    with _failures_lock:
        if len(_failures) > 500:  # keep the map bounded
            now = time.time()
            for k in [k for k, v in _failures.items() if v[0] <= now]:
                del _failures[k]
        _failures[key] = (time.time() + ttl, type(exc), str(exc))


def _recent_failure(key: str) -> Exception | None:
    with _failures_lock:
        entry = _failures.get(key)
        if entry is None:
            return None
        if entry[0] <= time.time():
            del _failures[key]
            return None
        return entry[1](entry[2])


def _clear_failure(key: str) -> None:
    with _failures_lock:
        _failures.pop(key, None)


def _is_upstream_failure(exc: Exception) -> bool:
    return isinstance(exc, (ProviderError, UpstreamRateLimitError, YFRateLimitError))


def _normalize_upstream_error(exc: Exception) -> Exception:
    if isinstance(exc, YFRateLimitError):
        return UpstreamRateLimitError(
            "Yahoo Finance is rate limiting requests. Please try again in a minute."
        )
    return exc


class _Coalescer:
    """
    Share one in-progress computation between concurrent callers.

    The first caller (leader) runs the producer inline; followers wait for its
    result up to `deadline`. Used for work that is already bounded internally
    (Finnhub HTTP calls with their own timeout).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Future] = {}

    def run(self, key: str, producer, deadline: float):
        with self._lock:
            fut = self._pending.get(key)
            leader = fut is None
            if leader:
                fut = Future()
                self._pending[key] = fut
        if not leader:
            try:
                return fut.result(timeout=deadline)
            except FutureTimeout:
                raise UpstreamTimeoutError(
                    "Market data provider did not respond in time. Please try again."
                ) from None
        try:
            result = producer()
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        else:
            fut.set_result(result)
            return result
        finally:
            with self._lock:
                self._pending.pop(key, None)


_quote_flight = _Coalescer()


# ---------------------------------------------------------------------------
# Historical candles (yfinance)
# ---------------------------------------------------------------------------

def _tag(df: pd.DataFrame, *, stale: bool, fetched_at: float | None) -> pd.DataFrame:
    """Attach provenance to a frame (df.attrs survives .copy() and pickling)."""
    out = df.copy()
    out.attrs = {**df.attrs, "stale": stale}
    if fetched_at is not None:
        out.attrs["fetched_at"] = fetched_at
    return out


def _fetch_history(symbol: str, period: str, interval: str, min_rows: int, cache_key: str, use_cache: bool):
    """Runs on a worker thread. Validates and caches, so a late result is not wasted."""
    data = yf.Ticker(symbol).history(
        period=period, interval=interval, timeout=YF_NETWORK_TIMEOUT_SECONDS
    )
    if data.empty:
        raise SymbolNotFoundError(f"No data found for symbol '{symbol}'.")
    if len(data) < min_rows:
        raise ValueError(
            f"Insufficient data for '{symbol}': need at least {min_rows} rows, "
            f"got {len(data)}."
        )
    data.attrs = {"stale": False, "fetched_at": time.time()}
    if use_cache:
        _data_cache.set(cache_key, data.copy())
    return data


def download_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    min_rows: int = MIN_HISTORY_ROWS,
    deadline: float | None = None,
) -> pd.DataFrame:
    """
    Download historical OHLCV data using yfinance.

    Order: fresh cache -> (recent failure? stale cache or error) -> bounded,
    coalesced yfinance call -> on failure, stale cache.

    The returned frame carries provenance in `df.attrs`:
    `stale` (True when an expired cached copy is served because the provider
    failed) and `fetched_at` (epoch seconds when the data was downloaded).

    Raises:
        ValueError / SymbolNotFoundError: no data or too few rows.
        UpstreamRateLimitError / UpstreamTimeoutError / UpstreamBusyError /
        ProviderError: upstream failed and no cached copy exists.
    """
    symbol = _normalize(symbol)
    cache_key = f"{symbol}:{period}:{interval}"
    fail_key = f"hist:{cache_key}"
    if use_cache:
        cached = _data_cache.get(cache_key)
        if cached is not None:
            return _tag(cached, stale=False, fetched_at=cached.attrs.get("fetched_at"))

    def serve_stale_or(exc: Exception):
        stale = _data_cache.get_stale(cache_key) if use_cache else None
        if stale is not None:
            print(f"[data] {type(exc).__name__} for {cache_key}; serving stale cache.")
            return _tag(stale, stale=True, fetched_at=stale.attrs.get("fetched_at"))
        raise exc

    recent = _recent_failure(fail_key)
    if recent is not None:
        return serve_stale_or(recent)

    effective_deadline = YF_DEADLINE_SECONDS if deadline is None else deadline
    try:
        data = _run_bounded(
            f"hist:{cache_key}:{min_rows}:{int(use_cache)}",
            lambda: _fetch_history(symbol, period, interval, min_rows, cache_key, use_cache),
            effective_deadline,
        )
    except Exception as exc:
        if not _is_upstream_failure(exc):
            raise  # not-found / insufficient rows: genuine answer, don't mask it
        exc = _normalize_upstream_error(exc)
        # A caller that chose a short deadline for an optional lookup timing
        # out says nothing about provider health; don't poison the key for others.
        if not (isinstance(exc, UpstreamTimeoutError) and deadline is not None):
            _remember_failure(fail_key, exc)
        return serve_stale_or(exc)

    _clear_failure(fail_key)
    return _tag(data, stale=False, fetched_at=data.attrs.get("fetched_at"))


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

def _quote_from_yfinance(symbol: str) -> dict:
    data = download_stock_data(symbol, period="5d", interval="1d", min_rows=1)
    latest = data.iloc[-1]
    price = float(latest["Close"])
    prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else None
    return {
        "price": price,
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "previous_close": prev_close,
        "change": price - prev_close if prev_close else None,
        "change_percent": (price - prev_close) / prev_close * 100 if prev_close else None,
        "volume": int(latest["Volume"]),
        # Provenance of the candles: they may be an expired cached copy.
        "stale": bool(data.attrs.get("stale", False)),
        "fetched_at": data.attrs.get("fetched_at"),
    }


def _volume_from_yfinance(symbol: str) -> int | None:
    """
    Best-effort volume for Finnhub quotes (its free quote has none).

    Uses a short deadline and never raises. Daily candles are cached, so this
    is at most one Yahoo call per symbol per cache TTL, and if it is slow the
    call keeps running in the background and fills the cache for next time.
    """
    try:
        data = download_stock_data(
            symbol, period="5d", interval="1d", min_rows=1,
            deadline=BEST_EFFORT_DEADLINE_SECONDS,
        )
        return int(data["Volume"].iloc[-1])
    except Exception:
        return None


def _fetch_quote(symbol: str) -> dict:
    if finnhub.is_configured():
        try:
            quote = finnhub.get_quote(symbol)
            quote["volume"] = _volume_from_yfinance(symbol)
            quote["source"] = "finnhub"
            quote["as_of"] = quote.get("timestamp")
            return quote
        except (ProviderError, UpstreamRateLimitError) as exc:
            print(f"[data] Finnhub quote failed for {symbol} ({exc}); using yfinance.")
    quote = {**_quote_from_yfinance(symbol), "source": "yfinance"}
    quote["as_of"] = None
    return quote


def get_quote(symbol: str) -> dict:
    """
    Return the latest quote for a symbol, tagged with its `source`.

    Order: fresh cache -> Finnhub -> yfinance -> stale cache.
    Every quote carries `fetched_at` (epoch seconds) and `stale`. `stale` is
    True only when an expired cached quote is served because providers failed.
    """
    symbol = _normalize(symbol)
    cached = _quote_cache.get(symbol)
    if cached is not None:
        return {**cached, "stale": False}

    fail_key = f"quote:{symbol}"

    def serve_stale_or(exc: Exception):
        stale = _quote_cache.get_stale(symbol)
        if stale is not None:
            print(f"[data] {type(exc).__name__} for quote {symbol}; serving stale quote.")
            return {**stale, "stale": True}
        raise exc

    recent = _recent_failure(fail_key)
    if recent is not None:
        return serve_stale_or(recent)

    try:
        quote = _quote_flight.run(
            symbol, lambda: _fetch_quote(symbol), YF_DEADLINE_SECONDS + 10
        )
    except Exception as exc:
        if not _is_upstream_failure(exc):
            raise
        exc = _normalize_upstream_error(exc)
        _remember_failure(fail_key, exc)
        return serve_stale_or(exc)

    quote = {**quote, "fetched_at": quote.get("fetched_at") or time.time()}
    if quote.get("stale"):
        # Built from an expired cached copy: report it as such and do not
        # cache it as fresh, which would extend its life.
        return quote
    _quote_cache.set(symbol, quote)
    _clear_failure(fail_key)
    return {**quote, "stale": False}


# ---------------------------------------------------------------------------
# Company name
# ---------------------------------------------------------------------------

def _lookup_company_name(symbol: str) -> str | None:
    """Finnhub profile if configured, else yfinance .info. Runs bounded."""
    if finnhub.is_configured():
        # Finnhub is the name source when configured; never fall through to
        # Yahoo's heavily rate-limited .info endpoint just for a display name.
        return finnhub.get_company_profile(symbol).get("name")
    return yf.Ticker(symbol).info.get("longName")


def _name_worker(symbol: str) -> str | None:
    name = _lookup_company_name(symbol)
    if name:
        _name_cache.set(symbol, name)
    return name


def get_company_name(symbol: str) -> str:
    """
    Return the company's long name, cached for a long time.

    Best-effort: waits only a few seconds and never raises. On a miss it
    returns the ticker and remembers the failure briefly instead of retrying
    on every request. A slow lookup keeps running and fills the cache when it
    completes.
    """
    symbol = _normalize(symbol)
    cached = _name_cache.get_stale(symbol)
    if cached is not None:
        return cached

    fail_key = f"name:{symbol}"
    if _recent_failure(fail_key) is not None:
        return symbol

    try:
        name = _run_bounded(
            fail_key, lambda: _name_worker(symbol), BEST_EFFORT_DEADLINE_SECONDS
        )
    except UpstreamTimeoutError:
        # Still running in the background; retry soon rather than in 5 minutes.
        _remember_failure(fail_key, UpstreamTimeoutError("name timeout"), ttl=FAILURE_CACHE_TTL_SECONDS)
        return symbol
    except UpstreamBusyError:
        return symbol  # local capacity limit; try again on the next request
    except Exception as exc:  # rate limit, network, parsing...
        print(f"[data] Could not fetch company name for {symbol}: {exc}")
        _remember_failure(fail_key, ProviderError(str(exc)), ttl=NAME_FAILURE_TTL_SECONDS)
        return symbol

    return name or symbol
