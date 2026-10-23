"""Application configuration loaded from environment variables."""

import os
from pathlib import Path


# Load backend/.env for local development (Render/Vercel use dashboard env vars)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(_env_path)
    except ImportError:
        pass

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "").strip().strip('"').strip("'")
NEWS_URL = os.environ.get("NEWS_URL", "https://newsapi.org/v2/everything").strip()


def get_news_api_key() -> str:
    """Read NEWS_API_KEY at call time so Render env updates apply after redeploy."""
    return os.environ.get("NEWS_API_KEY", "").strip().strip('"').strip("'")


def get_finnhub_api_key() -> str:
    """Finnhub key (quotes, company profile, company news). Free at finnhub.io."""
    return os.environ.get("FINNHUB_API_KEY", "").strip().strip('"').strip("'")


def get_redis_url() -> str:
    """Redis URL (e.g. rediss://default:xxx@...upstash.io:6379)."""
    url = os.environ.get("REDIS_URL", "").strip().strip('"').strip("'")
    # Upstash requires TLS/SSL (rediss://). If redis:// was copied, auto-upgrade it.
    if url.startswith("redis://") and "upstash.io" in url:
        url = "rediss://" + url[len("redis://"):]
    return url

QUOTE_CACHE_TTL_SECONDS = int(os.environ.get("QUOTE_CACHE_TTL_SECONDS", "30"))
NEWS_CACHE_TTL_SECONDS = int(os.environ.get("NEWS_CACHE_TTL_SECONDS", "600"))

# Comma-separated origins, e.g. "https://stock-view-ebon.vercel.app,http://localhost:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

TRAIN_RATIO = 0.8
CNN_SEQUENCE_LENGTH = 10
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 16
MIN_HISTORY_ROWS = 50
MIN_FEATURE_ROWS = 30

MODEL_CACHE_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "3600"))
DATA_CACHE_TTL_SECONDS = int(os.environ.get("DATA_CACHE_TTL_SECONDS", "300"))
COMPANY_NAME_CACHE_TTL_SECONDS = int(os.environ.get("COMPANY_NAME_CACHE_TTL_SECONDS", "86400"))

# --- Upstream request limits (performance / hang protection) ---------------
# Network timeout handed to yfinance so the underlying HTTP call really stops.
YF_NETWORK_TIMEOUT_SECONDS = float(os.environ.get("YF_NETWORK_TIMEOUT_SECONDS", "10"))
# Wall-clock limit a request thread will wait for a yfinance call to return.
YF_DEADLINE_SECONDS = float(os.environ.get("YF_DEADLINE_SECONDS", "15"))
# Short deadline for optional lookups (volume, company name) on the /price path.
BEST_EFFORT_DEADLINE_SECONDS = float(os.environ.get("BEST_EFFORT_DEADLINE_SECONDS", "3"))
# Worker threads for yfinance, and the max number of yfinance calls that may be
# running or queued at once (timed-out calls still count until they finish).
YF_MAX_WORKERS = int(os.environ.get("YF_MAX_WORKERS", "4"))
YF_MAX_PENDING = int(os.environ.get("YF_MAX_PENDING", "8"))
# How long a failed upstream call is remembered so we don't hammer a sick provider.
FAILURE_CACHE_TTL_SECONDS = int(os.environ.get("FAILURE_CACHE_TTL_SECONDS", "30"))
NAME_FAILURE_TTL_SECONDS = int(os.environ.get("NAME_FAILURE_TTL_SECONDS", "300"))
FINNHUB_TIMEOUT_SECONDS = float(os.environ.get("FINNHUB_TIMEOUT_SECONDS", "5"))

VALID_ALGORITHMS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "cnn",
]
