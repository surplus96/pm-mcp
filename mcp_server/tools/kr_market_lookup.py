"""KOSPI vs KOSDAQ classifier for 6-digit Korean stock codes.

Why a dedicated module?
    pykrx ``get_market_ticker_list`` and ``get_market_ohlcv_by_ticker``
    have been returning empty results / JSON parse errors against the
    KRX endpoint for several weeks (KRX-side change). We can't rely on
    a runtime "list all KOSPI codes, list all KOSDAQ codes" lookup, so
    instead we layer:

        1. Seed JSON (``mcp_server/data/kr_kosdaq_codes.json``) for the
           ~100 most-asked-about KOSDAQ tickers — covers users' first
           queries with zero network calls.
        2. yfinance probe — try ``CODE.KS`` first, then ``CODE.KQ``;
           cache the winner. Subsequent calls return from cache.
        3. Default to KOSPI (``.KS``) if both probes fail or the
           network is offline; the caller can still fall through to a
           PyKrx OHLCV path which works for both markets.

This keeps the module side-effect-free at import time.
"""
from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kr_kosdaq_codes.json"

# Probe cache: code -> "KS" | "KQ". Wrapped in a lock so concurrent
# requests don't fire duplicate yfinance lookups.
_probe_cache: dict[str, str] = {}
_probe_lock = threading.Lock()


@lru_cache(maxsize=1)
def _seed_kosdaq_codes() -> set[str]:
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(c) for c in raw.get("codes", []) if isinstance(c, str)}
    except FileNotFoundError:
        logger.warning("kr_kosdaq_codes.json missing — defaulting all unknown codes to KOSPI")
        return set()
    except Exception as e:  # noqa: BLE001
        logger.warning("kr_kosdaq_codes.json parse failed: %s", e)
        return set()


def _strip_suffix(code: str) -> str:
    t = (code or "").strip().upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        t = t[:-3]
    return t


def _probe_yfinance(code: str) -> str | None:
    """Try ``code.KS`` first (KOSPI is the larger market), then ``code.KQ``.

    Returns the winning suffix or None if both fail. The probe is a
    cheap ``info`` lookup; cached so the next call is free.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    for suffix in ("KS", "KQ"):
        try:
            ticker = f"{code}.{suffix}"
            info = yf.Ticker(ticker).fast_info
            # ``last_price`` is None when the symbol is unrecognised.
            price = getattr(info, "last_price", None)
            if price is not None and float(price) > 0:
                return suffix
        except Exception:  # noqa: BLE001
            continue
    return None


def market_suffix(code: str) -> str:
    """Return ``"KS"`` or ``"KQ"`` for a 6-digit Korean stock code.

    Defaults to ``"KS"`` (KOSPI) when the code is unknown and the
    yfinance probe fails — that matches pre-fix behaviour and lets the
    PyKrx OHLCV path (which doesn't need a suffix) keep working.
    """
    code = _strip_suffix(code)
    if not (code.isdigit() and len(code) == 6):
        return "KS"
    if code in _seed_kosdaq_codes():
        return "KQ"
    cached = _probe_cache.get(code)
    if cached:
        return cached
    with _probe_lock:
        cached = _probe_cache.get(code)
        if cached:
            return cached
        suffix = _probe_yfinance(code) or "KS"
        _probe_cache[code] = suffix
        return suffix


def is_kosdaq(code: str) -> bool:
    """Convenience: ``True`` if the code resolves to KOSDAQ (``.KQ``)."""
    return market_suffix(code) == "KQ"


def kr_yfinance_symbol(code: str) -> str:
    """Build the full ``CODE.KS`` / ``CODE.KQ`` symbol for yfinance."""
    code = _strip_suffix(code)
    if not (code.isdigit() and len(code) == 6):
        return code
    return f"{code}.{market_suffix(code)}"
