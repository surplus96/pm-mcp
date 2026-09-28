"""KIS Developers (한국투자증권) REST API thin client.

Why this exists
---------------
PyKrx scrapes KRX endpoints that block cloud-egress IPs (HF Spaces,
Vercel, etc.), and yfinance/Yahoo doesn't index the KRX special
listings (REIT, ETN, ELW, A-prefix stock loan codes — anything with a
letter in the 6-char ticker like ``0001A0``). KIS Developers exposes
the official KRX feed over plain HTTPS with OAuth2 and no IP block,
which gives us a stable third-tier source so those tickers stop
returning empty data on the deployed instance.

Scope
-----
- OAuth2 client-credentials flow (POST /oauth2/tokenP)
- Token cached in the shared diskcache (23h TTL — KIS issues 24h tokens)
- Single ``request()`` helper that adds Bearer + appkey/appsecret + tr_id
- Production endpoint only (paper trading toggle intentionally skipped)

Higher-level OHLCV / quote helpers live in ``kis_market_data.py``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import requests

from mcp_server.tools.cache_manager import cache_manager

logger = logging.getLogger(__name__)


KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
_TOKEN_CACHE_KEY = "kis_access_token_v1"
_TOKEN_TTL_SEC = 23 * 60 * 60  # KIS issues 24h tokens; refresh 1h early
_TOKEN_LOCK = threading.Lock()


def _creds() -> tuple[str, str] | None:
    app_key = os.getenv("KIS_APP_KEY", "").strip()
    app_secret = os.getenv("KIS_APP_SECRET", "").strip()
    if not app_key or not app_secret:
        return None
    return app_key, app_secret


def is_configured() -> bool:
    """True iff both KIS_APP_KEY and KIS_APP_SECRET are set."""
    return _creds() is not None


def _fetch_token(app_key: str, app_secret: str) -> str | None:
    """Mint a fresh OAuth2 access token from KIS."""
    try:
        resp = requests.post(
            f"{KIS_BASE_URL}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": app_key,
                "appsecret": app_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            logger.warning("KIS token endpoint returned no access_token: %s", data)
            return None
        logger.info("KIS access token issued (expires_in=%s).", data.get("expires_in"))
        return token
    except Exception as e:  # noqa: BLE001
        logger.warning("KIS token request failed: %s", e)
        return None


def get_access_token() -> str | None:
    """Return a cached KIS access token, minting one if needed.

    Diskcache is the single source of truth so concurrent uvicorn
    workers share the same token (KIS rate-limits token requests
    aggressively — minting per worker burns daily quota).
    """
    creds = _creds()
    if creds is None:
        return None

    cached = cache_manager.get(_TOKEN_CACHE_KEY)
    if isinstance(cached, str) and cached:
        return cached

    with _TOKEN_LOCK:
        cached = cache_manager.get(_TOKEN_CACHE_KEY)
        if isinstance(cached, str) and cached:
            return cached
        token = _fetch_token(*creds)
        if token:
            cache_manager.set(_TOKEN_CACHE_KEY, token, ttl=_TOKEN_TTL_SEC)
        return token


def request(
    path: str,
    *,
    tr_id: str,
    params: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Issue an authenticated KIS request and return parsed JSON.

    Returns ``None`` when KIS is not configured, the token cannot be
    minted, or the upstream call fails. Callers must handle ``None``
    so the caller's own fallback chain still runs.
    """
    creds = _creds()
    if creds is None:
        return None
    app_key, app_secret = creds
    token = get_access_token()
    if token is None:
        return None
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",  # P = personal (vs B = business)
    }
    url = f"{KIS_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)
        # KIS uses 200 for both success and "domain failure" — must check rt_cd
        resp.raise_for_status()
        data = resp.json()
        rt_cd = str(data.get("rt_cd", "0"))
        if rt_cd != "0":
            # Common: invalid ticker, special listing not in this market, etc.
            # Log at debug — caller decides whether the empty result is fatal.
            logger.debug("KIS %s rt_cd=%s msg=%s", tr_id, rt_cd, data.get("msg1"))
        return data
    except requests.HTTPError as e:
        # 401 means the cached token is stale (clock skew, manual revoke).
        # Drop it so the next call re-mints.
        if e.response is not None and e.response.status_code == 401:
            logger.info("KIS token rejected — clearing cache so next call re-mints.")
            cache_manager.delete(_TOKEN_CACHE_KEY)
        else:
            logger.warning("KIS HTTP error tr_id=%s status=%s", tr_id,
                           getattr(e.response, "status_code", "?"))
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("KIS request failed tr_id=%s err=%s", tr_id, e)
        return None
