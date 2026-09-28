"""KIS-backed KR market data adapter (OHLCV + current quote).

Sits between PyKrx and yfinance in the KR fall-through chain. Used
when the primary path returns empty — typically for cloud-egress
hosts where KRX blocks PyKrx, or for special listings (REIT/ETN/
A-prefix) that yfinance doesn't index.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from mcp_server.tools import kis_client
from mcp_server.tools.cache_manager import TTL, cached

logger = logging.getLogger(__name__)


# KIS market-division codes for FID_COND_MRKT_DIV_CODE.
# ``J`` covers regular common stock (KOSPI/KOSDAQ/KONEX); the others
# pick up the niche security types yfinance/Yahoo don't index. We try
# them in order until one returns data so a single resolver handles
# every flavour of 6-char KRX code.
_MARKET_DIV_CHAIN: tuple[str, ...] = ("J", "ETF", "ETN", "W")


def _normalize_date(value: str | None) -> str:
    """Coerce ``YYYY-MM-DD`` or ``YYYYMMDD`` (or None) to ``YYYYMMDD``."""
    if not value:
        return ""
    raw = str(value).replace("-", "").strip()
    return raw[:8] if len(raw) >= 8 else ""


def _clean_ticker(ticker: str) -> str:
    """Strip Yahoo suffixes — KIS expects the bare 6-char KRX code."""
    return str(ticker).strip().upper().replace(".KS", "").replace(".KQ", "")


@cached(ttl=TTL.DAILY, prefix="kis_ohlcv")
def get_ohlcv(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Daily OHLCV for any KRX security via KIS.

    Walks the market-division chain (J → ETF → ETN → W) so the same
    entry point handles common stock and special listings without the
    caller having to classify the ticker first.

    Returns an empty frame when KIS is unconfigured / unauthenticated /
    has no data, so it composes cleanly with the existing fallback
    chain in ``market_data.get_prices``.
    """
    if not kis_client.is_configured():
        return pd.DataFrame()

    code = _clean_ticker(ticker)
    if not code:
        return pd.DataFrame()

    today = datetime.now()
    end_date = _normalize_date(end) or today.strftime("%Y%m%d")
    # Default 1y look-back — KIS hard-caps a single call to ~100 daily
    # candles, but that already covers the typical 1y x 252-trading-day
    # span used by our momentum / chart tools.
    start_default = today - timedelta(days=365)
    start_date = _normalize_date(start) or start_default.strftime("%Y%m%d")

    base_params = {
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_date,
        "FID_INPUT_DATE_2": end_date,
        "FID_PERIOD_DIV_CODE": "D",  # D=daily, W=weekly, M=monthly
        "FID_ORG_ADJ_PRC": "1",       # 1 = adjusted for corporate actions
    }

    for market_div in _MARKET_DIV_CHAIN:
        params = {**base_params, "FID_COND_MRKT_DIV_CODE": market_div}
        data = kis_client.request(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id="FHKST03010100",
            params=params,
        )
        if not data:
            continue
        rows = data.get("output2") or []
        if not rows:
            continue

        # KIS returns oldest-first when dates are explicit; canonicalise.
        records = []
        for r in rows:
            date = r.get("stck_bsop_date")
            if not date:
                continue
            try:
                records.append({
                    "Date": pd.to_datetime(date, format="%Y%m%d"),
                    "Open": float(r.get("stck_oprc") or 0),
                    "High": float(r.get("stck_hgpr") or 0),
                    "Low": float(r.get("stck_lwpr") or 0),
                    "Close": float(r.get("stck_clpr") or 0),
                    "Volume": float(r.get("acml_vol") or 0),
                })
            except (TypeError, ValueError):
                continue
        if not records:
            continue
        df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
        logger.info("KIS OHLCV: %d rows for %s (market_div=%s).", len(df), code, market_div)
        return df

    logger.debug("KIS OHLCV: no data for %s across market_div=%s.", code, _MARKET_DIV_CHAIN)
    return pd.DataFrame()


@cached(ttl=TTL.FUNDAMENTAL, prefix="kis_quote")
def get_quote(ticker: str) -> dict:
    """Current-day quote snapshot (price, market cap, day range, etc.).

    Lighter-weight than ``get_ohlcv`` — used by the fundamentals path
    when we just need a last-traded price + outstanding-share count for
    a special-listing ticker.
    """
    if not kis_client.is_configured():
        return {}
    code = _clean_ticker(ticker)
    if not code:
        return {}

    for market_div in _MARKET_DIV_CHAIN:
        data = kis_client.request(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div,
                "FID_INPUT_ISCD": code,
            },
        )
        if not data:
            continue
        out = data.get("output") or {}
        if not out:
            continue
        try:
            return {
                "ticker": code,
                "last_price": float(out.get("stck_prpr") or 0) or None,
                "open": float(out.get("stck_oprc") or 0) or None,
                "high": float(out.get("stck_hgpr") or 0) or None,
                "low": float(out.get("stck_lwpr") or 0) or None,
                "volume": float(out.get("acml_vol") or 0) or None,
                "market_cap": float(out.get("hts_avls") or 0) * 1_000_000 or None,  # KIS reports in 백만원
                "shares_outstanding": float(out.get("lstn_stcn") or 0) or None,
                "per": float(out.get("per") or 0) or None,
                "pbr": float(out.get("pbr") or 0) or None,
                "eps": float(out.get("eps") or 0) or None,
                "bps": float(out.get("bps") or 0) or None,
                "currency": "KRW",
                "market_div": market_div,
            }
        except (TypeError, ValueError) as e:
            logger.debug("KIS quote parse failed for %s: %s", code, e)
            continue
    return {}
