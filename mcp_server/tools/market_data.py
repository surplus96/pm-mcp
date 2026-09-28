from __future__ import annotations
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import logging
from mcp_server.config import PROCESSED_PATH, safe_filename
from mcp_server.tools.cache_manager import cached, TTL
from mcp_server.tools.resilience import (
    retry_with_backoff, RetryConfig,
    circuit_yfinance, CircuitOpenError
)
from mcp_server.tools.yf_utils import detect_market, normalize_yf_columns

logger = logging.getLogger(__name__)


def _yf_symbol(ticker: str) -> str:
    """Map a Korean 6-digit code (or .KS/.KQ) to its Yahoo-suffixed form.

    Yahoo Finance returns 404 for bare KR codes like ``005930`` — they
    must carry ``.KS`` (KOSPI) or ``.KQ`` (KOSDAQ). Without this shim
    every KR call burnt a yfinance round-trip + ``possibly delisted``
    warning before falling back to PyKrx/DART. US tickers pass through
    unchanged.
    """
    if detect_market(ticker) == "KR":
        from mcp_server.tools.kr_market_lookup import kr_yfinance_symbol
        return kr_yfinance_symbol(ticker)
    return ticker


@retry_with_backoff(
    attempts=RetryConfig.YFINANCE["attempts"],
    min_wait=RetryConfig.YFINANCE["min_wait"],
    max_wait=RetryConfig.YFINANCE["max_wait"]
)
def _download_prices(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """yfinance 가격 다운로드 (재시도 + 서킷 브레이커)"""
    def _do_download():
        return yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df = circuit_yfinance.call(_do_download)
    return normalize_yf_columns(df)


_PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


@cached(ttl=TTL.DAILY, prefix="prices")
def get_prices(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    market: Optional[str] = None,
    period: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily OHLCV prices (cached TTL.DAILY).

    Dispatches to the KR path (``KoreanMarketAdapter`` via PyKrx) for
    Korean 6-digit codes / ``.KS`` / ``.KQ`` suffixes (or ``market="KR"``),
    otherwise falls back to yfinance. The cache key includes the raw ticker
    so both paths coexist safely — no collision.

    ``period`` (e.g. ``"6mo"``, ``"1y"``) is only used when ``start`` is not
    given; the default window is one year.
    """
    now = datetime.now()
    if not start:
        days = _PERIOD_DAYS.get(period or "1y", 365)
        start = (now - timedelta(days=days)).strftime('%Y-%m-%d')
    end = end or now.strftime('%Y-%m-%d')
    is_kr = (market or "").upper() == "KR" or detect_market(ticker) == "KR"

    # FR-K02: Korean tickers route through PyKrx first; if PyKrx returns
    # empty (KRX bot-blocks cloud-egress IPs like HF Spaces) we walk to
    # KIS Developers, which uses OAuth2 over plain HTTPS and indexes
    # every KRX security including special listings (REIT/ETN/A-prefix).
    # yfinance is the final fallback with .KS/.KQ suffix applied below.
    if is_kr:
        try:
            from mcp_server.tools.kr_market_data import get_kr_adapter
            df = get_kr_adapter().get_ohlcv(ticker, start=start, end=end)
            if df is not None and not df.empty:
                if df.index.name or df.index.dtype.kind == "M":
                    df = df.reset_index()
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning("PyKrx fetch failed for %s, trying KIS: %s", ticker, e)

        # KIS fall-through — handles common stock when PyKrx is blocked
        # AND special listings PyKrx/yfinance don't index at all.
        try:
            from mcp_server.tools import kis_market_data as kis
            df = kis.get_ohlcv(ticker, start=start, end=end)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            logger.warning("KIS fetch failed for %s, falling back to yfinance: %s", ticker, e)

    try:
        # KR ticker fall-through (PyKrx empty / failed): hit Yahoo with
        # the ``.KS`` / ``.KQ`` suffixed form so we don't 404 on bare
        # 6-digit codes.
        data = _download_prices(_yf_symbol(ticker), start, end, interval)
        return data.reset_index()
    except CircuitOpenError:
        logger.warning(f"yfinance circuit open for {ticker}, returning empty DataFrame")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to download prices for {ticker}: {e}")
        return pd.DataFrame()


def _safe_get(info: Dict[str, Any], key: str, default=None):
    try:
        v = info.get(key)
        if v is None:
            return default
        if isinstance(v, (int, float)):
                return float(v)
        return v
    except Exception:
        return default


@cached(ttl=TTL.FUNDAMENTAL, prefix="fundamentals")
def get_fundamentals_snapshot(ticker: str) -> dict:
    """펀더멘털 스냅샷 조회 (24시간 캐시, 서킷 브레이커 적용).

    yfinance lookup uses ``kr_yfinance_symbol`` for Korean codes so
    ``005930`` becomes ``005930.KS`` (or ``247540.KQ`` for KOSDAQ).
    Without the suffix yfinance returns 404 and noisy ``possibly
    delisted`` warnings on every chat / report turn even when the DART
    path already populated the financial factors successfully.
    """
    is_kr = detect_market(ticker) == "KR"
    if is_kr:
        from mcp_server.tools.kr_market_lookup import kr_yfinance_symbol
        yf_symbol = kr_yfinance_symbol(ticker)
    else:
        yf_symbol = ticker

    # KR seed via KIS — gives the Stock Analyzer page real numbers for
    # KRX special listings (REIT/ETN/A-prefix stock-loan/ELW codes like
    # ``0001A0``) that yfinance refuses to index. KIS quote returns
    # last price / market cap / shares / PER / PBR / EPS / BPS in KRW;
    # yfinance fields below overwrite anything KIS can't supply (sector
    # / industry / cashflow), so the existing US path is unchanged.
    kis_seed: dict = {}
    if is_kr:
        try:
            from mcp_server.tools import kis_market_data as _kis
            q = _kis.get_quote(ticker)
            if q:
                kis_seed = {
                    "last_price": q.get("last_price"),
                    "market_cap": q.get("market_cap"),
                    "shares": q.get("shares_outstanding"),
                    "currency": q.get("currency") or "KRW",
                    "pe": q.get("per"),
                    "pb": q.get("pbr"),
                    "eps": q.get("eps"),
                    "bps": q.get("bps"),
                }
        except Exception as e:  # noqa: BLE001
            logger.debug("KIS quote seed failed for %s: %s", ticker, e)

    # If yfinance can't help (KR special listing) skip the network round
    # trip and just return whatever KIS gave us. The 5+ HTTP 404s would
    # otherwise burn ~1s and re-introduce the log noise we just cleaned.
    from mcp_server.tools.yf_utils import is_yfinance_supported
    if not is_yfinance_supported(ticker):
        return {"ticker": ticker, **{k: v for k, v in kis_seed.items() if v is not None}}

    try:
        def _fetch_info():
            tk = yf.Ticker(yf_symbol)
            info = {}
            try:
                info = tk.info if isinstance(getattr(tk, 'info', None), dict) else {}
            except Exception:
                info = {}

            fast = getattr(tk, 'fast_info', None)
            def _fast_get(name: str):
                try:
                    return getattr(fast, name)
                except Exception:
                    return None
            out = {
                "ticker": ticker,
                "market_cap": _fast_get('market_cap') if fast else None,
                "shares": _fast_get('shares') if fast else None,
                "currency": _fast_get('currency') if fast else None,
                "last_price": _fast_get('last_price') if fast else None,
                "sector": _safe_get(info, 'sector'),
                "industry": _safe_get(info, 'industry'),
                "pe": _safe_get(info, 'trailingPE'),
                "pb": _safe_get(info, 'priceToBook'),
                "eps": _safe_get(info, 'trailingEps'),
                "forwardEps": _safe_get(info, 'forwardEps'),
                "revenueGrowth": _safe_get(info, 'revenueGrowth'),
                "earningsQuarterlyGrowth": _safe_get(info, 'earningsQuarterlyGrowth'),
                "profitMargins": _safe_get(info, 'profitMargins'),
                "returnOnEquity": _safe_get(info, 'returnOnEquity'),
                "returnOnAssets": _safe_get(info, 'returnOnAssets'),
                "roic": _safe_get(info, 'returnOnCapitalEmployed'),
            }
            try:
                cf = getattr(tk, 'cashflow', None)
                if cf is not None and not cf.empty:
                    for label in ("Free Cash Flow", "FreeCashFlow"):
                        if label in cf.index:
                            fcf_series = cf.loc[label]
                            out["freeCashFlow"] = float(fcf_series.iloc[0]) if len(fcf_series) else None
                            break
            except Exception:
                pass
            return out

        result = circuit_yfinance.call(_fetch_info)
        # Layer KIS seed under yfinance — yfinance fields win when both
        # have a value (more granular fundamentals), KIS fills in the
        # ones yfinance left as None for KR tickers.
        for k, v in kis_seed.items():
            if v is not None and result.get(k) in (None, "", 0, 0.0):
                result[k] = v
        return result
    except CircuitOpenError:
        logger.warning(f"yfinance circuit open for fundamentals: {ticker}")
        return {"ticker": ticker, "error": "circuit_open",
                **{k: v for k, v in kis_seed.items() if v is not None}}
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e),
                **{k: v for k, v in kis_seed.items() if v is not None}}


@cached(ttl=TTL.DAILY, prefix="momentum")
def get_momentum_metrics(ticker: str) -> dict:
    """안정적 모멘텀 계산 (4시간 캐시, 서킷 브레이커 적용): yfinance download 실패 시 Ticker().history로 폴백."""
    yf_sym = _yf_symbol(ticker)
    hist = None
    try:
        def _download():
            return yf.download(yf_sym, period="1y", interval="1d", progress=False, auto_adjust=True)
        hist = normalize_yf_columns(circuit_yfinance.call(_download))
    except CircuitOpenError:
        logger.warning(f"yfinance circuit open for momentum: {ticker}")
        hist = None
    except Exception:
        hist = None

    if hist is None or hist.empty:
        try:
            def _history():
                tk = yf.Ticker(yf_sym)
                return tk.history(period="1y", interval="1d", auto_adjust=True)
            hist = normalize_yf_columns(circuit_yfinance.call(_history))
        except CircuitOpenError:
            logger.warning(f"yfinance circuit open for momentum fallback: {ticker}")
            hist = None
        except Exception:
            hist = None

    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"mom1": None, "mom3": None, "mom6": None, "mom12": None}
    close = hist["Close"].reset_index(drop=True)
    def ret(n):
        try:
            if len(close) < n:
                return None
            return float((close.iloc[-1] / close.iloc[-n]) - 1.0)
        except Exception:
            return None
    return {"mom1": ret(21), "mom3": ret(63), "mom6": ret(126), "mom12": ret(252)}


# -------- Token-saving helpers --------


@cached(ttl=TTL.DAILY, prefix="prices_summary")
def get_prices_summary(ticker: str, period: str = "1y", interval: str = "1d", agg: str = "W") -> dict:
    """가격 요약 조회 (4시간 캐시)"""
    hist = normalize_yf_columns(
        yf.download(_yf_symbol(ticker), period=period, interval=interval, progress=False, auto_adjust=True)
    )
    if hist.empty:
        return {"ticker": ticker, "count": 0}
    if agg:
        ohlc = hist.resample(agg).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    else:
        ohlc = hist
    desc = hist["Close"].describe().to_dict()
    return {"ticker": ticker, "count": int(len(hist)), "agg": agg, "ohlc_rows": int(len(ohlc)), "close_stats": {k: float(v) for k,v in desc.items()}}


def write_prices_csv(ticker: str, start: Optional[str], end: Optional[str], interval: str = "1d") -> str:
    df = get_prices(ticker, start=start, end=end, interval=interval)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    out_path = os.path.join(PROCESSED_PATH, f"prices_{safe_filename(ticker)}_{date_str}.csv")
    df.to_csv(out_path, index=False)
    return out_path
