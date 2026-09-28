from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import yfinance as yf

from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.yf_utils import normalize_yf_columns


def _pct(series: pd.Series, periods: int) -> Optional[float]:
    try:
        return float(series.pct_change(periods).iloc[-1])
    except Exception:
        return None


def _stdev(series: pd.Series, window: int) -> Optional[float]:
    try:
        return float(series.pct_change().rolling(window).std().iloc[-1])
    except Exception:
        return None


def _max_drawdown(series: pd.Series, lookback: int) -> Optional[float]:
    try:
        s = series.tail(lookback)
        roll_max = s.cummax()
        dd = (s - roll_max) / roll_max
        return float(dd.min())
    except Exception:
        return None


def _corr(a: pd.Series, b: pd.Series, window: int) -> Optional[float]:
    try:
        g = pd.concat([a.pct_change(), b.pct_change()], axis=1).dropna()
        if g.empty:
            return None
        return float(g.tail(window).corr().iloc[0, 1])
    except Exception:
        return None


def compute_basic_metrics(ticker: str, period: str = "2y", interval: str = "1d", use_cache: bool = True) -> Dict:
    """가격 기반 핵심 메트릭 산출: 모멘텀, 변동성, 최대낙폭, SPY 상관.
    diskcache 기반 TTL 캐싱 적용 (4시간).
    """
    cache_key = f"metrics:{ticker}:{period}:{interval}"

    # 캐시 확인 (TTL 자동 관리)
    if use_cache:
        cached_data = cache_manager.get(cache_key)
        if cached_data is not None:
            return cached_data

    try:
        hist = normalize_yf_columns(
            yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        )
        if hist.empty or "Close" not in hist.columns:
            raise RuntimeError("no price data")
        close = hist["Close"].dropna()

        # 모멘텀(일수 기준 대략치): 1M~12M
        mom1 = _pct(close, 21)
        mom3 = _pct(close, 63)
        mom6 = _pct(close, 126)
        mom12 = _pct(close, 252)
        ret20 = _pct(close, 20)

        # 변동성/최대낙폭/상관
        vol30 = _stdev(close, 30)
        vol60 = _stdev(close, 60)
        dd180 = _max_drawdown(close, 180)
        try:
            spy_df = normalize_yf_columns(
                yf.download("SPY", period=period, interval=interval, progress=False, auto_adjust=True)
            )
            spy = spy_df["Close"]
            corr_spy = _corr(close, spy, 90)
        except Exception:
            corr_spy = None

        data = {
            "ticker": ticker,
            "mom1": mom1, "mom3": mom3, "mom6": mom6, "mom12": mom12,
            "ret20": ret20,
            "vol30": vol30, "vol60": vol60,
            "dd180": dd180,
            "corr_spy": corr_spy,
            "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # diskcache에 저장 (TTL 4시간)
        if use_cache:
            cache_manager.set(cache_key, data, TTL.METRICS)

        return data
    except Exception:
        return {"ticker": ticker}

