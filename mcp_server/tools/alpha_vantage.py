"""
Alpha Vantage API 모듈 - 기술적 지표

Features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- SMA/EMA (Simple/Exponential Moving Average)
- ADX (Average Directional Index)
- 캐싱 및 폴백 지원
"""
from __future__ import annotations
from typing import Dict
import os
import logging
import requests

from mcp_server.config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_CALL_DELAY
from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.resilience import (
    retry_with_backoff, Timeout, CircuitBreaker, CircuitOpenError
)

logger = logging.getLogger(__name__)

# API 설정
BASE_URL = "https://www.alphavantage.co/query"
circuit_av = CircuitBreaker.get_instance("alpha_vantage", failure_threshold=5, reset_timeout=120)


def _get_api_key() -> str:
    """API 키 가져오기"""
    key = ALPHA_VANTAGE_API_KEY or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is not set")
    return key


@retry_with_backoff(attempts=2, min_wait=1, max_wait=5)
def _call_api(function: str, symbol: str, **params) -> Dict:
    """Alpha Vantage API 호출"""
    def _do_request():
        api_key = _get_api_key()
        query_params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            **params
        }
        resp = requests.get(BASE_URL, params=query_params, timeout=Timeout.DEFAULT)
        resp.raise_for_status()
        data = resp.json()

        # API 에러 체크
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if "Note" in data:
            # Rate limit 경고
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
            raise ValueError("Rate limit exceeded")
        if "Information" in data:
            logger.warning(f"Alpha Vantage info response for {function} {symbol}: {data['Information'][:120]}")
            raise ValueError(f"API limit: {data['Information'][:120]}")

        # Diagnostic: warn if no data keys found
        meta_keys = {"Meta Data", "Error Message", "Note", "Information"}
        data_keys = [k for k in data.keys() if k not in meta_keys]
        if not data_keys:
            logger.warning(f"Alpha Vantage returned no data keys for {function} {symbol}. Keys: {list(data.keys())}")

        return data

    return circuit_av.call(_do_request)


# ===== 기술적 지표 함수들 =====

def get_rsi(
    symbol: str,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
    use_cache: bool = True
) -> Dict:
    """RSI (Relative Strength Index) 조회

    Args:
        symbol: 티커 심볼
        interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        time_period: RSI 계산 기간 (기본 14)
        series_type: close, open, high, low

    Returns:
        {"symbol": str, "indicator": "RSI", "interval": str, "values": [{date, rsi}, ...]}
    """
    cache_key = f"av:rsi:{symbol}:{interval}:{time_period}"

    if use_cache:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

    try:
        data = _call_api(
            "RSI",
            symbol,
            interval=interval,
            time_period=time_period,
            series_type=series_type
        )

        # 데이터 파싱
        key = f"Technical Analysis: RSI"
        raw_data = data.get(key, {})

        values = []
        for date, val in list(raw_data.items())[:30]:  # 최근 30개
            values.append({
                "date": date,
                "rsi": float(val.get("RSI", 0))
            })

        result = {
            "symbol": symbol,
            "indicator": "RSI",
            "interval": interval,
            "time_period": time_period,
            "values": values,
            "latest": values[0] if values else None
        }

        if use_cache and values:
            cache_manager.set(cache_key, result, TTL.DAILY)

        return result

    except CircuitOpenError:
        logger.warning(f"Alpha Vantage circuit open for RSI: {symbol}")
        return {"symbol": symbol, "indicator": "RSI", "error": "circuit_open", "values": []}
    except Exception as e:
        logger.warning(f"Failed to get RSI for {symbol}: {e}")
        return {"symbol": symbol, "indicator": "RSI", "error": str(e), "values": []}


def get_macd(
    symbol: str,
    interval: str = "daily",
    series_type: str = "close",
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
    use_cache: bool = True
) -> Dict:
    """MACD (Moving Average Convergence Divergence) 조회

    Args:
        symbol: 티커 심볼
        interval: 시간 간격
        fastperiod: 빠른 EMA 기간 (기본 12)
        slowperiod: 느린 EMA 기간 (기본 26)
        signalperiod: 시그널 라인 기간 (기본 9)

    Returns:
        {"symbol": str, "indicator": "MACD", "values": [{date, macd, signal, histogram}, ...]}
    """
    cache_key = f"av:macd:{symbol}:{interval}:{fastperiod}:{slowperiod}:{signalperiod}"

    if use_cache:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

    try:
        data = _call_api(
            "MACD",
            symbol,
            interval=interval,
            series_type=series_type,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )

        key = "Technical Analysis: MACD"
        raw_data = data.get(key, {})

        values = []
        for date, val in list(raw_data.items())[:30]:
            values.append({
                "date": date,
                "macd": float(val.get("MACD", 0)),
                "signal": float(val.get("MACD_Signal", 0)),
                "histogram": float(val.get("MACD_Hist", 0))
            })

        result = {
            "symbol": symbol,
            "indicator": "MACD",
            "interval": interval,
            "params": {"fast": fastperiod, "slow": slowperiod, "signal": signalperiod},
            "values": values,
            "latest": values[0] if values else None
        }

        if use_cache and values:
            cache_manager.set(cache_key, result, TTL.DAILY)

        return result

    except CircuitOpenError:
        logger.warning(f"Alpha Vantage circuit open for MACD: {symbol}")
        return {"symbol": symbol, "indicator": "MACD", "error": "circuit_open", "values": []}
    except Exception as e:
        logger.warning(f"Failed to get MACD for {symbol}: {e}")
        return {"symbol": symbol, "indicator": "MACD", "error": str(e), "values": []}


def get_bbands(
    symbol: str,
    interval: str = "daily",
    time_period: int = 20,
    series_type: str = "close",
    nbdevup: int = 2,
    nbdevdn: int = 2,
    use_cache: bool = True
) -> Dict:
    """Bollinger Bands 조회

    Args:
        symbol: 티커 심볼
        time_period: 이동평균 기간 (기본 20)
        nbdevup: 상단 밴드 표준편차 배수 (기본 2)
        nbdevdn: 하단 밴드 표준편차 배수 (기본 2)

    Returns:
        {"symbol": str, "indicator": "BBANDS", "values": [{date, upper, middle, lower}, ...]}
    """
    cache_key = f"av:bbands:{symbol}:{interval}:{time_period}"

    if use_cache:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

    try:
        data = _call_api(
            "BBANDS",
            symbol,
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )

        key = "Technical Analysis: BBANDS"
        raw_data = data.get(key, {})

        values = []
        for date, val in list(raw_data.items())[:30]:
            values.append({
                "date": date,
                "upper": float(val.get("Real Upper Band", 0)),
                "middle": float(val.get("Real Middle Band", 0)),
                "lower": float(val.get("Real Lower Band", 0))
            })

        result = {
            "symbol": symbol,
            "indicator": "BBANDS",
            "interval": interval,
            "time_period": time_period,
            "values": values,
            "latest": values[0] if values else None
        }

        if use_cache and values:
            cache_manager.set(cache_key, result, TTL.DAILY)

        return result

    except CircuitOpenError:
        logger.warning(f"Alpha Vantage circuit open for BBANDS: {symbol}")
        return {"symbol": symbol, "indicator": "BBANDS", "error": "circuit_open", "values": []}
    except Exception as e:
        logger.warning(f"Failed to get BBANDS for {symbol}: {e}")
        return {"symbol": symbol, "indicator": "BBANDS", "error": str(e), "values": []}


# ===== 종합 기술적 분석 =====

def get_technical_summary(symbol: str, use_cache: bool = True) -> Dict:
    """종합 기술적 분석 요약

    RSI, MACD, Bollinger Bands를 한 번에 조회하고 매매 신호 해석

    Returns:
        {
            "symbol": str,
            "rsi": {...},
            "macd": {...},
            "bbands": {...},
            "signals": {"rsi": str, "macd": str, "bbands": str, "overall": str}
        }
    """
    cache_key = f"av:summary:{symbol}"

    if use_cache:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

    import time

    # 각 지표 조회 with rate-limit delay between API calls.
    # Cache hits are instant; delays only matter when actual API calls are made.
    rsi = get_rsi(symbol, use_cache=use_cache)
    time.sleep(ALPHA_VANTAGE_CALL_DELAY)
    macd = get_macd(symbol, use_cache=use_cache)
    time.sleep(ALPHA_VANTAGE_CALL_DELAY)
    bbands = get_bbands(symbol, use_cache=use_cache)

    # 신호 해석
    signals = _interpret_signals(rsi, macd, bbands)

    result = {
        "symbol": symbol,
        "rsi": rsi,
        "macd": macd,
        "bbands": bbands,
        "signals": signals
    }

    if use_cache:
        cache_manager.set(cache_key, result, TTL.DAILY)

    return result


def _interpret_signals(rsi: Dict, macd: Dict, bbands: Dict) -> Dict:
    """기술적 지표 신호 해석"""
    signals = {
        "rsi": "neutral",
        "macd": "neutral",
        "bbands": "neutral",
        "overall": "neutral"
    }

    bullish = 0
    bearish = 0

    # RSI 해석
    rsi_latest = rsi.get("latest", {})
    rsi_val = rsi_latest.get("rsi") if rsi_latest else None
    if rsi_val is not None:
        if rsi_val < 30:
            signals["rsi"] = "oversold (bullish)"
            bullish += 1
        elif rsi_val > 70:
            signals["rsi"] = "overbought (bearish)"
            bearish += 1
        elif rsi_val < 50:
            signals["rsi"] = "weak"
        else:
            signals["rsi"] = "strong"

    # MACD 해석
    macd_latest = macd.get("latest", {})
    if macd_latest:
        macd_val = macd_latest.get("macd", 0)
        signal_val = macd_latest.get("signal", 0)
        histogram = macd_latest.get("histogram", 0)

        if histogram > 0 and macd_val > signal_val:
            signals["macd"] = "bullish crossover"
            bullish += 1
        elif histogram < 0 and macd_val < signal_val:
            signals["macd"] = "bearish crossover"
            bearish += 1
        elif histogram > 0:
            signals["macd"] = "bullish"
        else:
            signals["macd"] = "bearish"

    # Bollinger Bands 해석 (현재 가격 위치 필요 - 여기서는 중간값 기준)
    bbands_latest = bbands.get("latest", {})
    if bbands_latest:
        upper = bbands_latest.get("upper", 0)
        middle = bbands_latest.get("middle", 0)
        lower = bbands_latest.get("lower", 0)

        # 밴드 폭 분석
        if upper > 0 and lower > 0:
            band_width = (upper - lower) / middle if middle > 0 else 0
            if band_width < 0.05:
                signals["bbands"] = "squeeze (volatility contraction)"
            elif band_width > 0.15:
                signals["bbands"] = "expansion (high volatility)"
            else:
                signals["bbands"] = "normal"

    # 종합 신호
    if bullish > bearish:
        signals["overall"] = "bullish"
    elif bearish > bullish:
        signals["overall"] = "bearish"
    else:
        signals["overall"] = "neutral"

    signals["score"] = {"bullish": bullish, "bearish": bearish}

    return signals


# ===== API 상태 확인 =====

