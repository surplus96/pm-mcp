"""
Finnhub API Integration Module
- Company News
- Insider Trading
- Analyst Recommendations
- Company Earnings
- Basic Financials
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.resilience import (
    CircuitBreaker, retry_with_backoff, Timeout
)

# Finnhub Circuit Breaker
circuit_finnhub = CircuitBreaker(name="finnhub", failure_threshold=3, reset_timeout=120)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _get_api_key() -> str:
    """Finnhub API 키 가져오기"""
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise ValueError("FINNHUB_API_KEY 환경변수가 설정되지 않았습니다.")
    return key


@retry_with_backoff(attempts=3, min_wait=1, max_wait=10)
def _finnhub_request(endpoint: str, params: Dict = None) -> Dict:
    """Finnhub API 공통 요청 함수"""

    def _do_request():
        api_key = _get_api_key()
        url = f"{FINNHUB_BASE_URL}/{endpoint}"

        request_params = params or {}
        request_params["token"] = api_key

        response = requests.get(url, params=request_params, timeout=Timeout.DEFAULT)
        response.raise_for_status()
        return response.json()

    return circuit_finnhub.call(_do_request)


# ============================================================
# Company News
# ============================================================

def get_company_news(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    회사 관련 뉴스 가져오기

    Args:
        symbol: 종목 심볼 (예: AAPL)
        from_date: 시작일 (YYYY-MM-DD)
        to_date: 종료일 (YYYY-MM-DD)

    Returns:
        뉴스 목록과 요약
    """
    cache_key = f"finnhub_news_{symbol}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    # 기본 날짜 설정 (최근 7일)
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        data = _finnhub_request("company-news", {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date
        })

        # 뉴스 정리
        news_items = []
        for item in data[:20]:  # 최대 20개
            news_items.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", "")[:300] if item.get("summary") else "",
                "source": item.get("source", ""),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M") if item.get("datetime") else "",
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "related": item.get("related", "")
            })

        # 감성 분류 (간단한 키워드 기반)
        positive_keywords = ["surge", "beat", "record", "growth", "profit", "upgrade", "bullish", "gain", "rally"]
        negative_keywords = ["drop", "fall", "miss", "decline", "loss", "downgrade", "bearish", "plunge", "crash"]

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for item in news_items:
            text = (item["headline"] + " " + item["summary"]).lower()
            pos_count = sum(1 for kw in positive_keywords if kw in text)
            neg_count = sum(1 for kw in negative_keywords if kw in text)

            if pos_count > neg_count:
                sentiment_counts["positive"] += 1
            elif neg_count > pos_count:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1

        result = {
            "symbol": symbol.upper(),
            "period": f"{from_date} ~ {to_date}",
            "total_count": len(news_items),
            "sentiment_summary": sentiment_counts,
            "news": news_items,
            "fetched_at": datetime.now().isoformat()
        }

        cache_manager.set(cache_key, result, ttl=TTL.NEWS)
        return result

    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "news": []
        }


# ============================================================
# Insider Trading
# ============================================================

def get_insider_transactions(symbol: str) -> Dict[str, Any]:
    """
    내부자 거래 내역 가져오기

    Args:
        symbol: 종목 심볼

    Returns:
        내부자 거래 목록과 요약
    """
    cache_key = f"finnhub_insider_{symbol}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        data = _finnhub_request("stock/insider-transactions", {"symbol": symbol.upper()})

        transactions = data.get("data", [])[:30]  # 최대 30개

        # 거래 정리
        processed = []
        buy_volume = 0
        sell_volume = 0

        for tx in transactions:
            tx_type = tx.get("transactionType", "")
            shares = tx.get("share", 0) or 0

            if "Buy" in tx_type or "Purchase" in tx_type:
                buy_volume += shares
            elif "Sale" in tx_type or "Sell" in tx_type:
                sell_volume += shares

            processed.append({
                "name": tx.get("name", ""),
                "transaction_type": tx_type,
                "shares": shares,
                "price": tx.get("transactionPrice", 0),
                "value": tx.get("transactionValue", 0),
                "filing_date": tx.get("filingDate", ""),
                "transaction_date": tx.get("transactionDate", "")
            })

        # 내부자 신호 판단
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
            if buy_ratio > 0.7:
                insider_signal = "Strong Buy Signal"
            elif buy_ratio > 0.5:
                insider_signal = "Moderate Buy Signal"
            elif buy_ratio < 0.3:
                insider_signal = "Strong Sell Signal"
            elif buy_ratio < 0.5:
                insider_signal = "Moderate Sell Signal"
            else:
                insider_signal = "Neutral"
        else:
            insider_signal = "No Recent Activity"

        result = {
            "symbol": symbol.upper(),
            "summary": {
                "total_transactions": len(processed),
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "net_volume": buy_volume - sell_volume,
                "insider_signal": insider_signal
            },
            "transactions": processed,
            "fetched_at": datetime.now().isoformat()
        }

        cache_manager.set(cache_key, result, ttl=TTL.FILING)
        return result

    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "transactions": []
        }


# ============================================================
# Analyst Recommendations
# ============================================================

def get_analyst_recommendations(symbol: str) -> Dict[str, Any]:
    """
    애널리스트 추천 등급 가져오기

    Args:
        symbol: 종목 심볼

    Returns:
        애널리스트 추천 현황
    """
    cache_key = f"finnhub_analyst_{symbol}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        data = _finnhub_request("stock/recommendation", {"symbol": symbol.upper()})

        if not data:
            return {
                "symbol": symbol.upper(),
                "message": "No analyst recommendations available",
                "recommendations": []
            }

        # 최근 추천 데이터
        recent = data[:6]  # 최근 6개월

        recommendations = []
        for rec in recent:
            recommendations.append({
                "period": rec.get("period", ""),
                "strong_buy": rec.get("strongBuy", 0),
                "buy": rec.get("buy", 0),
                "hold": rec.get("hold", 0),
                "sell": rec.get("sell", 0),
                "strong_sell": rec.get("strongSell", 0)
            })

        # 최신 컨센서스 계산
        if recommendations:
            latest = recommendations[0]
            total = (latest["strong_buy"] + latest["buy"] + latest["hold"] +
                    latest["sell"] + latest["strong_sell"])

            if total > 0:
                # 가중 평균 (Strong Buy=5, Buy=4, Hold=3, Sell=2, Strong Sell=1)
                weighted_sum = (
                    latest["strong_buy"] * 5 +
                    latest["buy"] * 4 +
                    latest["hold"] * 3 +
                    latest["sell"] * 2 +
                    latest["strong_sell"] * 1
                )
                avg_rating = weighted_sum / total

                if avg_rating >= 4.5:
                    consensus = "Strong Buy"
                elif avg_rating >= 3.5:
                    consensus = "Buy"
                elif avg_rating >= 2.5:
                    consensus = "Hold"
                elif avg_rating >= 1.5:
                    consensus = "Sell"
                else:
                    consensus = "Strong Sell"

                consensus_detail = {
                    "rating": round(avg_rating, 2),
                    "consensus": consensus,
                    "total_analysts": total
                }
            else:
                consensus_detail = {"message": "No data"}
        else:
            consensus_detail = {"message": "No data"}

        # 트렌드 분석
        if len(recommendations) >= 2:
            latest = recommendations[0]
            previous = recommendations[1]

            latest_bullish = latest["strong_buy"] + latest["buy"]
            previous_bullish = previous["strong_buy"] + previous["buy"]

            if latest_bullish > previous_bullish:
                trend = "Improving"
            elif latest_bullish < previous_bullish:
                trend = "Deteriorating"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient Data"

        result = {
            "symbol": symbol.upper(),
            "consensus": consensus_detail,
            "trend": trend,
            "recommendations": recommendations,
            "fetched_at": datetime.now().isoformat()
        }

        cache_manager.set(cache_key, result, ttl=TTL.FILING)
        return result

    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "recommendations": []
        }


# ============================================================
# Earnings Calendar
# ============================================================

def get_earnings_calendar(
    symbol: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    실적 발표 일정 가져오기

    Args:
        symbol: 종목 심볼 (None이면 전체)
        from_date: 시작일
        to_date: 종료일

    Returns:
        실적 발표 일정
    """
    # 기본 날짜 (이번 주 ~ 다음 주)
    if not from_date:
        from_date = datetime.now().strftime("%Y-%m-%d")
    if not to_date:
        to_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

    cache_key = f"finnhub_earnings_{symbol or 'all'}_{from_date}_{to_date}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        params = {"from": from_date, "to": to_date}
        if symbol:
            params["symbol"] = symbol.upper()

        data = _finnhub_request("calendar/earnings", params)

        earnings = data.get("earningsCalendar", [])

        # 심볼 필터링
        if symbol:
            earnings = [e for e in earnings if e.get("symbol", "").upper() == symbol.upper()]

        processed = []
        for e in earnings[:50]:  # 최대 50개
            processed.append({
                "symbol": e.get("symbol", ""),
                "date": e.get("date", ""),
                "hour": e.get("hour", ""),  # bmo (before market open), amc (after market close)
                "eps_estimate": e.get("epsEstimate"),
                "eps_actual": e.get("epsActual"),
                "revenue_estimate": e.get("revenueEstimate"),
                "revenue_actual": e.get("revenueActual"),
                "quarter": e.get("quarter"),
                "year": e.get("year")
            })

        result = {
            "period": f"{from_date} ~ {to_date}",
            "count": len(processed),
            "earnings": processed,
            "fetched_at": datetime.now().isoformat()
        }

        if symbol:
            result["symbol"] = symbol.upper()

        cache_manager.set(cache_key, result, ttl=TTL.NEWS)
        return result

    except Exception as e:
        return {
            "error": str(e),
            "earnings": []
        }


# ============================================================
# Basic Financials
# ============================================================

def get_basic_financials(symbol: str, metric: str = "all") -> Dict[str, Any]:
    """
    기본 재무 지표 가져오기

    Args:
        symbol: 종목 심볼
        metric: 'all', 'price', 'valuation', 'margin' 등

    Returns:
        재무 지표
    """
    cache_key = f"finnhub_financials_{symbol}_{metric}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        data = _finnhub_request("stock/metric", {
            "symbol": symbol.upper(),
            "metric": metric
        })

        metrics = data.get("metric", {})

        # 주요 지표 추출
        key_metrics = {
            "valuation": {
                "pe_ratio": metrics.get("peNormalizedAnnual"),
                "pe_ttm": metrics.get("peTTM"),
                "pb_ratio": metrics.get("pbQuarterly"),
                "ps_ratio": metrics.get("psTTM"),
                "ev_ebitda": metrics.get("enterpriseValueEbitdaTTM"),
                "peg_ratio": metrics.get("pegRatio")
            },
            "profitability": {
                "roe": metrics.get("roeTTM"),
                "roa": metrics.get("roaTTM"),
                "gross_margin": metrics.get("grossMarginTTM"),
                "operating_margin": metrics.get("operatingMarginTTM"),
                "net_margin": metrics.get("netMarginTTM")
            },
            "growth": {
                "revenue_growth_3y": metrics.get("revenueGrowth3Y"),
                "revenue_growth_5y": metrics.get("revenueGrowth5Y"),
                "eps_growth_3y": metrics.get("epsGrowth3Y"),
                "eps_growth_5y": metrics.get("epsGrowth5Y")
            },
            "dividend": {
                "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
                "dividend_per_share": metrics.get("dividendPerShareAnnual"),
                "payout_ratio": metrics.get("payoutRatioTTM")
            },
            "price": {
                "52w_high": metrics.get("52WeekHigh"),
                "52w_low": metrics.get("52WeekLow"),
                "52w_return": metrics.get("52WeekPriceReturnDaily"),
                "beta": metrics.get("beta"),
                "10d_avg_volume": metrics.get("10DayAverageTradingVolume")
            }
        }

        # 투자 점수 계산 (간단한 기준)
        scores = {}

        # Valuation Score (낮을수록 좋음)
        pe = key_metrics["valuation"]["pe_ttm"]
        if pe and pe > 0:
            if pe < 15:
                scores["valuation"] = "Undervalued"
            elif pe < 25:
                scores["valuation"] = "Fair"
            else:
                scores["valuation"] = "Expensive"

        # Profitability Score
        roe = key_metrics["profitability"]["roe"]
        if roe:
            if roe > 20:
                scores["profitability"] = "Excellent"
            elif roe > 10:
                scores["profitability"] = "Good"
            else:
                scores["profitability"] = "Poor"

        # Growth Score
        rev_growth = key_metrics["growth"]["revenue_growth_3y"]
        if rev_growth:
            if rev_growth > 20:
                scores["growth"] = "High Growth"
            elif rev_growth > 5:
                scores["growth"] = "Moderate"
            else:
                scores["growth"] = "Slow"

        result = {
            "symbol": symbol.upper(),
            "metrics": key_metrics,
            "scores": scores,
            "fetched_at": datetime.now().isoformat()
        }

        cache_manager.set(cache_key, result, ttl=TTL.DAILY)
        return result

    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "metrics": {}
        }


# ============================================================
# Combined Summary
# ============================================================

def get_finnhub_summary(symbol: str) -> Dict[str, Any]:
    """
    Finnhub 데이터 종합 요약

    Args:
        symbol: 종목 심볼

    Returns:
        뉴스, 내부자, 애널리스트, 재무 종합 요약
    """
    cache_key = f"finnhub_summary_{symbol}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    # 모든 데이터 수집
    news = get_company_news(symbol)
    insider = get_insider_transactions(symbol)
    analyst = get_analyst_recommendations(symbol)
    financials = get_basic_financials(symbol)

    # 종합 신호 계산
    signals = []

    # 뉴스 감성
    if news.get("sentiment_summary"):
        sentiment = news["sentiment_summary"]
        if sentiment.get("positive", 0) > sentiment.get("negative", 0):
            signals.append(("News Sentiment", "Positive", 1))
        elif sentiment.get("negative", 0) > sentiment.get("positive", 0):
            signals.append(("News Sentiment", "Negative", -1))
        else:
            signals.append(("News Sentiment", "Neutral", 0))

    # 내부자 신호
    if insider.get("summary", {}).get("insider_signal"):
        signal = insider["summary"]["insider_signal"]
        if "Buy" in signal:
            signals.append(("Insider Activity", signal, 1 if "Strong" in signal else 0.5))
        elif "Sell" in signal:
            signals.append(("Insider Activity", signal, -1 if "Strong" in signal else -0.5))
        else:
            signals.append(("Insider Activity", signal, 0))

    # 애널리스트 컨센서스
    if analyst.get("consensus", {}).get("consensus"):
        consensus = analyst["consensus"]["consensus"]
        if consensus in ["Strong Buy", "Buy"]:
            signals.append(("Analyst Consensus", consensus, 1 if "Strong" in consensus else 0.5))
        elif consensus in ["Strong Sell", "Sell"]:
            signals.append(("Analyst Consensus", consensus, -1 if "Strong" in consensus else -0.5))
        else:
            signals.append(("Analyst Consensus", consensus, 0))

    # 종합 점수
    if signals:
        total_score = sum(s[2] for s in signals)
        avg_score = total_score / len(signals)

        if avg_score > 0.5:
            overall = "Bullish"
        elif avg_score < -0.5:
            overall = "Bearish"
        else:
            overall = "Neutral"
    else:
        overall = "Insufficient Data"
        avg_score = 0

    result = {
        "symbol": symbol.upper(),
        "overall_signal": overall,
        "signal_score": round(avg_score, 2),
        "signals": [{"factor": s[0], "signal": s[1], "score": s[2]} for s in signals],
        "news_summary": {
            "count": news.get("total_count", 0),
            "sentiment": news.get("sentiment_summary", {})
        },
        "insider_summary": insider.get("summary", {}),
        "analyst_summary": analyst.get("consensus", {}),
        "financial_scores": financials.get("scores", {}),
        "fetched_at": datetime.now().isoformat()
    }

    cache_manager.set(cache_key, result, ttl=TTL.NEWS)
    return result
