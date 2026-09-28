"""
Data Integration Layer
- Alpha Vantage + Finnhub + Yahoo Finance 통합
- 멀티소스 데이터 병합
- 신호 강도 분석
"""

from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.alpha_vantage import (
    get_technical_summary
)
from mcp_server.tools.finnhub_api import (
    get_company_news, get_insider_transactions,
    get_analyst_recommendations, get_basic_financials
)
from mcp_server.tools.market_data import get_prices


class DataIntegrator:
    """멀티소스 데이터 통합 클래스"""

    def __init__(self):
        self.sources = {
            "technical": "Alpha Vantage",
            "fundamental": "Finnhub",
            "price": "Yahoo Finance"
        }

    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        종합 분석 데이터 가져오기

        Args:
            symbol: 종목 심볼

        Returns:
            기술적/기본적/뉴스 종합 분석
        """
        cache_key = f"integrated_analysis_{symbol}"
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

        results = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "data_sources": self.sources
        }

        # 병렬로 데이터 수집
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._get_technical_data, symbol): "technical",
                executor.submit(self._get_fundamental_data, symbol): "fundamental",
                executor.submit(self._get_news_sentiment, symbol): "sentiment",
                executor.submit(self._get_price_data, symbol): "price"
            }

            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    data = future.result()
                    results[data_type] = data
                except Exception as e:
                    results[data_type] = {"error": str(e)}

        # 종합 신호 계산
        results["composite_signal"] = self._calculate_composite_signal(results)

        cache_manager.set(cache_key, results, ttl=TTL.NEWS)
        return results

    def _get_technical_data(self, symbol: str) -> Dict:
        """기술적 분석 데이터.

        US → Alpha Vantage (premium endpoints, free-tier는 top-3만).
        KR → 로컬 계산 (PyKrx OHLCV + ``TechnicalFactors.calculate_all``).
        Alpha Vantage 는 한국 종목을 지원하지 않으므로 로그가 warning 으로
        가득 차면서 최종적으로 빈 결과를 반환하던 문제를 FR-K02 경로로 해결.
        """
        from mcp_server.tools.yf_utils import detect_market

        if detect_market(symbol) == "KR":
            return self._get_technical_data_kr(symbol)
        try:
            summary = get_technical_summary(symbol)
            return {
                "rsi": summary.get("rsi", {}),
                "macd": summary.get("macd", {}),
                "bbands": summary.get("bbands", {}),
                "signals": summary.get("signals", {}),
                "overall": summary.get("signals", {}).get("overall", "N/A"),
            }
        except Exception as e:
            return {"error": str(e), "overall": "N/A"}

    def _get_technical_data_kr(self, symbol: str) -> Dict:
        """KR-path: OHLCV 로컬 계산으로 RSI/MACD/BBands/신호 도출.

        ``TechnicalFactors.calculate_all`` 의 반환 키는 PascalCase 이고
        각 지표는 지표 고유 단위의 raw 값이다 (RSI 0~100, MACD KRW 단위
        모멘텀 차 등). 따라서 임계치도 각 지표별로 달라야 한다.
        """
        try:
            from mcp_server.tools.technical_indicators import TechnicalFactors
            df = get_prices(symbol)
            if df is None or df.empty:
                return {"overall": "N/A", "note": "no OHLCV from PyKrx", "source": "pykrx+local"}
            factors = TechnicalFactors.calculate_all(df)

            rsi_v = factors.get("RSI")
            macd_v = factors.get("MACD")
            bb_v = factors.get("BB_Width")
            adx_v = factors.get("ADX")
            ma_cross_v = factors.get("MA_Cross")

            def _rsi_label(v):
                if v is None:
                    return "N/A"
                if v >= 70:
                    return "Overbought"
                if v <= 30:
                    return "Oversold"
                return "Neutral"

            def _sign_label(v, pos="Bullish", neg="Bearish"):
                if v is None:
                    return "N/A"
                if v > 0:
                    return pos
                if v < 0:
                    return neg
                return "Neutral"

            # 종합 신호: MACD 양수 + RSI 40~70 + ADX>20 → Bullish, MACD 음수 + RSI<50 → Bearish
            bull_votes = sum([
                1 if (macd_v or 0) > 0 else 0,
                1 if (rsi_v or 50) >= 50 and (rsi_v or 50) < 70 else 0,
                1 if (adx_v or 0) > 20 and (ma_cross_v or 0) > 0 else 0,
            ])
            bear_votes = sum([
                1 if (macd_v or 0) < 0 else 0,
                1 if (rsi_v or 50) < 50 else 0,
                1 if (ma_cross_v or 0) < 0 else 0,
            ])
            if bull_votes >= 2 and bull_votes > bear_votes:
                overall = "Bullish"
                score = min(1.0, 0.3 + 0.2 * bull_votes)
            elif bear_votes >= 2 and bear_votes > bull_votes:
                overall = "Bearish"
                score = -min(1.0, 0.3 + 0.2 * bear_votes)
            else:
                overall = "Neutral"
                score = 0.0

            return {
                "rsi": {"value": rsi_v, "signal": _rsi_label(rsi_v)},
                "macd": {"value": macd_v, "signal": _sign_label(macd_v)},
                "bbands": {"value": bb_v, "signal": "Expanding" if (bb_v or 0) > 20 else "Squeezing"},
                "adx": {"value": adx_v, "signal": "Trending" if (adx_v or 0) > 20 else "Range"},
                "signals": {"overall": overall, "score": score},
                "overall": overall,
                "source": "pykrx+local",
            }
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "overall": "N/A", "source": "pykrx+local"}

    def _get_fundamental_data(self, symbol: str) -> Dict:
        """기본적 분석 데이터.

        US → Finnhub (기존 path).
        KR → DART (ROE/ROA/Operating_Margin) + PyKrx fundamentals.
        Finnhub 는 한국 상장사를 커버하지 않으므로 KR 에서는 아예 호출하지 않는다.
        """
        from mcp_server.tools.yf_utils import detect_market

        if detect_market(symbol) == "KR":
            return self._get_fundamental_data_kr(symbol)
        try:
            financials = get_basic_financials(symbol)
            analyst = get_analyst_recommendations(symbol)
            insider = get_insider_transactions(symbol)

            return {
                "valuation": financials.get("metrics", {}).get("valuation", {}),
                "profitability": financials.get("metrics", {}).get("profitability", {}),
                "growth": financials.get("metrics", {}).get("growth", {}),
                "scores": financials.get("scores", {}),
                "analyst_consensus": analyst.get("consensus", {}),
                "analyst_trend": analyst.get("trend", "N/A"),
                "insider_signal": insider.get("summary", {}).get("insider_signal", "N/A")
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_fundamental_data_kr(self, symbol: str) -> Dict:
        """KR 기본적 분석 — DART + PyKrx 조합."""
        out: Dict[str, Any] = {"source": "dart+pykrx"}
        # DART (K-IFRS) — ROE/ROA/Operating_Margin/Net_Margin/Revenue_Growth
        try:
            from mcp_server.tools.dart import get_dart_client
            client = get_dart_client()
            if client.ready:
                fin = client.get_financials(symbol)
                out["profitability"] = {
                    "roe": fin.get("ROE"),
                    "roa": fin.get("ROA"),
                    "operating_margin": fin.get("Operating_Margin"),
                    "net_margin": fin.get("Net_Margin"),
                }
                out["growth"] = {"revenue_growth": fin.get("Revenue_Growth")}
                out["year"] = fin.get("year")
        except Exception as e:  # noqa: BLE001
            out["dart_error"] = str(e)

        # PyKrx fundamentals (PER/PBR/EPS 등 장 마감 기준 스냅샷)
        try:
            from mcp_server.tools.kr_market_data import get_kr_adapter
            snap = get_kr_adapter().get_fundamental(symbol)
            if snap:
                out["valuation"] = {
                    "pe": snap.get("PER"),
                    "pb": snap.get("PBR"),
                    "eps": snap.get("EPS"),
                    "bps": snap.get("BPS"),
                    "div_yield": snap.get("DIV"),
                }
        except Exception as e:  # noqa: BLE001
            out["pykrx_error"] = str(e)

        # Finnhub 에만 있던 인사이더/애널리스트 필드는 KR 에서 비움 (명시적 표시).
        out.setdefault("analyst_consensus", {})
        out.setdefault("analyst_trend", "N/A")
        out.setdefault("insider_signal", "N/A")
        return out

    def _get_news_sentiment(self, symbol: str) -> Dict:
        """뉴스 감성 분석.

        US → Finnhub 기업 뉴스.
        KR → Google News KR RSS (``news_search_kr``) → ``analyze_ticker_news``.
        """
        from mcp_server.tools.yf_utils import detect_market

        if detect_market(symbol) == "KR":
            try:
                from mcp_server.tools.news_sentiment import analyze_ticker_news
                r = analyze_ticker_news(symbol, lookback_days=14)
                if isinstance(r, dict):
                    return {
                        "count": r.get("total", 0),
                        "sentiment": r.get("sentiment_distribution", {}),
                        "overall": r.get("overall", "neutral"),
                        "score": r.get("score", 0),
                        "period": "14d",
                        "source": "google-news-kr",
                    }
                return {"error": "unexpected KR sentiment shape"}
            except Exception as e:  # noqa: BLE001
                return {"error": str(e), "source": "google-news-kr"}
        try:
            news = get_company_news(symbol)
            return {
                "count": news.get("total_count", 0),
                "sentiment": news.get("sentiment_summary", {}),
                "period": news.get("period", "")
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_price_data(self, symbol: str) -> Dict:
        """가격 데이터"""
        try:
            from datetime import timedelta
            end = datetime.now()
            start = end - timedelta(days=90)
            df = get_prices(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )

            if df is None or df.empty or "Close" not in df.columns:
                return {"error": "No price data"}

            closes = df["Close"].dropna()
            if len(closes) < 2:
                return {"error": "Insufficient price data"}

            first_close = float(closes.iloc[0])
            last_close = float(closes.iloc[-1])

            # 수익률 계산
            returns_3m = ((last_close - first_close) / first_close) * 100 if first_close > 0 else None

            # 변동성 계산
            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) > 1:
                import statistics
                volatility = statistics.stdev(daily_returns.tolist()) * (252 ** 0.5) * 100
            else:
                volatility = None

            # 날짜 정보
            latest_date = None
            if "Date" in df.columns:
                latest_date = str(df["Date"].iloc[-1])

            return {
                "latest_price": round(last_close, 2),
                "latest_date": latest_date,
                "returns_3m": round(returns_3m, 2) if returns_3m is not None else None,
                "volatility_annual": round(volatility, 2) if volatility is not None else None,
                "data_points": len(df)
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_composite_signal(self, data: Dict) -> Dict:
        """종합 신호 계산"""
        signals = []
        weights = {
            "technical": 0.30,
            "fundamental": 0.35,
            "sentiment": 0.20,
            "momentum": 0.15
        }

        # 기술적 신호 (-1 to 1)
        tech = data.get("technical", {})
        tech_signal = tech.get("overall", "N/A")
        if tech_signal == "Bullish":
            signals.append(("Technical", 1.0, weights["technical"]))
        elif tech_signal == "Bearish":
            signals.append(("Technical", -1.0, weights["technical"]))
        elif tech_signal == "Neutral":
            signals.append(("Technical", 0.0, weights["technical"]))

        # 기본적 신호
        fund = data.get("fundamental", {})

        # 애널리스트 컨센서스
        consensus = fund.get("analyst_consensus", {}).get("consensus", "")
        if consensus in ["Strong Buy", "Buy"]:
            signals.append(("Analyst", 1.0 if "Strong" in consensus else 0.5, weights["fundamental"] * 0.5))
        elif consensus in ["Strong Sell", "Sell"]:
            signals.append(("Analyst", -1.0 if "Strong" in consensus else -0.5, weights["fundamental"] * 0.5))
        elif consensus == "Hold":
            signals.append(("Analyst", 0.0, weights["fundamental"] * 0.5))

        # 내부자 신호
        insider = fund.get("insider_signal", "")
        if "Buy" in insider:
            signals.append(("Insider", 0.5 if "Moderate" in insider else 1.0, weights["fundamental"] * 0.5))
        elif "Sell" in insider:
            signals.append(("Insider", -0.5 if "Moderate" in insider else -1.0, weights["fundamental"] * 0.5))

        # 뉴스 감성
        sentiment = data.get("sentiment", {}).get("sentiment", {})
        if sentiment:
            pos = sentiment.get("positive", 0)
            neg = sentiment.get("negative", 0)
            total = pos + neg + sentiment.get("neutral", 0)
            if total > 0:
                score = (pos - neg) / total
                signals.append(("News", score, weights["sentiment"]))

        # 모멘텀 (3개월 수익률 기반)
        price = data.get("price", {})
        returns_3m = price.get("returns_3m")
        if returns_3m is not None:
            # -30% ~ +30% 를 -1 ~ +1 로 매핑
            momentum_score = max(-1, min(1, returns_3m / 30))
            signals.append(("Momentum", momentum_score, weights["momentum"]))

        # 가중 평균 계산
        if signals:
            total_weight = sum(s[2] for s in signals)
            weighted_sum = sum(s[1] * s[2] for s in signals)
            composite_score = weighted_sum / total_weight if total_weight > 0 else 0

            if composite_score > 0.3:
                overall = "Bullish"
            elif composite_score < -0.3:
                overall = "Bearish"
            else:
                overall = "Neutral"

            return {
                "overall": overall,
                "score": round(composite_score, 3),
                "components": [
                    {"factor": s[0], "signal": s[1], "weight": s[2]}
                    for s in signals
                ],
                "confidence": round(total_weight, 2)
            }

        return {"overall": "Insufficient Data", "score": 0, "components": []}

    def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """
        여러 종목 비교 분석

        Args:
            symbols: 종목 심볼 리스트

        Returns:
            비교 분석 결과
        """
        results = {
            "comparison_date": datetime.now().isoformat(),
            "stocks": []
        }

        # 병렬로 데이터 수집
        with ThreadPoolExecutor(max_workers=min(len(symbols), 5)) as executor:
            futures = {
                executor.submit(self.get_comprehensive_analysis, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    results["stocks"].append({
                        "symbol": symbol.upper(),
                        "composite": data.get("composite_signal", {}),
                        "technical": data.get("technical", {}).get("overall", "N/A"),
                        "analyst": data.get("fundamental", {}).get("analyst_consensus", {}).get("consensus", "N/A"),
                        "insider": data.get("fundamental", {}).get("insider_signal", "N/A"),
                        "returns_3m": data.get("price", {}).get("returns_3m"),
                        "volatility": data.get("price", {}).get("volatility_annual")
                    })
                except Exception as e:
                    results["stocks"].append({
                        "symbol": symbol.upper(),
                        "error": str(e)
                    })

        # 점수 기준 정렬
        results["stocks"].sort(
            key=lambda x: x.get("composite", {}).get("score", -999),
            reverse=True
        )

        # 랭킹 추가
        for i, stock in enumerate(results["stocks"], 1):
            stock["rank"] = i

        return results

    def get_investment_signals(self, symbol: str) -> Dict[str, Any]:
        """
        투자 신호 요약 (의사결정 지원용)

        Args:
            symbol: 종목 심볼

        Returns:
            Buy/Hold/Sell 신호와 근거
        """
        analysis = self.get_comprehensive_analysis(symbol)
        composite = analysis.get("composite_signal", {})

        score = composite.get("score", 0)
        overall = composite.get("overall", "Neutral")

        # 의사결정 신호
        if score > 0.5:
            decision = "Strong Buy"
            confidence = "High"
        elif score > 0.2:
            decision = "Buy"
            confidence = "Moderate"
        elif score > -0.2:
            decision = "Hold"
            confidence = "Moderate" if abs(score) < 0.1 else "Low"
        elif score > -0.5:
            decision = "Sell"
            confidence = "Moderate"
        else:
            decision = "Strong Sell"
            confidence = "High"

        # 주요 근거 추출
        reasons = []
        components = composite.get("components", [])

        for comp in components:
            factor = comp.get("factor", "")
            signal = comp.get("signal", 0)

            if signal > 0.5:
                reasons.append(f"{factor}: Strong positive signal")
            elif signal > 0.2:
                reasons.append(f"{factor}: Positive signal")
            elif signal < -0.5:
                reasons.append(f"{factor}: Strong negative signal")
            elif signal < -0.2:
                reasons.append(f"{factor}: Negative signal")

        # 리스크 요소
        risks = []
        volatility = analysis.get("price", {}).get("volatility_annual")
        if volatility and volatility > 40:
            risks.append(f"High volatility ({volatility:.1f}% annual)")

        insider = analysis.get("fundamental", {}).get("insider_signal", "")
        if "Sell" in insider:
            risks.append("Insider selling activity detected")

        returns_3m = analysis.get("price", {}).get("returns_3m")
        if returns_3m and returns_3m < -20:
            risks.append(f"Significant price decline ({returns_3m:.1f}% in 3 months)")

        return {
            "symbol": symbol.upper(),
            "decision": decision,
            "confidence": confidence,
            "score": round(score, 3),
            "reasons": reasons[:5],  # 상위 5개
            "risks": risks,
            "analysis_date": datetime.now().isoformat(),
            "disclaimer": "This is not financial advice. Always do your own research."
        }


# 싱글톤 인스턴스
data_integrator = DataIntegrator()


# 편의 함수들
def get_stock_analysis(symbol: str) -> Dict[str, Any]:
    """종목 종합 분석"""
    return data_integrator.get_comprehensive_analysis(symbol)


def compare_stocks(symbols: List[str]) -> Dict[str, Any]:
    """종목 비교"""
    return data_integrator.compare_stocks(symbols)


def get_investment_signal(symbol: str) -> Dict[str, Any]:
    """투자 신호"""
    return data_integrator.get_investment_signals(symbol)
