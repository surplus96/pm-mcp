"""감성 분석 팩터 계산 모듈 (10개)

Phase 2-2: 감성 분석 확장
- 뉴스 감성 분석 (3개): News Sentiment, News Volume, News Sentiment Std
- 공시 텍스트 분석 (2개): Filing Sentiment, Filing Frequency
- 시장 심리 지표 (3개): Put/Call Ratio, Market VIX, Short Interest Ratio
- 전문가 의견 집계 (2개): Analyst Rating, Target Price Upside
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import logging
import feedparser
import yfinance as yf
from .yf_utils import normalize_ticker_multi_market, is_yfinance_supported

logger = logging.getLogger(__name__)

# VADER Sentiment Analyzer (optional)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available, news sentiment will be disabled")


class SentimentFactors:
    """감성 분석 팩터 계산 클래스 (10개)"""

    # ============================================================
    # 그룹 1: 뉴스 감성 분석 (3개)
    # ============================================================
    @staticmethod
    def analyze_news_sentiment(ticker: str, days: int = 7, market: str = "US") -> Dict[str, float]:
        """뉴스 감성 분석

        Args:
            ticker: 종목 코드
            days: 분석 기간 (일)

        Returns:
            {
                'News_Sentiment': float,      # 평균 감성 점수 (-1 ~ 1)
                'News_Volume': float,          # 뉴스 개수
                'News_Sentiment_Std': float    # 감성 표준편차 (변동성)
            }
        """
        if not VADER_AVAILABLE:
            logger.warning("VADER not available, skipping news sentiment analysis")
            return {}

        try:
            analyzer = SentimentIntensityAnalyzer()

            # Google News RSS Feed — locale-aware for KR vs US (FR-K12).
            # The previous URL was built with an unescaped space ("ticker stock")
            # which caused ``URL can't contain control characters`` from
            # feedparser. We now url-encode the query and switch to Korean
            # locale + Korean company name when the ticker is KR so the
            # match rate is meaningful instead of returning empty.
            import urllib.parse as _urlparse
            from mcp_server.tools.yf_utils import detect_market

            is_kr = (market or "").upper() == "KR" or detect_market(ticker) == "KR"
            if is_kr:
                # Resolve Korean company name when possible — much better
                # recall than searching the 6-digit code alone.
                query_text = ticker
                try:
                    from mcp_server.tools.kr_market_data import get_kr_adapter
                    nm = get_kr_adapter().get_ticker_name(ticker) or ""
                    if nm:
                        query_text = nm
                except Exception:  # noqa: BLE001
                    pass
                q = _urlparse.quote(query_text)
                news_url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
            else:
                q = _urlparse.quote(f"{ticker} stock")
                news_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

            try:
                feed = feedparser.parse(news_url)
            except Exception as e:
                logger.warning(f"Failed to fetch news for {ticker}: {e}")
                return {}

            if not feed.entries:
                logger.debug(f"No news found for {ticker}")
                return {}

            sentiments = []
            for entry in feed.entries[:20]:  # 최근 20개 기사
                # 제목 + 요약 결합
                text = entry.title
                if hasattr(entry, 'summary'):
                    text += " " + entry.summary

                # VADER 감성 분석
                scores = analyzer.polarity_scores(text)
                sentiments.append(scores['compound'])

            if not sentiments:
                return {}

            return {
                'News_Sentiment': float(np.mean(sentiments)),
                'News_Volume': float(len(feed.entries)),
                'News_Sentiment_Std': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
            }

        except Exception as e:
            logger.warning(f"News sentiment analysis failed for {ticker}: {e}")
            return {}

    # ============================================================
    # 그룹 2: 공시 텍스트 분석 (2개) - 간소화 버전
    # ============================================================
    @staticmethod
    def analyze_filings(ticker: str, market: str = "US") -> Dict[str, float]:
        """공시 텍스트 감성 분석 (간소화)

        Note: SEC EDGAR API는 별도 구현 필요.
              현재는 yfinance 기본 정보로 대체

        Returns:
            {
                'Filing_Sentiment': float,     # 공시 감성 평균
                'Filing_Frequency': float      # 공시 빈도
            }
        """
        # Week 3에서는 간소화 버전으로 구현
        # 실제 공시 파싱은 복잡하므로 추후 개선
        try:
            normalized_ticker = normalize_ticker_multi_market(ticker, market)
            if not is_yfinance_supported(ticker, market):
                return {}
            stock = yf.Ticker(normalized_ticker)
            info = stock.info

            # 간단한 근사치: 회사 상태 지표로 감성 추정
            # 실제로는 SEC EDGAR 또는 DART에서 공시 텍스트를 파싱해야 함

            # Filing Frequency: 최근 분기 실적 발표 여부로 근사
            # (실제 구현 시 SEC EDGAR API 사용)
            filing_frequency = 4.0  # 분기별 기본값

            # Filing Sentiment: 간단히 0으로 설정 (중립)
            # (실제 구현 시 NLP 분석 필요)
            filing_sentiment = 0.0

            return {
                'Filing_Sentiment': filing_sentiment,
                'Filing_Frequency': filing_frequency
            }

        except Exception as e:
            logger.warning(f"Filing analysis failed for {ticker}: {e}")
            return {}

    # ============================================================
    # 그룹 3: 시장 심리 지표 (3개)
    # ============================================================
    @staticmethod
    def calculate_market_sentiment(ticker: str, market: str = "US") -> Dict[str, float]:
        """시장 심리 지표

        Returns:
            {
                'Put_Call_Ratio': float,       # Put/Call 비율
                'Market_VIX': float,           # VIX 지수
                'Short_Interest_Ratio': float  # 공매도 비율
            }
        """
        try:
            normalized_ticker = normalize_ticker_multi_market(ticker, market)
            if not is_yfinance_supported(ticker, market):
                return {}
            stock = yf.Ticker(normalized_ticker)
            info = stock.info

            # Put/Call Ratio (옵션 데이터)
            try:
                options = stock.options
                if options and len(options) > 0:
                    # 가장 가까운 만기일의 옵션 체인
                    opt_chain = stock.option_chain(options[0])
                    put_vol = opt_chain.puts['volume'].sum()
                    call_vol = opt_chain.calls['volume'].sum()

                    if call_vol > 0:
                        put_call_ratio = put_vol / call_vol
                    else:
                        put_call_ratio = np.nan
                else:
                    put_call_ratio = np.nan
            except Exception as e:
                logger.debug(f"Put/Call ratio calculation failed: {e}")
                put_call_ratio = np.nan

            # VIX Level (시장 공포 지수)
            try:
                vix = yf.Ticker("^VIX").history(period="1d")
                if not vix.empty:
                    market_vix = float(vix['Close'].iloc[-1])
                else:
                    market_vix = np.nan
            except Exception as e:
                logger.debug(f"VIX fetch failed: {e}")
                market_vix = np.nan

            # Short Interest Ratio
            short_ratio = info.get('shortRatio', np.nan)
            short_percent = info.get('shortPercentOfFloat', np.nan)

            # shortRatio가 없으면 shortPercentOfFloat 사용
            if pd.isna(short_ratio) and not pd.isna(short_percent):
                short_ratio = short_percent * 100  # %로 변환

            return {
                'Put_Call_Ratio': put_call_ratio,
                'Market_VIX': market_vix,
                'Short_Interest_Ratio': short_ratio
            }

        except Exception as e:
            logger.warning(f"Market sentiment calculation failed for {ticker}: {e}")
            return {}

    # ============================================================
    # 그룹 4: 전문가 의견 집계 (2개)
    # ============================================================
    @staticmethod
    def analyze_analyst_opinion(ticker: str, market: str = "US") -> Dict[str, float]:
        """애널리스트 의견 집계

        Returns:
            {
                'Analyst_Rating': float,       # 평균 추천 등급 (1-5, 높을수록 긍정)
                'Target_Price_Upside': float   # 목표가 상승 여력 (%)
            }
        """
        try:
            normalized_ticker = normalize_ticker_multi_market(ticker, market)
            if not is_yfinance_supported(ticker, market):
                return {}
            stock = yf.Ticker(normalized_ticker)
            info = stock.info

            # Analyst Recommendation (1=Strong Buy, 5=Strong Sell)
            recommendation = info.get('recommendationMean', np.nan)

            # 역변환: 높을수록 긍정적으로
            if not pd.isna(recommendation):
                analyst_rating = 6 - recommendation  # 5 → 1, 1 → 5
            else:
                analyst_rating = np.nan

            # Target Price Upside
            target_price = info.get('targetMeanPrice', np.nan)
            current_price = info.get('currentPrice', np.nan)

            if not pd.isna(target_price) and not pd.isna(current_price) and current_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
            else:
                upside = np.nan

            return {
                'Analyst_Rating': analyst_rating,
                'Target_Price_Upside': upside
            }

        except Exception as e:
            logger.warning(f"Analyst opinion analysis failed for {ticker}: {e}")
            return {}

    # ============================================================
    # 통합 함수
    # ============================================================
    @staticmethod
    def calculate_all(ticker: str, market: str = "US", days: int = 7) -> Dict[str, float]:
        """10개 감성 팩터 통합 계산

        Args:
            ticker: 종목 코드
            market: 시장 구분 ("US", "KR")
            days: 뉴스 분석 기간 (일)

        Returns:
            감성 팩터 딕셔너리 (NaN 값 제거)
        """
        factors = {}

        try:
            # 1. 뉴스 감성 (3개) — market 전달해 KR 티커는 한국어 RSS 사용
            news = SentimentFactors.analyze_news_sentiment(ticker, days, market=market)
            factors.update(news)

            # 2. 공시 분석 (2개)
            filings = SentimentFactors.analyze_filings(ticker, market)
            factors.update(filings)

            # 3. 시장 심리 (3개)
            market_sent = SentimentFactors.calculate_market_sentiment(ticker, market)
            factors.update(market_sent)

            # 4. 전문가 의견 (2개)
            analyst = SentimentFactors.analyze_analyst_opinion(ticker, market)
            factors.update(analyst)

        except Exception as e:
            logger.error(f"Failed to calculate sentiment factors for {ticker}: {e}")

        # NaN 값 제거
        factors = {k: v for k, v in factors.items() if not pd.isna(v)}

        logger.info(f"Calculated {len(factors)}/10 sentiment factors for {ticker}")
        return factors

    @staticmethod
    def get_factor_interpretation(factors: Dict[str, float]) -> Dict[str, str]:
        """감성 팩터 해석

        Args:
            factors: 계산된 감성 팩터

        Returns:
            팩터별 해석 딕셔너리
        """
        interpretation = {}

        # News Sentiment 해석
        if 'News_Sentiment' in factors:
            sent = factors['News_Sentiment']
            if sent > 0.2:
                interpretation['News_Sentiment'] = f"긍정적 뉴스 우세 (Bullish, {sent:.2f})"
            elif sent < -0.2:
                interpretation['News_Sentiment'] = f"부정적 뉴스 우세 (Bearish, {sent:.2f})"
            else:
                interpretation['News_Sentiment'] = f"중립적 뉴스 (Neutral, {sent:.2f})"

        # News Volume 해석
        if 'News_Volume' in factors:
            vol = factors['News_Volume']
            if vol > 50:
                interpretation['News_Volume'] = f"매우 높은 언론 관심 ({vol:.0f}건)"
            elif vol > 20:
                interpretation['News_Volume'] = f"높은 언론 관심 ({vol:.0f}건)"
            elif vol > 10:
                interpretation['News_Volume'] = f"보통 언론 관심 ({vol:.0f}건)"
            else:
                interpretation['News_Volume'] = f"낮은 언론 관심 ({vol:.0f}건)"

        # Put/Call Ratio 해석
        if 'Put_Call_Ratio' in factors:
            pcr = factors['Put_Call_Ratio']
            if pcr > 1.0:
                interpretation['Put_Call_Ratio'] = f"약세 심리 (Bearish, {pcr:.2f})"
            elif pcr < 0.7:
                interpretation['Put_Call_Ratio'] = f"강세 심리 (Bullish, {pcr:.2f})"
            else:
                interpretation['Put_Call_Ratio'] = f"중립 심리 (Neutral, {pcr:.2f})"

        # Market VIX 해석
        if 'Market_VIX' in factors:
            vix = factors['Market_VIX']
            if vix > 30:
                interpretation['Market_VIX'] = f"높은 시장 공포 (High Fear, {vix:.1f})"
            elif vix > 20:
                interpretation['Market_VIX'] = f"보통 시장 불안 (Moderate Fear, {vix:.1f})"
            else:
                interpretation['Market_VIX'] = f"낮은 시장 공포 (Low Fear, {vix:.1f})"

        # Short Interest 해석
        if 'Short_Interest_Ratio' in factors:
            sir = factors['Short_Interest_Ratio']
            if sir > 10:
                interpretation['Short_Interest_Ratio'] = f"매우 높은 공매도 ({sir:.1f}%)"
            elif sir > 5:
                interpretation['Short_Interest_Ratio'] = f"높은 공매도 ({sir:.1f}%)"
            elif sir > 2:
                interpretation['Short_Interest_Ratio'] = f"보통 공매도 ({sir:.1f}%)"
            else:
                interpretation['Short_Interest_Ratio'] = f"낮은 공매도 ({sir:.1f}%)"

        # Analyst Rating 해석
        if 'Analyst_Rating' in factors:
            rating = factors['Analyst_Rating']
            if rating > 4.5:
                interpretation['Analyst_Rating'] = f"강력 매수 (Strong Buy, {rating:.2f})"
            elif rating > 3.5:
                interpretation['Analyst_Rating'] = f"매수 (Buy, {rating:.2f})"
            elif rating > 2.5:
                interpretation['Analyst_Rating'] = f"보유 (Hold, {rating:.2f})"
            elif rating > 1.5:
                interpretation['Analyst_Rating'] = f"매도 (Sell, {rating:.2f})"
            else:
                interpretation['Analyst_Rating'] = f"강력 매도 (Strong Sell, {rating:.2f})"

        # Target Price Upside 해석
        if 'Target_Price_Upside' in factors:
            upside = factors['Target_Price_Upside']
            if upside > 20:
                interpretation['Target_Price_Upside'] = f"높은 상승 여력 (+{upside:.1f}%)"
            elif upside > 10:
                interpretation['Target_Price_Upside'] = f"적정 상승 여력 (+{upside:.1f}%)"
            elif upside > 0:
                interpretation['Target_Price_Upside'] = f"제한적 상승 여력 (+{upside:.1f}%)"
            else:
                interpretation['Target_Price_Upside'] = f"하락 여력 ({upside:.1f}%)"

        return interpretation


# 편의 함수
def calculate_sentiment_score(factors: Dict[str, float]) -> float:
    """감성 팩터 종합 스코어 계산

    Args:
        factors: 감성 팩터 딕셔너리

    Returns:
        0-100 사이의 종합 스코어
    """
    if not factors:
        return 50.0  # 중립

    # 정규화 및 스코어 계산
    # (간단한 구현 - 추후 개선 가능)

    scores = []

    # News Sentiment (-1 ~ 1) → 0 ~ 100
    if 'News_Sentiment' in factors:
        ns = factors['News_Sentiment']
        scores.append((ns + 1) * 50)  # -1 → 0, 0 → 50, 1 → 100

    # Put/Call Ratio (낮을수록 좋음)
    if 'Put_Call_Ratio' in factors:
        pcr = factors['Put_Call_Ratio']
        # 0.5 → 100, 1.0 → 50, 1.5 → 0
        score = max(0, min(100, (1.5 - pcr) * 100))
        scores.append(score)

    # Analyst Rating (1-5, 높을수록 좋음)
    if 'Analyst_Rating' in factors:
        rating = factors['Analyst_Rating']
        scores.append((rating - 1) * 25)  # 1 → 0, 5 → 100

    # Target Price Upside
    if 'Target_Price_Upside' in factors:
        upside = factors['Target_Price_Upside']
        # -20% → 0, 0% → 50, +20% → 100
        score = max(0, min(100, (upside + 20) * 2.5))
        scores.append(score)

    if scores:
        return float(np.mean(scores))
    else:
        return 50.0
