#!/usr/bin/env python3
"""
Theme Factor Integrator Module

테마 발굴과 팩터 분석을 통합하여 원스톱 투자 분석 제공
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
import time

from mcp_server.tools.interaction import propose_tickers
from mcp_server.tools.factor_aggregator import FactorAggregator
from mcp_server.tools.backtest_engine import BacktestEngine
from mcp_server.tools.news_search import search_news
from mcp_server.tools.cache_manager import cache_manager

logger = logging.getLogger(__name__)

THEME_SENTIMENT_TTL = 30 * 60  # 30분


class ThemeFactorIntegrator:
    """테마 + 팩터 통합 분석"""

    # 테마별 기본 설정
    THEME_CONFIG = {
        "AI": {
            "etfs": ["BOTZ", "AIQ"],
            "min_market_cap": 10_000_000_000,  # $10B
            "max_candidates": 15
        },
        "semiconductor": {
            "etfs": ["SMH", "SOXX"],
            "min_market_cap": 5_000_000_000,  # $5B
            "max_candidates": 12
        },
        "cloud": {
            "etfs": ["CLOU", "WCLD"],
            "min_market_cap": 5_000_000_000,
            "max_candidates": 12
        },
        "cybersecurity": {
            "etfs": ["HACK", "CIBR"],
            "min_market_cap": 3_000_000_000,
            "max_candidates": 10
        },
        "biotech": {
            "etfs": ["XBI", "IBB"],
            "min_market_cap": 2_000_000_000,
            "max_candidates": 12
        },
        "default": {
            "etfs": [],
            "min_market_cap": 1_000_000_000,  # $1B
            "max_candidates": 10
        }
    }

    @staticmethod
    def analyze_theme(
        theme: str,
        top_n: int = 5,
        include_backtest: bool = False,
        include_sentiment: bool = True,
        rerank_by_backtest: bool = False,
        market: str = "US",
        backtest_start: str = "2024-01-01",
        backtest_end: str = "2024-12-31",
        factor_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """테마 기반 종합 분석 (Response Time Benchmarking 포함)

        Args:
            theme: 테마 키워드 (예: "AI", "semiconductor")
            top_n: 상위 N개 종목 반환
            include_backtest: 백테스트 포함 여부
            include_sentiment: 테마 감성 분석 포함 여부
            rerank_by_backtest: 백테스트 성과 기반 재정렬 (Week 2)
            market: 시장 (US/KR)
            backtest_start: 백테스트 시작일
            backtest_end: 백테스트 종료일
            factor_weights: 팩터 가중치 (optional)

        Returns:
            테마 분석 결과 (performance_metrics 포함)
        """
        # Performance benchmarking start
        start_time = time.time()
        stage_timings = {}

        try:
            logger.info(f"Analyzing theme: {theme}, top_n={top_n}, backtest={include_backtest}")

            # Step 1: 테마 종목 발굴
            step1_start = time.time()
            tickers = propose_tickers(theme)
            stage_timings['ticker_discovery'] = round(time.time() - step1_start, 3)

            if not tickers:
                logger.warning(f"No tickers found for theme: {theme}")
                return {
                    'error': f'No tickers found for theme: {theme}',
                    'theme': theme,
                    'suggestion': 'Try alternative themes like: AI, semiconductor, cloud, biotech'
                }

            logger.info(f"Found {len(tickers)} candidates for theme '{theme}'")

            # Step 2: 팩터 기반 랭킹
            step2_start = time.time()
            ranked_stocks = ThemeFactorIntegrator.rank_theme_stocks(
                tickers=tickers,
                market=market,
                factor_weights=factor_weights
            )
            stage_timings['factor_ranking'] = round(time.time() - step2_start, 3)

            if not ranked_stocks:
                logger.warning(f"No valid stocks after factor analysis for theme: {theme}")
                return {
                    'error': f'No valid stocks after factor analysis',
                    'theme': theme,
                    'candidates': len(tickers)
                }

            # Step 3: 상위 N개 선택
            top_stocks = ranked_stocks[:top_n]

            logger.info(f"Selected top {len(top_stocks)} stocks from {len(ranked_stocks)} analyzed")

            # Step 4: 백테스트 추가 (optional)
            if include_backtest:
                step4_start = time.time()
                logger.info(f"Running backtest for top {len(top_stocks)} stocks")
                top_stocks = ThemeFactorIntegrator.enrich_with_backtest(
                    stocks=top_stocks,
                    start_date=backtest_start,
                    end_date=backtest_end,
                    market=market
                )
                stage_timings['backtest'] = round(time.time() - step4_start, 3)

                # Step 4.5: 백테스트 품질 검증 및 성과 기반 재정렬 (Week 2)
                if rerank_by_backtest and any('backtest' in s and s['backtest'].get('total_return') is not None for s in top_stocks):
                    step45_start = time.time()
                    logger.info("Reranking by backtest performance")
                    top_stocks = ThemeFactorIntegrator.rerank_by_performance(
                        stocks=top_stocks,
                        factor_weight=0.6,
                        backtest_weight=0.4
                    )
                    stage_timings['rerank'] = round(time.time() - step45_start, 3)
                    # 상위 N개 재선택
                    top_stocks = top_stocks[:top_n]

            # Step 5: 테마 감성 분석 (optional)
            theme_sentiment = None
            if include_sentiment:
                step5_start = time.time()
                try:
                    theme_sentiment = ThemeFactorIntegrator.get_theme_sentiment(theme)
                    logger.info(f"Theme sentiment: {theme_sentiment.get('sentiment_label', 'Unknown')}")
                    stage_timings['sentiment'] = round(time.time() - step5_start, 3)
                except Exception as e:
                    logger.warning(f"Theme sentiment analysis failed: {e}")
                    stage_timings['sentiment'] = round(time.time() - step5_start, 3)

            # Step 6: 추천 생성
            step6_start = time.time()
            recommendation = ThemeFactorIntegrator.generate_recommendation(
                theme=theme,
                top_stocks=top_stocks,
                theme_sentiment=theme_sentiment
            )
            stage_timings['recommendation'] = round(time.time() - step6_start, 3)

            # Calculate total time
            total_time = round(time.time() - start_time, 3)

            # Step 7: 결과 반환
            return {
                'theme': theme,
                'market': market,
                'total_candidates': len(tickers),
                'analyzed_stocks': len(ranked_stocks),
                'top_n': top_n,
                'top_stocks': top_stocks,
                'theme_sentiment': theme_sentiment,
                'recommendation': recommendation,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'performance_metrics': {
                    'total_time_seconds': total_time,
                    'stage_timings': stage_timings
                }
            }

        except Exception as e:
            logger.error(f"Theme analysis failed: {e}")
            return {
                'error': str(e),
                'theme': theme
            }

    @staticmethod
    def get_theme_sentiment(theme: str, lookback_days: int = 7) -> Dict:
        """테마 감성 분석 (Week 3 Enhanced, Week 4 Cached)

        Args:
            theme: 테마 키워드
            lookback_days: 분석 기간 (일)

        Returns:
            테마 감성 정보 (모멘텀, 변동성, 키워드 포함)
        """
        # Week 4: 캐시 확인
        cache_key = f"theme_sentiment:{theme.lower()}:{lookback_days}"

        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            logger.info(f"Theme sentiment cache hit: {theme}")
            return cached_result

        try:
            # Step 1: 테마 관련 뉴스 수집
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)

            news_articles = search_news(
                keywords=[theme],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if not news_articles:
                logger.info(f"No news articles found for theme: {theme}")
                return {
                    'sentiment_score': 0.0,
                    'sentiment_label': 'Neutral',
                    'sentiment_std': 0.0,
                    'news_volume': 0,
                    'trending': False,
                    'momentum': 'Stable',
                    'momentum_score': 0.0,
                    'confidence': 'Low',
                    'key_topics': [],
                    'lookback_days': lookback_days
                }

            # Step 2: 감성 분석
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("vaderSentiment not installed, using fallback")
                return {
                    'sentiment_score': 0.0,
                    'sentiment_label': 'Neutral',
                    'sentiment_std': 0.0,
                    'news_volume': len(news_articles),
                    'trending': False,
                    'momentum': 'Unknown',
                    'confidence': 'Low',
                    'lookback_days': lookback_days,
                    'note': 'Sentiment analysis unavailable (vaderSentiment not installed)'
                }

            # Step 3: 시간별 감성 분석 (모멘텀 계산용)
            sentiments_with_time = []
            all_texts = []

            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    scores = analyzer.polarity_scores(text)
                    pub_date = article.get('published_date', end_date)
                    sentiments_with_time.append({
                        'score': scores['compound'],
                        'date': pub_date,
                        'text': text
                    })
                    all_texts.append(text.lower())

            if not sentiments_with_time:
                return {
                    'sentiment_score': 0.0,
                    'sentiment_label': 'Neutral',
                    'news_volume': 0,
                    'trending': False,
                    'momentum': 'Stable',
                    'confidence': 'Low'
                }

            # Step 4: 전체 통계 계산
            all_scores = [s['score'] for s in sentiments_with_time]
            avg_sentiment = np.mean(all_scores)
            sentiment_std = np.std(all_scores) if len(all_scores) > 1 else 0.0
            news_volume = len(news_articles)

            # Step 5: 감성 모멘텀 계산 (최근 vs 과거)
            # 최근 30% vs 과거 70% 비교
            split_point = max(1, int(len(sentiments_with_time) * 0.3))
            sorted_by_date = sorted(sentiments_with_time, key=lambda x: x['date'], reverse=True)

            recent_scores = [s['score'] for s in sorted_by_date[:split_point]]
            earlier_scores = [s['score'] for s in sorted_by_date[split_point:]]

            recent_avg = np.mean(recent_scores) if recent_scores else avg_sentiment
            earlier_avg = np.mean(earlier_scores) if earlier_scores else avg_sentiment
            momentum_score = recent_avg - earlier_avg

            # 모멘텀 레이블
            if momentum_score > 0.15:
                momentum = 'Strong Positive'
            elif momentum_score > 0.05:
                momentum = 'Positive'
            elif momentum_score < -0.15:
                momentum = 'Strong Negative'
            elif momentum_score < -0.05:
                momentum = 'Negative'
            else:
                momentum = 'Stable'

            # Step 6: 신뢰도 평가 (뉴스량 + 변동성)
            # 뉴스가 많고 변동성이 낮을수록 신뢰도 높음
            volume_score = min(news_volume / 20, 1.0)  # 20개 이상이면 만점
            consistency_score = max(0, 1 - sentiment_std)  # 변동성 낮을수록 높음
            confidence_score = (volume_score * 0.6) + (consistency_score * 0.4)

            if confidence_score > 0.7:
                confidence = 'High'
            elif confidence_score > 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'

            # Step 7: 주요 키워드 추출 (간단한 빈도 기반)
            from collections import Counter
            import re

            # 불용어 제거 및 단어 추출
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
            words = []
            for text in all_texts:
                # 간단한 단어 추출 (2-15자 단어만)
                text_words = re.findall(r'\b[a-z]{2,15}\b', text)
                words.extend([w for w in text_words if w not in stop_words])

            # 빈도 상위 5개
            word_freq = Counter(words)
            key_topics = [word for word, count in word_freq.most_common(5) if count > 1]

            # Step 8: 트렌드 판단 (기존 + 모멘텀)
            # 뉴스량이 많고 (>10) 긍정적이고 (>0.2) 모멘텀이 양수면 트렌딩
            trending = news_volume > 10 and avg_sentiment > 0.2 and momentum_score > 0

            # Step 9: 레이블 결정
            if avg_sentiment > 0.3:
                label = 'Bullish'
            elif avg_sentiment > 0.1:
                label = 'Slightly Bullish'
            elif avg_sentiment > -0.1:
                label = 'Neutral'
            elif avg_sentiment > -0.3:
                label = 'Slightly Bearish'
            else:
                label = 'Bearish'

            result = {
                'sentiment_score': round(float(avg_sentiment), 3),
                'sentiment_label': label,
                'sentiment_std': round(float(sentiment_std), 3),
                'news_volume': news_volume,
                'trending': trending,
                'momentum': momentum,
                'momentum_score': round(float(momentum_score), 3),
                'confidence': confidence,
                'confidence_score': round(float(confidence_score), 3),
                'key_topics': key_topics[:5],
                'lookback_days': lookback_days
            }

            # Week 4: 결과 캐싱
            cache_manager.set(cache_key, result, ttl=THEME_SENTIMENT_TTL)
            logger.info(f"Theme sentiment cached: {theme}")

            return result

        except Exception as e:
            logger.error(f"Theme sentiment analysis failed: {e}")
            # 에러는 캐싱하지 않음
            return {
                'error': str(e),
                'sentiment_score': 0.0,
                'sentiment_label': 'Unknown',
                'news_volume': 0,
                'trending': False,
                'momentum': 'Unknown',
                'confidence': 'Low'
            }

    @staticmethod
    def rank_theme_stocks(
        tickers: List[str],
        market: str = "US",
        factor_weights: Optional[Dict[str, float]] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> List[Dict]:
        """테마 종목 랭킹 (팩터 기반, Retry with Exponential Backoff)

        Args:
            tickers: 종목 리스트
            market: 시장
            factor_weights: 팩터 가중치
            max_retries: 최대 재시도 횟수 (기본: 3)
            initial_delay: 초기 지연 시간 (초, 기본: 1.0)

        Returns:
            랭킹된 종목 리스트
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Ranking {len(tickers)} stocks with factor analysis (attempt {attempt + 1}/{max_retries})")

                # Step 1: 팩터 기반 랭킹 (기존 함수 재사용)
                ranked = FactorAggregator.rank_stocks(
                    tickers=tickers,
                    market=market,
                    include_technical=True,
                    include_financial=True,
                    include_sentiment=True,
                    factor_weights=factor_weights
                )

                # Step 2: 에러 필터링
                valid_stocks = [s for s in ranked if 'error' not in s]

                logger.info(f"Successfully analyzed {len(valid_stocks)} out of {len(tickers)} stocks")

                # Step 3: 추천 등급 추가
                for stock in valid_stocks:
                    score = stock['composite_score']
                    stock['recommendation'] = FactorAggregator.get_recommendation(score)

                return valid_stocks

            except Exception as e:
                last_exception = e

                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Ranking failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Ranking theme stocks failed after {max_retries} attempts: {e}")

        # All retries exhausted
        logger.error(f"Final failure: {last_exception}")
        return []

    @staticmethod
    def enrich_with_backtest(
        stocks: List[Dict],
        start_date: str,
        end_date: str,
        market: str = "US"
    ) -> List[Dict]:
        """백테스트 정보 추가

        Args:
            stocks: 종목 리스트
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            market: 시장

        Returns:
            백테스트 정보가 추가된 종목 리스트
        """
        enriched = []

        for stock in stocks:
            ticker = stock['ticker']

            try:
                # Step 1: 백테스트 실행
                backtest_result = BacktestEngine.run_backtest(
                    ticker=ticker,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_period=30,
                    buy_threshold=60.0,
                    sell_threshold=40.0,
                    initial_capital=10000.0
                )

                # Step 2: 주요 지표 추출
                perf = backtest_result.get('performance', {})
                stock['backtest'] = {
                    'total_return': backtest_result.get('total_return', 0.0),
                    'cagr': perf.get('CAGR', 0.0),
                    'max_drawdown': perf.get('Max_Drawdown', 0.0),
                    'sharpe_ratio': perf.get('Sharpe_Ratio', 0.0),
                    'win_rate': perf.get('Win_Rate', 0.0),
                    'trade_count': backtest_result.get('trade_count', 0)
                }

                logger.info(f"Backtest completed for {ticker}: {stock['backtest']['total_return']:.2f}%")

            except Exception as e:
                logger.warning(f"Backtest failed for {ticker}: {e}")
                stock['backtest'] = {
                    'error': str(e),
                    'total_return': None
                }

            enriched.append(stock)

        return enriched

    @staticmethod
    def rerank_by_performance(
        stocks: List[Dict],
        factor_weight: float = 0.6,
        backtest_weight: float = 0.4
    ) -> List[Dict]:
        """성과 기반 재정렬 (Week 2)

        팩터 점수와 백테스트 성과를 결합하여 재정렬

        Args:
            stocks: 종목 리스트 (백테스트 정보 포함)
            factor_weight: 팩터 점수 가중치 (기본: 0.6)
            backtest_weight: 백테스트 성과 가중치 (기본: 0.4)

        Returns:
            재정렬된 종목 리스트
        """
        try:
            # 백테스트 데이터가 있는 종목만 재정렬
            stocks_with_backtest = [
                s for s in stocks
                if 'backtest' in s and s['backtest'].get('total_return') is not None
            ]

            if not stocks_with_backtest:
                logger.info("No valid backtest data, skipping reranking")
                return stocks

            # 백테스트 수익률 정규화 (0-100 스케일)
            returns = [s['backtest']['total_return'] for s in stocks_with_backtest]
            min_return = min(returns)
            max_return = max(returns)

            for stock in stocks_with_backtest:
                # 팩터 점수 (이미 0-100)
                factor_score = stock.get('composite_score', 50.0)

                # 백테스트 수익률 정규화
                bt_return = stock['backtest']['total_return']
                if max_return > min_return:
                    bt_score = ((bt_return - min_return) / (max_return - min_return)) * 100
                else:
                    bt_score = 50.0

                # 복합 점수 계산
                combined_score = (factor_score * factor_weight) + (bt_score * backtest_weight)

                stock['combined_score'] = round(combined_score, 2)
                stock['original_rank'] = stock.get('rank', 0)

                logger.debug(f"{stock['ticker']}: factor={factor_score:.1f}, bt={bt_score:.1f}, combined={combined_score:.1f}")

            # 복합 점수 기준 재정렬
            stocks_with_backtest.sort(key=lambda x: x['combined_score'], reverse=True)

            # 랭킹 업데이트
            for i, stock in enumerate(stocks_with_backtest, 1):
                stock['rank'] = i

            # 백테스트 없는 종목은 뒤에 추가
            stocks_without_backtest = [
                s for s in stocks
                if s not in stocks_with_backtest
            ]

            return stocks_with_backtest + stocks_without_backtest

        except Exception as e:
            logger.error(f"Reranking by performance failed: {e}")
            return stocks

    @staticmethod
    def validate_backtest_quality(
        backtest_result: Dict
    ) -> Dict:
        """백테스트 품질 검증 (Week 2)

        백테스트 결과의 신뢰도 평가

        Args:
            backtest_result: 백테스트 결과

        Returns:
            품질 평가 결과
        """
        try:
            quality_score = 100.0
            issues = []
            warnings = []

            # 1. 거래 횟수 확인
            trade_count = backtest_result.get('trade_count', 0)
            if trade_count == 0:
                quality_score -= 50
                issues.append("No trades executed")
            elif trade_count < 3:
                quality_score -= 20
                warnings.append(f"Low trade count ({trade_count})")

            # 2. 샤프 비율 확인
            perf = backtest_result.get('performance', {})
            sharpe = perf.get('Sharpe_Ratio', 0)
            if sharpe < 0:
                quality_score -= 15
                warnings.append(f"Negative Sharpe ratio ({sharpe:.2f})")
            elif sharpe < 0.5:
                quality_score -= 5
                warnings.append(f"Low Sharpe ratio ({sharpe:.2f})")

            # 3. 최대 낙폭 확인
            max_dd = perf.get('Max_Drawdown', 0)
            if max_dd > 50:
                quality_score -= 20
                issues.append(f"Excessive drawdown ({max_dd:.1f}%)")
            elif max_dd > 30:
                quality_score -= 10
                warnings.append(f"High drawdown ({max_dd:.1f}%)")

            # 4. 승률 확인
            win_rate = perf.get('Win_Rate', 0)
            if win_rate < 30:
                quality_score -= 10
                warnings.append(f"Low win rate ({win_rate:.1f}%)")

            # 5. 품질 등급
            if quality_score >= 80:
                grade = "Excellent"
            elif quality_score >= 60:
                grade = "Good"
            elif quality_score >= 40:
                grade = "Fair"
            else:
                grade = "Poor"

            return {
                'quality_score': max(0, quality_score),
                'grade': grade,
                'issues': issues,
                'warnings': warnings,
                'reliable': quality_score >= 60
            }

        except Exception as e:
            logger.error(f"Backtest quality validation failed: {e}")
            return {
                'quality_score': 0,
                'grade': 'Unknown',
                'issues': [str(e)],
                'warnings': [],
                'reliable': False
            }

    @staticmethod
    def generate_recommendation(
        theme: str,
        top_stocks: List[Dict],
        theme_sentiment: Optional[Dict] = None
    ) -> Dict:
        """투자 추천 생성 (Week 3 Enhanced)

        Args:
            theme: 테마
            top_stocks: 상위 종목 리스트
            theme_sentiment: 테마 감성 정보

        Returns:
            투자 추천 (텍스트 + 메타데이터)
        """
        if not top_stocks:
            return {
                'summary': f"No valid stocks found for theme '{theme}'",
                'action': 'AVOID',
                'confidence': 'Low',
                'risk_level': 'N/A',
                'signals': {}
            }

        # Step 1: 신호 수집 및 점수화
        signals = {}

        # 1.1 팩터 점수
        avg_score = np.mean([s['composite_score'] for s in top_stocks])
        signals['factor_score'] = round(float(avg_score), 1)

        if avg_score >= 70:
            signals['factor_signal'] = 'Strong'
            factor_points = 3
        elif avg_score >= 60:
            signals['factor_signal'] = 'Moderate'
            factor_points = 2
        else:
            signals['factor_signal'] = 'Weak'
            factor_points = 1

        # 1.2 백테스트 수익률
        backtest_returns = [
            s['backtest']['total_return']
            for s in top_stocks
            if 'backtest' in s and s['backtest'].get('total_return') is not None
        ]
        avg_backtest = np.mean(backtest_returns) if backtest_returns else None

        if avg_backtest is not None:
            signals['backtest_return'] = round(float(avg_backtest), 1)
            if avg_backtest > 20:
                signals['backtest_signal'] = 'Strong'
                backtest_points = 3
            elif avg_backtest > 10:
                signals['backtest_signal'] = 'Good'
                backtest_points = 2
            elif avg_backtest > 0:
                signals['backtest_signal'] = 'Positive'
                backtest_points = 1
            else:
                signals['backtest_signal'] = 'Negative'
                backtest_points = 0
        else:
            backtest_points = 1  # 중립

        # 1.3 테마 감성 (Week 3 Enhanced)
        sentiment_points = 1  # 기본값
        if theme_sentiment and 'error' not in theme_sentiment:
            sentiment_score = theme_sentiment.get('sentiment_score', 0.0)
            sentiment_label = theme_sentiment.get('sentiment_label', 'Unknown')
            momentum = theme_sentiment.get('momentum', 'Stable')
            confidence = theme_sentiment.get('confidence', 'Low')
            trending = theme_sentiment.get('trending', False)

            signals['sentiment_score'] = sentiment_score
            signals['sentiment_label'] = sentiment_label
            signals['sentiment_momentum'] = momentum
            signals['sentiment_confidence'] = confidence
            signals['trending'] = trending

            # 감성 점수화
            if trending and sentiment_score > 0.2:
                signals['sentiment_signal'] = 'Very Bullish'
                sentiment_points = 3
            elif sentiment_score > 0.2:
                signals['sentiment_signal'] = 'Bullish'
                sentiment_points = 2
            elif sentiment_score > -0.1:
                signals['sentiment_signal'] = 'Neutral'
                sentiment_points = 1
            else:
                signals['sentiment_signal'] = 'Bearish'
                sentiment_points = 0

            # 모멘텀 보너스/페널티
            if 'Positive' in momentum:
                sentiment_points += 0.5
            elif 'Negative' in momentum:
                sentiment_points -= 0.5

        # Step 2: 종합 신호 점수 (0-9)
        total_points = factor_points + backtest_points + sentiment_points
        max_points = 9.0

        # Step 3: 리스크 평가
        risk_factors = []

        # 팩터 약세
        if avg_score < 60:
            risk_factors.append("Weak fundamentals")

        # 백테스트 부진
        if avg_backtest is not None and avg_backtest < 0:
            risk_factors.append("Negative historical performance")

        # 감성 변동성
        if theme_sentiment:
            sentiment_std = theme_sentiment.get('sentiment_std', 0)
            if sentiment_std > 0.3:
                risk_factors.append("High sentiment volatility")

            # 모멘텀 악화
            if 'Negative' in theme_sentiment.get('momentum', ''):
                risk_factors.append("Deteriorating sentiment")

        # 리스크 레벨
        if len(risk_factors) >= 3:
            risk_level = 'High'
        elif len(risk_factors) >= 1:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        signals['risk_factors'] = risk_factors
        signals['risk_level'] = risk_level

        # Step 4: 추천 액션 및 신뢰도
        confidence_score = total_points / max_points

        if confidence_score >= 0.75:
            confidence = 'High'
        elif confidence_score >= 0.5:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        # 액션 결정
        if total_points >= 7 and risk_level != 'High':
            action = 'BUY'
            top_tickers = ', '.join([s['ticker'] for s in top_stocks[:3]])
            action_detail = f"Consider accumulating top 3 stocks: {top_tickers}"
        elif total_points >= 5 and risk_level != 'High':
            action = 'WATCH'
            top_tickers = ', '.join([s['ticker'] for s in top_stocks[:2]])
            action_detail = f"Monitor top stocks: {top_tickers}"
        elif total_points >= 4:
            action = 'HOLD'
            action_detail = "Wait for clearer signals or better entry points"
        else:
            action = 'AVOID'
            action_detail = "Consider alternative themes or wait for improvement"

        # Step 5: 요약 생성
        summary_parts = []

        # 테마 트렌드
        if theme_sentiment and theme_sentiment.get('trending'):
            summary_parts.append(f"✅ '{theme}' theme is trending")

        # 주요 신호
        if signals.get('factor_signal') == 'Strong':
            summary_parts.append(f"Strong fundamentals ({avg_score:.1f}/100)")
        elif signals.get('factor_signal') == 'Weak':
            summary_parts.append(f"Weak fundamentals ({avg_score:.1f}/100)")

        if signals.get('backtest_signal') in ['Strong', 'Good']:
            summary_parts.append(f"Positive backtest ({avg_backtest:.1f}%)")

        if signals.get('sentiment_signal') in ['Very Bullish', 'Bullish']:
            summary_parts.append(f"{signals['sentiment_label']} sentiment")
        elif signals.get('sentiment_signal') == 'Bearish':
            summary_parts.append(f"⚠️ {signals['sentiment_label']} sentiment")

        # 리스크
        if risk_level == 'High':
            summary_parts.append(f"⚠️ High risk: {', '.join(risk_factors[:2])}")
        elif risk_level == 'Medium':
            summary_parts.append(f"Medium risk")

        summary = ' | '.join(summary_parts) if summary_parts else "Mixed signals"

        return {
            'summary': summary,
            'action': action,
            'action_detail': action_detail,
            'confidence': confidence,
            'confidence_score': round(float(confidence_score), 2),
            'risk_level': risk_level,
            'total_score': round(float(total_points), 1),
            'max_score': max_points,
            'signals': signals
        }
