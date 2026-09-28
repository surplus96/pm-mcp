#!/usr/bin/env python3
"""
Factor Aggregator Module

팩터 정규화, 가중치 기반 종합 점수 계산, 다종목 랭킹 기능 제공
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from mcp_server.tools.financial_factors import FinancialFactors
from mcp_server.tools.sentiment_analysis import SentimentFactors

logger = logging.getLogger(__name__)


class FactorAggregator:
    """팩터 통합 및 종합 점수 계산"""

    # 팩터별 정규화 범위 (최적값 기준)
    FACTOR_RANGES = {
        # 재무 팩터
        'ROE': (0, 0.3, 'higher'),  # (min, max, direction)
        'ROA': (0, 0.2, 'higher'),
        'ROIC': (0, 0.25, 'higher'),
        'Operating_Margin': (0, 0.4, 'higher'),
        'Net_Margin': (0, 0.3, 'higher'),
        'Debt_to_Equity': (0, 2.0, 'lower'),
        'Current_Ratio': (1.0, 3.0, 'higher'),
        'Quick_Ratio': (0.5, 2.5, 'higher'),
        'Interest_Coverage': (0, 50, 'higher'),
        'Debt_to_Asset': (0, 0.7, 'lower'),
        'Asset_Turnover': (0, 3.0, 'higher'),
        'Inventory_Turnover': (0, 20, 'higher'),
        'Receivables_Turnover': (0, 20, 'higher'),
        'Working_Capital_Turnover': (0, 10, 'higher'),
        'FCF_to_Sales': (0, 0.3, 'higher'),
        'Dividend_Yield': (0, 0.08, 'higher'),
        'Payout_Ratio': (0, 1.0, 'optimal_50'),  # 40-60% 최적
        'Dividend_Growth': (-0.2, 0.3, 'higher'),
        'Revenue_Growth': (-0.2, 0.5, 'higher'),
        'EPS_Growth': (-0.3, 0.6, 'higher'),

        # 기술적 팩터
        'RSI': (0, 100, 'optimal_50'),  # 30-70 최적
        'MACD_Signal': (-10, 10, 'higher'),
        'Stochastic': (0, 100, 'optimal_50'),
        'Williams_R': (-100, 0, 'optimal_50'),
        'CCI': (-300, 300, 'optimal_0'),
        'MA_Cross': (-20, 20, 'higher'),
        'ADX': (0, 100, 'higher'),
        'BB_Width': (0, 50, 'lower'),  # 낮을수록 안정
        'ATR': (0, 50, 'lower'),
        'Volume_Ratio': (0, 5, 'optimal_1'),  # 1 근처 최적

        # 감성 팩터
        'News_Sentiment': (-1, 1, 'higher'),
        'News_Volume': (0, 100, 'higher'),
        'News_Sentiment_Std': (0, 0.5, 'lower'),
        'Filing_Sentiment': (-1, 1, 'higher'),
        'Risk_Factor_Count': (0, 50, 'lower'),
        'Put_Call_Ratio': (0.5, 2.0, 'optimal_1'),
        'Market_VIX': (10, 50, 'lower'),
        'Short_Interest_Ratio': (0, 20, 'lower'),
        'Analyst_Rating': (1, 5, 'higher'),
        'Target_Price_Upside': (-50, 100, 'higher'),
    }

    # 기본 가중치 (카테고리별)
    DEFAULT_WEIGHTS = {
        # 재무 (40%)
        'profitability': 0.15,  # 수익성
        'health': 0.10,         # 건전성
        'efficiency': 0.08,     # 효율성
        'dividend': 0.04,       # 배당
        'growth': 0.03,         # 성장

        # 기술적 (30%)
        'technical': 0.30,

        # 감성 (30%)
        'sentiment': 0.30,
    }

    @staticmethod
    def normalize_factors(factors: Dict[str, float]) -> Dict[str, float]:
        """팩터를 0-100 범위로 정규화

        Args:
            factors: 원본 팩터 딕셔너리

        Returns:
            정규화된 팩터 딕셔너리 (0-100)
        """
        normalized = {}

        for key, value in factors.items():
            if pd.isna(value):
                normalized[key] = 50.0  # 결측치는 중립값
                continue

            if key not in FactorAggregator.FACTOR_RANGES:
                # 범위 정의 없는 팩터는 그대로 유지
                normalized[key] = value
                continue

            min_val, max_val, direction = FactorAggregator.FACTOR_RANGES[key]

            # 범위 클리핑
            clipped = max(min_val, min(max_val, value))

            # 정규화
            if direction == 'higher':
                # 높을수록 좋음
                score = ((clipped - min_val) / (max_val - min_val)) * 100
            elif direction == 'lower':
                # 낮을수록 좋음
                score = (1 - (clipped - min_val) / (max_val - min_val)) * 100
            elif direction == 'optimal_0':
                # 0에 가까울수록 좋음
                distance = abs(clipped) / max(abs(min_val), abs(max_val))
                score = (1 - distance) * 100
            elif direction == 'optimal_1':
                # 1에 가까울수록 좋음
                distance = abs(clipped - 1.0) / max(abs(max_val - 1.0), abs(min_val - 1.0))
                score = (1 - distance) * 100
            elif direction == 'optimal_50':
                # 50에 가까울수록 좋음 (RSI, Stochastic 등)
                normalized_val = ((clipped - min_val) / (max_val - min_val)) * 100
                distance = abs(normalized_val - 50) / 50
                score = (1 - distance) * 100
            else:
                score = 50.0

            normalized[key] = max(0, min(100, score))

        return normalized

    @staticmethod
    def calculate_composite_score(
        factors: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """종합 점수 계산 (0-100)

        Args:
            factors: 정규화된 팩터 딕셔너리
            weights: 카테고리별 가중치 (미지정시 기본값)

        Returns:
            종합 점수 (0-100)
        """
        if not factors:
            return 50.0

        if weights is None:
            weights = FactorAggregator.DEFAULT_WEIGHTS.copy()

        # 카테고리별 팩터 분류
        profitability_factors = ['ROE', 'ROA', 'ROIC', 'Operating_Margin', 'Net_Margin']
        health_factors = ['Debt_to_Equity', 'Current_Ratio', 'Quick_Ratio', 'Interest_Coverage', 'Debt_to_Asset']
        efficiency_factors = ['Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover', 'Working_Capital_Turnover', 'FCF_to_Sales']
        dividend_factors = ['Dividend_Yield', 'Payout_Ratio', 'Dividend_Growth']
        growth_factors = ['Revenue_Growth', 'EPS_Growth']

        technical_factors = ['RSI', 'MACD_Signal', 'Stochastic', 'Williams_R', 'CCI', 'MA_Cross', 'ADX', 'BB_Width', 'ATR', 'Volume_Ratio']

        sentiment_factors = ['News_Sentiment', 'News_Volume', 'News_Sentiment_Std', 'Filing_Sentiment', 'Risk_Factor_Count',
                           'Put_Call_Ratio', 'Market_VIX', 'Short_Interest_Ratio', 'Analyst_Rating', 'Target_Price_Upside']

        # 카테고리별 평균 계산
        category_scores = {}

        # 수익성
        prof_vals = [factors[k] for k in profitability_factors if k in factors and not pd.isna(factors[k])]
        category_scores['profitability'] = np.mean(prof_vals) if prof_vals else 50.0

        # 건전성
        health_vals = [factors[k] for k in health_factors if k in factors and not pd.isna(factors[k])]
        category_scores['health'] = np.mean(health_vals) if health_vals else 50.0

        # 효율성
        eff_vals = [factors[k] for k in efficiency_factors if k in factors and not pd.isna(factors[k])]
        category_scores['efficiency'] = np.mean(eff_vals) if eff_vals else 50.0

        # 배당
        div_vals = [factors[k] for k in dividend_factors if k in factors and not pd.isna(factors[k])]
        category_scores['dividend'] = np.mean(div_vals) if div_vals else 50.0

        # 성장
        growth_vals = [factors[k] for k in growth_factors if k in factors and not pd.isna(factors[k])]
        category_scores['growth'] = np.mean(growth_vals) if growth_vals else 50.0

        # 기술적
        tech_vals = [factors[k] for k in technical_factors if k in factors and not pd.isna(factors[k])]
        category_scores['technical'] = np.mean(tech_vals) if tech_vals else 50.0

        # 감성
        sent_vals = [factors[k] for k in sentiment_factors if k in factors and not pd.isna(factors[k])]
        category_scores['sentiment'] = np.mean(sent_vals) if sent_vals else 50.0

        # 가중 평균
        total_weight = sum(weights.values())
        composite_score = sum(category_scores[cat] * weights[cat] for cat in category_scores if cat in weights)
        composite_score = composite_score / total_weight if total_weight > 0 else 50.0

        return round(composite_score, 2)

    @staticmethod
    def rank_stocks(
        tickers: List[str],
        market: str = "US",
        factor_weights: Optional[Dict[str, float]] = None,
        include_technical: bool = True,
        include_financial: bool = True,
        include_sentiment: bool = True
    ) -> List[Dict]:
        """다종목 팩터 기반 랭킹

        Args:
            tickers: 종목 코드 리스트
            market: 시장 (US/KR)
            factor_weights: 카테고리 가중치
            include_technical: 기술적 팩터 포함 여부
            include_financial: 재무 팩터 포함 여부
            include_sentiment: 감성 팩터 포함 여부

        Returns:
            종목별 점수 및 랭킹 정보
        """
        results = []

        for ticker in tickers:
            try:
                # 팩터 수집
                all_factors = {}

                if include_financial:
                    fin_factors = FinancialFactors.calculate_all(ticker, market)
                    all_factors.update(fin_factors)

                if include_technical:
                    try:
                        from mcp_server.tools.technical_indicators import TechnicalFactors
                        import yfinance as yf
                        stock = yf.Ticker(ticker)
                        df = stock.history(period="6mo")
                        if not df.empty:
                            tech_factors = TechnicalFactors.calculate_all(df)
                            all_factors.update(tech_factors)
                    except Exception as e:
                        logger.warning(f"Technical factors failed for {ticker}: {e}")

                if include_sentiment:
                    sent_factors = SentimentFactors.calculate_all(ticker, market, days=7)
                    all_factors.update(sent_factors)

                # 정규화
                normalized = FactorAggregator.normalize_factors(all_factors)

                # 종합 점수
                composite = FactorAggregator.calculate_composite_score(normalized, factor_weights)

                results.append({
                    'ticker': ticker,
                    'composite_score': composite,
                    'factor_count': len(all_factors),
                    'factors': all_factors,
                    'normalized_factors': normalized
                })

            except Exception as e:
                logger.error(f"Ranking failed for {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'composite_score': 0.0,
                    'factor_count': 0,
                    'error': str(e)
                })

        # 점수 기준 정렬
        results.sort(key=lambda x: x['composite_score'], reverse=True)

        # 랭킹 추가
        for i, result in enumerate(results, 1):
            result['rank'] = i

        return results

    @staticmethod
    def get_recommendation(composite_score: float) -> str:
        """종합 점수 기반 투자 의견

        Args:
            composite_score: 종합 점수 (0-100)

        Returns:
            투자 의견
        """
        if composite_score >= 75:
            return "Strong Buy"
        elif composite_score >= 65:
            return "Buy"
        elif composite_score >= 55:
            return "Hold"
        elif composite_score >= 45:
            return "Neutral"
        elif composite_score >= 35:
            return "Sell"
        else:
            return "Strong Sell"

    @staticmethod
    def explain_score_breakdown(
        normalized_factors: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict]:
        """점수 구성 상세 설명

        Args:
            normalized_factors: 정규화된 팩터
            weights: 가중치

        Returns:
            카테고리별 세부 점수
        """
        if weights is None:
            weights = FactorAggregator.DEFAULT_WEIGHTS.copy()

        profitability_factors = ['ROE', 'ROA', 'ROIC', 'Operating_Margin', 'Net_Margin']
        health_factors = ['Debt_to_Equity', 'Current_Ratio', 'Quick_Ratio', 'Interest_Coverage', 'Debt_to_Asset']
        efficiency_factors = ['Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover', 'Working_Capital_Turnover', 'FCF_to_Sales']
        dividend_factors = ['Dividend_Yield', 'Payout_Ratio', 'Dividend_Growth']
        growth_factors = ['Revenue_Growth', 'EPS_Growth']
        technical_factors = ['RSI', 'MACD_Signal', 'Stochastic', 'Williams_R', 'CCI', 'MA_Cross', 'ADX', 'BB_Width', 'ATR', 'Volume_Ratio']
        sentiment_factors = ['News_Sentiment', 'News_Volume', 'News_Sentiment_Std', 'Filing_Sentiment', 'Risk_Factor_Count',
                           'Put_Call_Ratio', 'Market_VIX', 'Short_Interest_Ratio', 'Analyst_Rating', 'Target_Price_Upside']

        breakdown = {}

        categories = {
            'profitability': profitability_factors,
            'health': health_factors,
            'efficiency': efficiency_factors,
            'dividend': dividend_factors,
            'growth': growth_factors,
            'technical': technical_factors,
            'sentiment': sentiment_factors
        }

        for category, factor_list in categories.items():
            vals = [normalized_factors[k] for k in factor_list if k in normalized_factors and not pd.isna(normalized_factors[k])]
            avg_score = np.mean(vals) if vals else 50.0
            weight = weights.get(category, 0.0)

            breakdown[category] = {
                'score': round(avg_score, 2),
                'weight': weight,
                'contribution': round(avg_score * weight, 2),
                'factor_count': len(vals)
            }

        return breakdown
