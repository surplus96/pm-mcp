#!/usr/bin/env python3
"""
Backtest Engine Module

팩터 기반 백테스트 실행 및 성과 분석
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
import logging

from mcp_server.tools.financial_factors import FinancialFactors
from mcp_server.tools.technical_indicators import TechnicalFactors
from mcp_server.tools.factor_aggregator import FactorAggregator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """팩터 기반 백테스트 엔진"""

    @staticmethod
    def run_backtest(
        ticker: str,
        market: str = "US",
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        factor_weights: Optional[Dict[str, float]] = None,
        rebalance_period: int = 30,
        buy_threshold: float = 60.0,
        sell_threshold: float = 40.0,
        initial_capital: float = 10000.0
    ) -> Dict:
        """팩터 기반 백테스트 실행

        Args:
            ticker: 종목 코드
            market: 시장 (US/KR)
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            factor_weights: 팩터 가중치
            rebalance_period: 리밸런싱 주기 (일)
            buy_threshold: 매수 임계값 (팩터 점수)
            sell_threshold: 매도 임계값
            initial_capital: 초기 자본

        Returns:
            백테스트 결과
        """
        try:
            # 가격 데이터 다운로드
            stock = yf.Ticker(ticker)
            prices = stock.history(start=start_date, end=end_date)

            if prices.empty:
                raise ValueError(f"No price data for {ticker}")

            # 리밸런싱 날짜 생성
            rebalance_dates = pd.date_range(
                start=prices.index[0],
                end=prices.index[-1],
                freq=f'{rebalance_period}D'
            )

            # 거래 기록
            trades = []
            position = 0  # 0: 보유 없음, 1: 보유 중
            shares = 0
            cash = initial_capital

            for date in rebalance_dates:
                # 가장 가까운 거래일 찾기
                if date not in prices.index:
                    nearby_dates = prices.index[prices.index >= date]
                    if len(nearby_dates) == 0:
                        continue
                    date = nearby_dates[0]

                current_price = prices.loc[date, 'Close']

                # 팩터 점수 계산 (해당 시점 기준)
                try:
                    # 기술적 팩터 (최근 6개월 데이터)
                    hist_data = prices.loc[:date].tail(120)
                    tech_factors = TechnicalFactors.calculate_all(hist_data) if len(hist_data) > 20 else {}

                    # 재무 팩터 (백테스트에서는 현재 시점 데이터 사용)
                    # 실제로는 해당 시점의 재무제표가 필요하나 단순화
                    fin_factors = FinancialFactors.calculate_all(ticker, market)

                    # 감성 팩터는 백테스트에서 제외 (과거 데이터 없음)
                    all_factors = {**tech_factors, **fin_factors}

                    # 정규화 및 점수 계산
                    normalized = FactorAggregator.normalize_factors(all_factors)
                    composite_score = FactorAggregator.calculate_composite_score(normalized, factor_weights)

                except Exception as e:
                    logger.warning(f"Factor calculation failed at {date}: {e}")
                    composite_score = 50.0

                # 매매 로직
                if composite_score >= buy_threshold and position == 0:
                    # 매수
                    shares = cash / current_price
                    cash = 0
                    position = 1
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'factor_score': composite_score,
                        'portfolio_value': shares * current_price
                    })

                elif composite_score <= sell_threshold and position == 1:
                    # 매도
                    cash = shares * current_price
                    portfolio_value = cash
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'factor_score': composite_score,
                        'portfolio_value': portfolio_value,
                        'profit': cash - initial_capital,
                        'return': ((cash - initial_capital) / initial_capital) * 100
                    })
                    shares = 0
                    position = 0

            # 마지막 포지션 정리
            final_date = prices.index[-1]
            final_price = prices.loc[final_date, 'Close']
            if position == 1:
                cash = shares * final_price
                trades.append({
                    'date': final_date,
                    'action': 'SELL',
                    'price': final_price,
                    'shares': shares,
                    'factor_score': 0.0,
                    'portfolio_value': cash,
                    'profit': cash - initial_capital,
                    'return': ((cash - initial_capital) / initial_capital) * 100
                })

            final_value = cash if position == 0 else shares * final_price

            # 성과 분석
            performance = BacktestEngine.calculate_performance(trades, prices, initial_capital, final_value)

            # 자산 곡선
            equity_curve = BacktestEngine.generate_equity_curve(trades, prices, initial_capital)

            # 벤치마크 비교
            benchmark = BacktestEngine.compare_with_benchmark(
                equity_curve, prices, "SPY" if market == "US" else "^KS11", start_date, end_date
            )

            return {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': ((final_value - initial_capital) / initial_capital) * 100,
                'trades': trades,
                'trade_count': len(trades),
                'performance': performance,
                'equity_curve': equity_curve.to_dict() if isinstance(equity_curve, pd.Series) else equity_curve,
                'benchmark': benchmark
            }

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    @staticmethod
    def calculate_performance(
        trades: List[Dict],
        prices: pd.DataFrame,
        initial_capital: float,
        final_value: float
    ) -> Dict:
        """성과 지표 계산

        Args:
            trades: 거래 내역
            prices: 가격 데이터
            initial_capital: 초기 자본
            final_value: 최종 가치

        Returns:
            성과 지표
        """
        if not trades:
            return {
                'CAGR': 0.0,
                'Total_Return': 0.0,
                'Max_Drawdown': 0.0,
                'Sharpe_Ratio': 0.0,
                'Win_Rate': 0.0,
                'Avg_Win': 0.0,
                'Avg_Loss': 0.0,
                'Profit_Factor': 0.0
            }

        # 수익률
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        # CAGR 계산
        start_date = trades[0]['date']
        end_date = trades[-1]['date']
        years = (end_date - start_date).days / 365.25
        cagr = (((final_value / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0.0

        # 매매별 손익
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        trade_returns = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            trade_return = ((sell_price - buy_price) / buy_price) * 100
            trade_returns.append(trade_return)

        # Win Rate
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = (len(wins) / len(trade_returns)) * 100 if trade_returns else 0.0

        # 평균 수익/손실
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Profit Factor
        total_gains = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

        # Maximum Drawdown
        portfolio_values = []
        current_value = initial_capital
        for trade in trades:
            if 'portfolio_value' in trade:
                current_value = trade['portfolio_value']
                portfolio_values.append(current_value)

        if portfolio_values:
            peak = portfolio_values[0]
            max_dd = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = ((peak - value) / peak) * 100
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0

        # Sharpe Ratio (간단 계산)
        if trade_returns:
            returns_std = np.std(trade_returns)
            avg_return = np.mean(trade_returns)
            sharpe = (avg_return / returns_std) if returns_std > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            'CAGR': round(cagr, 2),
            'Total_Return': round(total_return, 2),
            'Max_Drawdown': round(max_dd, 2),
            'Sharpe_Ratio': round(sharpe, 2),
            'Win_Rate': round(win_rate, 2),
            'Avg_Win': round(avg_win, 2),
            'Avg_Loss': round(avg_loss, 2),
            'Profit_Factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
            'Total_Trades': len(trade_returns)
        }

    @staticmethod
    def generate_equity_curve(
        trades: List[Dict],
        prices: pd.DataFrame,
        initial_capital: float
    ) -> pd.Series:
        """자산 곡선 생성

        Args:
            trades: 거래 내역
            prices: 가격 데이터
            initial_capital: 초기 자본

        Returns:
            날짜별 포트폴리오 가치
        """
        equity = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = initial_capital

        position = 0
        shares = 0
        cash = initial_capital
        trade_idx = 0

        for i, date in enumerate(prices.index):
            # 해당 날짜에 거래가 있는지 확인
            while trade_idx < len(trades) and trades[trade_idx]['date'] <= date:
                trade = trades[trade_idx]
                if trade['action'] == 'BUY':
                    shares = trade['shares']
                    cash = 0
                    position = 1
                elif trade['action'] == 'SELL':
                    cash = trade['portfolio_value']
                    shares = 0
                    position = 0
                trade_idx += 1

            # 포트폴리오 가치 계산
            if position == 1:
                equity.iloc[i] = shares * prices.loc[date, 'Close']
            else:
                equity.iloc[i] = cash

        return equity

    @staticmethod
    def compare_with_benchmark(
        equity_curve: pd.Series,
        prices: pd.DataFrame,
        benchmark_ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict:
        """벤치마크 비교

        Args:
            equity_curve: 전략 자산 곡선
            prices: 전략 가격 데이터
            benchmark_ticker: 벤치마크 티커
            start_date: 시작일
            end_date: 종료일

        Returns:
            벤치마크 비교 결과
        """
        try:
            # 벤치마크 데이터 다운로드
            benchmark = yf.Ticker(benchmark_ticker)
            bench_prices = benchmark.history(start=start_date, end=end_date)

            if bench_prices.empty:
                return {'error': 'Benchmark data not available'}

            # 벤치마크 수익률
            bench_start = bench_prices['Close'].iloc[0]
            bench_end = bench_prices['Close'].iloc[-1]
            bench_return = ((bench_end - bench_start) / bench_start) * 100

            # 전략 수익률
            strategy_start = equity_curve.iloc[0]
            strategy_end = equity_curve.iloc[-1]
            strategy_return = ((strategy_end - strategy_start) / strategy_start) * 100

            # 초과 수익률
            excess_return = strategy_return - bench_return

            # 벤치마크 연환산 수익률
            years = (bench_prices.index[-1] - bench_prices.index[0]).days / 365.25
            bench_cagr = (((bench_end / bench_start) ** (1 / years)) - 1) * 100 if years > 0 else 0.0

            return {
                'benchmark_ticker': benchmark_ticker,
                'benchmark_return': round(bench_return, 2),
                'benchmark_cagr': round(bench_cagr, 2),
                'strategy_return': round(strategy_return, 2),
                'excess_return': round(excess_return, 2),
                'outperformance': excess_return > 0
            }

        except Exception as e:
            logger.warning(f"Benchmark comparison failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def optimize_weights(
        ticker: str,
        market: str,
        start_date: str,
        end_date: str,
        weight_candidates: List[Dict[str, float]],
        rebalance_period: int = 30
    ) -> Dict:
        """가중치 최적화 (그리드 서치)

        Args:
            ticker: 종목 코드
            market: 시장
            start_date: 시작일
            end_date: 종료일
            weight_candidates: 가중치 조합 리스트
            rebalance_period: 리밸런싱 주기

        Returns:
            최적 가중치 및 성과
        """
        best_result = None
        best_return = -float('inf')

        for weights in weight_candidates:
            try:
                result = BacktestEngine.run_backtest(
                    ticker=ticker,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    factor_weights=weights,
                    rebalance_period=rebalance_period
                )

                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_result = {
                        'weights': weights,
                        'performance': result['performance'],
                        'total_return': result['total_return']
                    }

            except Exception as e:
                logger.warning(f"Optimization failed for weights {weights}: {e}")
                continue

        return best_result if best_result else {'error': 'Optimization failed'}
