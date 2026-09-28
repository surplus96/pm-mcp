"""Single-stock analysis, ranking, market context and backtesting tools."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Literal

import anyio
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.endpoints._common import READ_EXTERNAL, Market, Period, ToolSpec, check, clean_tickers

logger = logging.getLogger(__name__)

Section = Literal["technical", "financial", "sentiment"]


def stock_snapshot(ticker: str, include_signal: bool = False) -> dict[str, Any]:
    """종목 스냅샷: 가격·변동성, 밸류에이션·수익성·성장성, 애널리스트 컨센서스, 뉴스 감성, 기술적 신호와 종합 신호(composite_signal).

    단일 종목 질문의 기본 도구입니다. 미국 티커(AAPL)와 한국 종목코드(005930) 모두 지원합니다.
    include_signal=True면 Buy/Hold/Sell 의사결정 신호와 근거·리스크를 함께 반환합니다.
    """
    from mcp_server.tools.data_integrator import get_investment_signal, get_stock_analysis

    if include_signal:
        return check(get_investment_signal(ticker))
    return check(get_stock_analysis(ticker))


def stock_factors(
    ticker: str,
    market: Market = "US",
    sections: list[Section] | None = None,
    period: Period = "1y",
    sentiment_days: Annotated[int, Field(ge=1, le=60)] = 7,
) -> dict[str, Any]:
    """팩터 분석: 기술적(10) · 재무(20) · 감성(10) 지표 값, 해석, 섹션별 점수와 종합 점수/추천 등급.

    sections로 필요한 영역만 계산할 수 있습니다(예: ["technical"]). 한국 주식은 market="KR".
    """
    from mcp_server.tools.factor_aggregator import FactorAggregator
    from mcp_server.tools.financial_factors import FinancialFactors
    from mcp_server.tools.market_data import get_prices
    from mcp_server.tools.sentiment_analysis import SentimentFactors, calculate_sentiment_score
    from mcp_server.tools.technical_indicators import TechnicalFactors, calculate_technical_score

    wanted = set(sections or ("technical", "financial", "sentiment"))
    factors: dict[str, dict[str, float]] = {}
    interpretation: dict[str, str] = {}
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    if "technical" in wanted:
        try:
            df = get_prices(ticker, market=market, period=period)
            if df.empty:
                raise ValueError(f"가격 데이터가 없습니다: {ticker}")
            factors["technical"] = TechnicalFactors.calculate_all(df)
            interpretation.update(TechnicalFactors.get_factor_interpretation(factors["technical"]))
            scores["technical"] = round(calculate_technical_score(df), 2)
        except Exception as e:  # noqa: BLE001
            logger.warning("technical factors failed for %s: %s", ticker, e)
            errors["technical"] = str(e)

    if "financial" in wanted:
        try:
            factors["financial"] = FinancialFactors.calculate_all(ticker, market)
            interpretation.update(FinancialFactors.get_factor_interpretation(factors["financial"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("financial factors failed for %s: %s", ticker, e)
            errors["financial"] = str(e)

    if "sentiment" in wanted:
        try:
            factors["sentiment"] = SentimentFactors.calculate_all(ticker, market, sentiment_days)
            interpretation.update(SentimentFactors.get_factor_interpretation(factors["sentiment"]))
            scores["sentiment"] = round(calculate_sentiment_score(factors["sentiment"]), 2)
        except Exception as e:  # noqa: BLE001
            logger.warning("sentiment factors failed for %s: %s", ticker, e)
            errors["sentiment"] = str(e)

    flat = {k: v for group in factors.values() for k, v in group.items()}
    if not flat:
        raise ToolError(f"{ticker}: 팩터를 계산하지 못했습니다. {errors}")

    composite = FactorAggregator.calculate_composite_score(FactorAggregator.normalize_factors(flat))
    result: dict[str, Any] = {
        "ticker": ticker,
        "market": market,
        "timestamp": datetime.now().isoformat(),
        "factors": factors,
        "interpretation": interpretation,
        "section_scores": scores,
        "composite_score": round(composite, 2),
        "recommendation": FactorAggregator.get_recommendation(composite),
        "total_factors": len(flat),
    }
    if errors:
        result["errors"] = errors
    return result


def stock_compare(tickers: Annotated[list[str], Field(min_length=2, max_length=5)]) -> dict[str, Any]:
    """2~5개 종목을 스냅샷 지표(가격·밸류에이션·감성·종합 신호)로 나란히 비교하고 종합 점수 순으로 정렬합니다."""
    from mcp_server.tools.data_integrator import compare_stocks

    return check(compare_stocks(clean_tickers(tickers, limit=5, minimum=2)))


async def stock_rank(
    tickers: list[str],
    ctx: Context,
    method: Literal["factor", "advanced", "fundamental"] = "factor",
    market: Market = "US",
    include_technical: bool = True,
    include_financial: bool = True,
    include_sentiment: bool = True,
    use_sector_weights: bool = True,
    use_market_adjustment: bool = True,
    sector_neutral: bool = False,
    dip_weight: float = 0.12,
    use_dip_bonus: bool = True,
) -> list[dict[str, Any]]:
    """여러 종목을 점수화해 순위를 매깁니다.

    - method="factor": 기술/재무/감성 팩터 종합 점수 + 추천 등급 (include_* 옵션 적용, US/KR)
    - method="advanced": 6개 팩터 Z-score + 섹터별 가중치 + 시장 국면 보정 (use_sector_weights, use_market_adjustment, sector_neutral, dip_* 적용)
    - method="fundamental": 펀더멘털 4팩터(성장·수익성·밸류·퀄리티) + 낙폭 보너스 (dip_* 적용)
    """
    symbols = clean_tickers(tickers)
    await ctx.report_progress(0, 1, f"{len(symbols)}개 종목 순위 계산 중 ({method})")

    if method == "advanced":
        from mcp_server.tools.ranking_engine import rank_advanced_async

        result = await rank_advanced_async(
            symbols,
            use_sector_weights=use_sector_weights,
            use_market_adjustment=use_market_adjustment,
            sector_neutral=sector_neutral,
            dip_weight=dip_weight,
            use_dip_bonus=use_dip_bonus,
        )
    elif method == "fundamental":
        from mcp_server.tools.analytics import rank_tickers_with_fundamentals_async

        result = await rank_tickers_with_fundamentals_async(symbols, dip_weight=dip_weight, use_dip_bonus=use_dip_bonus)
    else:
        from mcp_server.tools.factor_aggregator import FactorAggregator

        def _rank() -> list[dict[str, Any]]:
            rows = FactorAggregator.rank_stocks(
                tickers=symbols,
                market=market,
                include_technical=include_technical,
                include_financial=include_financial,
                include_sentiment=include_sentiment,
            )
            for row in rows:
                if "composite_score" in row:
                    row["recommendation"] = FactorAggregator.get_recommendation(row["composite_score"])
            return rows

        result = await anyio.to_thread.run_sync(_rank)

    await ctx.report_progress(1, 1, "완료")
    return check(result)


def market_overview(sector: str | None = None) -> dict[str, Any]:
    """시장 국면(강세/약세/횡보)과 랭킹 엔진이 쓰는 섹터별 팩터 가중치를 반환합니다. 매수/매도 판단 답변의 맥락으로 사용합니다."""
    from mcp_server.tools.ranking_engine import DEFAULT_WEIGHTS, SECTOR_WEIGHTS, get_ranking_engine

    out: dict[str, Any] = {"market_condition": get_ranking_engine().detect_market()}
    if sector:
        out["sector"] = sector
        out["weights"] = SECTOR_WEIGHTS.get(sector, DEFAULT_WEIGHTS)
    else:
        out["sectors"] = list(SECTOR_WEIGHTS)
        out["default_weights"] = DEFAULT_WEIGHTS
    return out


async def backtest_strategy(
    ticker: str,
    ctx: Context,
    market: Market = "US",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    rebalance_period: Annotated[int, Field(ge=1, le=365, description="리밸런싱 주기(일)")] = 30,
    buy_threshold: Annotated[float, Field(ge=0, le=100)] = 60.0,
    sell_threshold: Annotated[float, Field(ge=0, le=100)] = 40.0,
    initial_capital: Annotated[float, Field(gt=0)] = 10000.0,
) -> dict[str, Any]:
    """팩터 점수 기반 매매 전략 백테스트. CAGR·MDD·Sharpe·승률·거래 내역과 SPY 대비 성과를 반환합니다.

    결과를 설명할 때는 과최적화·생존 편향 가능성을 함께 언급하세요.
    """
    from mcp_server.tools.backtest_engine import BacktestEngine

    if sell_threshold >= buy_threshold:
        raise ToolError("sell_threshold는 buy_threshold보다 작아야 합니다.")

    await ctx.report_progress(0, 1, f"{ticker} 백테스트 실행 중")
    result = await anyio.to_thread.run_sync(
        lambda: BacktestEngine.run_backtest(
            ticker=ticker,
            market=market,
            start_date=start_date,
            end_date=end_date,
            rebalance_period=rebalance_period,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            initial_capital=initial_capital,
        )
    )
    await ctx.report_progress(1, 1, "완료")
    return check(result)


TOOLS: list[ToolSpec] = [
    (stock_snapshot, READ_EXTERNAL),
    (stock_factors, READ_EXTERNAL),
    (stock_compare, READ_EXTERNAL),
    (stock_rank, READ_EXTERNAL),
    (market_overview, READ_EXTERNAL),
    (backtest_strategy, READ_EXTERNAL),
]
