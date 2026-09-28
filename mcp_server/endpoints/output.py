"""Chart and Markdown report tools."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.endpoints._common import READ_EXTERNAL, WRITE_EXTERNAL, Period, ToolSpec, check, clean_tickers, parse_holdings

ChartKind = Literal[
    "candlestick",
    "technical",
    "comparison",
    "relative_strength",
    "returns",
    "allocation",
    "correlation",
    "sectors",
    "dashboard",
]


def _require(value: Any, field: str, kind: str) -> Any:
    if not value:
        raise ToolError(f"kind='{kind}'에는 {field}가 필요합니다.")
    return value


def chart(
    kind: ChartKind,
    ticker: str | None = None,
    tickers: list[str] | None = None,
    holdings_text: str | None = None,
    period: Period = "6mo",
    benchmark: str = "SPY",
    indicators: list[Literal["rsi", "macd", "bbands", "volume"]] | None = None,
    ma_periods: list[int] | None = None,
    show_volume: bool = True,
    normalize: bool = True,
    save_as: Annotated[str | None, Field(description="지정하면 data/charts/<save_as>.html로도 저장")] = None,
) -> dict[str, Any]:
    """Plotly 차트를 생성해 HTML 조각(chart_html, CDN 스크립트 사용)으로 반환합니다.

    - ticker 필요: candlestick(ma_periods, show_volume), technical(indicators), relative_strength(benchmark),
      returns(수익률 분포 + 통계), dashboard(4종 세트 → charts)
    - tickers 필요: comparison(normalize), correlation(히트맵 + 분산 점수)
    - holdings_text 필요: allocation(비중 파이), sectors(섹터 막대)
    """
    from mcp_server.tools import visualizer as viz

    result: dict[str, Any] = {"kind": kind, "period": period}

    if kind == "dashboard":
        symbol = _require(ticker, "ticker", kind)
        figs = viz.create_stock_dashboard(symbol, period)
        result["ticker"] = symbol
        result["charts"] = {name: viz.chart_to_html(fig) for name, fig in figs.items()}
        if save_as:
            result["saved_paths"] = {name: viz.save_chart(fig, f"{save_as}_{name}") for name, fig in figs.items()}
        return result

    if kind == "candlestick":
        symbol = _require(ticker, "ticker", kind)
        fig = viz.create_candlestick_chart(symbol, period, show_volume, ma_periods or [20, 50])
        result["ticker"] = symbol
    elif kind == "technical":
        symbol = _require(ticker, "ticker", kind)
        ind = list(indicators or ["rsi", "macd"])
        fig = viz.create_technical_chart(symbol, period, ind)
        result.update(ticker=symbol, indicators=ind)
    elif kind == "relative_strength":
        symbol = _require(ticker, "ticker", kind)
        fig = viz.create_relative_strength_chart(symbol, benchmark, period)
        result.update(ticker=symbol, benchmark=benchmark)
    elif kind == "returns":
        symbol = _require(ticker, "ticker", kind)
        fig = viz.create_returns_distribution(symbol, period)
        df = viz._get_ohlcv(symbol, period)
        if not df.empty:
            r = df["Close"].pct_change().dropna() * 100
            result["statistics"] = {
                "mean": round(float(r.mean()), 3),
                "std": round(float(r.std()), 3),
                "skewness": round(float(r.skew()), 3),
                "kurtosis": round(float(r.kurtosis()), 3),
                "var_5pct": round(float(r.quantile(0.05)), 3),
                "max_daily_gain": round(float(r.max()), 3),
                "max_daily_loss": round(float(r.min()), 3),
            }
        result["ticker"] = symbol
    elif kind == "comparison":
        symbols = clean_tickers(_require(tickers, "tickers", kind), minimum=2)
        fig = viz.create_comparison_chart(symbols, period, normalize)
        result.update(tickers=symbols, normalized=normalize)
    elif kind == "correlation":
        from mcp_server.tools.portfolio_manager import analyze_correlation

        symbols = clean_tickers(_require(tickers, "tickers", kind), minimum=2)
        corr = check(analyze_correlation(symbols, period))
        fig = viz.create_correlation_heatmap(corr["correlation_matrix"])
        result.update(
            tickers=symbols,
            diversification_score=corr.get("diversification_score"),
            average_correlation=corr.get("average_correlation"),
        )
    elif kind == "allocation":
        from mcp_server.tools.portfolio_manager import _get_current_price

        holdings = parse_holdings(_require(holdings_text, "holdings_text", kind))
        values = {}
        for h in holdings:
            price = _get_current_price(h.ticker)
            if price:
                values[h.ticker] = h.shares * price
        fig = viz.create_portfolio_pie_chart(values)
        result.update(holdings_count=len(values), total_value=sum(values.values()))
    else:  # sectors
        from mcp_server.tools.portfolio_manager import analyze_sector_exposure

        holdings = parse_holdings(_require(holdings_text, "holdings_text", kind))
        sectors = check(analyze_sector_exposure(holdings))
        fig = viz.create_sector_bar_chart(sectors["sectors"])
        result.update(
            sector_count=sectors["sector_count"],
            concentration_level=sectors["concentration_level"],
            sectors=sectors["sectors"],
        )

    if save_as:
        result["saved_path"] = viz.save_chart(fig, save_as)
    result["chart_html"] = viz.chart_to_html(fig)
    return result


def report_create(
    kind: Literal["theme", "theme_overview", "selection", "portfolio_phase", "portfolio_overview"],
    tickers: Annotated[list[str], Field(min_length=1)],
    theme: str | None = None,
    with_images: Annotated[bool, Field(description="overview 리포트에 가격 차트 PNG 경로 포함")] = False,
    days: Annotated[int | None, Field(ge=5, le=730, description="overview 차트/이력 기간(일)")] = None,
) -> str:
    """마크다운 리포트를 생성해 본문을 반환합니다.

    - theme: 테마 뉴스 요약(LLM) + SEC 공시 요약 + 펀더멘털 랭킹 스냅샷 (theme 필요)
    - theme_overview: 테마 뉴스·가격 개요 표 (theme 필요)
    - selection: 선택 종목의 랭킹과 최근 공시 수 요약 (theme 필요)
    - portfolio_phase: 보유 종목 페이즈·점수 리포트
    - portfolio_overview: 보유 종목 페이즈·가격 이력 개요
    """
    symbols = clean_tickers(tickers)
    if kind in ("theme", "theme_overview", "selection") and not theme:
        raise ToolError(f"kind='{kind}'에는 theme이 필요합니다.")

    if kind == "theme":
        from mcp_server.pipelines.theme_report import run_theme_report

        return run_theme_report(theme, symbols)
    if kind == "theme_overview":
        from mcp_server.tools.presenter import present_theme_overview

        extra = {"chart_days": days} if days else {}
        return present_theme_overview(theme, symbols, with_images=with_images, **extra)
    if kind == "selection":
        from mcp_server.tools.interaction import analyze_selection

        return analyze_selection(theme, symbols)
    if kind == "portfolio_phase":
        from mcp_server.pipelines.portfolio_report import run_portfolio_report

        return run_portfolio_report(symbols)
    from mcp_server.tools.presenter import present_portfolio_overview

    extra = {"history_days": days} if days else {}
    return present_portfolio_overview(symbols, with_images=with_images, **extra)


TOOLS: list[ToolSpec] = [
    (chart, WRITE_EXTERNAL),
    (report_create, READ_EXTERNAL),
]
