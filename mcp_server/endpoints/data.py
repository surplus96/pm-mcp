"""Raw data access tools: prices, news, filings, Finnhub."""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.endpoints._common import READ_EXTERNAL, READ_LOCAL, WRITE_EXTERNAL, Market, Period, ToolSpec, check, clean_tickers


def market_prices(
    ticker: str,
    market: Market | None = None,
    start: str | None = None,
    end: str | None = None,
    period: Period | None = None,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    view: Literal["rows", "summary", "csv"] = "rows",
    cursor: Annotated[int, Field(ge=0)] = 0,
    page_size: Annotated[int, Field(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """OHLCV 시세 조회 (US/KR). 기본 구간은 최근 1년이며 start 또는 period로 지정합니다.

    - view="rows": 행 데이터를 page_size 단위로 반환 (next_cursor로 이어서 조회)
    - view="summary": 주간 리샘플 요약 통계 (토큰 절약)
    - view="csv": 전체 구간을 data/processed에 CSV로 저장하고 경로를 반환
    """
    from mcp_server.tools.market_data import get_prices, get_prices_summary, write_prices_csv

    if view == "summary":
        summary = get_prices_summary(ticker, period=period or "1y", interval=interval, agg="W")
        if not summary.get("count"):
            raise ToolError(f"{ticker}: 가격 데이터가 없습니다.")
        return summary
    if view == "csv":
        return {"ticker": ticker, "csv_path": write_prices_csv(ticker, start=start, end=end, interval=interval)}

    df = get_prices(ticker, start=start, end=end, interval=interval, market=market, period=period)
    if df.empty:
        raise ToolError(f"{ticker}: 가격 데이터가 없습니다.")
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    page = rows[cursor: cursor + page_size]
    next_cursor = cursor + page_size if cursor + page_size < len(rows) else None
    return {"ticker": ticker, "total_rows": len(rows), "rows": page, "next_cursor": next_cursor}


def news_search(
    queries: Annotated[list[str], Field(min_length=1)],
    lookback_days: Annotated[int, Field(ge=1, le=90)] = 7,
    max_results: Annotated[int, Field(ge=1, le=50)] = 10,
) -> list[dict[str, Any]]:
    """Google News RSS에서 검색어별 최근 기사(제목·출처·URL·요약)를 가져옵니다."""
    from mcp_server.tools.news_search import search_news

    return search_news(queries, lookback_days=lookback_days, max_results=max_results)


def _timeline_view(ticker: str, lookback_days: int) -> dict[str, Any]:
    from mcp_server.tools.news_sentiment import analyze_ticker_news

    result = analyze_ticker_news(ticker, lookback_days=lookback_days, use_llm=False)
    timeline = result.get("timeline", [])
    trend = []
    for day in timeline:
        items = day.get("items", [])
        if items:
            trend.append({
                "date": day.get("date", "unknown"),
                "score": round(sum(i.get("sentiment_score", 0) for i in items) / len(items), 3),
                "count": len(items),
            })
    return {
        "ticker": ticker,
        "period_days": lookback_days,
        "timeline": timeline,
        "sentiment_trend": trend,
        "overall": result.get("overall", "neutral"),
        "investment_signal": result.get("investment_signal", ""),
    }


def news_sentiment(
    tickers: Annotated[list[str], Field(min_length=1)],
    lookback_days: Annotated[int, Field(ge=1, le=60)] = 7,
    view: Literal["detail", "compare", "timeline"] = "detail",
    use_llm: bool = False,
) -> dict[str, Any]:
    """종목 뉴스 감성 분석 (bullish/bearish 점수, 분포, 투자 신호).

    - view="detail": 종목별 상세 결과 (최대 5종목)
    - view="compare": 여러 종목의 감성 점수 순위와 최고/최저 종목 (최대 10종목)
    - view="timeline": 날짜별 뉴스 흐름과 감성 추이 (종목별)
    """
    from mcp_server.tools.news_sentiment import analyze_ticker_news, compare_tickers_sentiment

    if view == "compare":
        return compare_tickers_sentiment(clean_tickers(tickers, limit=10), lookback_days=lookback_days)
    symbols = clean_tickers(tickers, limit=5)
    if view == "timeline":
        return {t: _timeline_view(t, lookback_days) for t in symbols}
    return {t: analyze_ticker_news(t, lookback_days=lookback_days, use_llm=use_llm) for t in symbols}


def text_sentiment(text: Annotated[str, Field(min_length=1)]) -> dict[str, Any]:
    """임의의 텍스트(뉴스 헤드라인·본문)의 감성, 점수, 영향도와 매칭된 키워드를 분석합니다."""
    from mcp_server.tools.news_sentiment import get_analyzer

    analyzer = get_analyzer()
    sentiment = analyzer.analyze_text(text)
    impact = analyzer.analyze_impact(text)
    return {**sentiment, "impact": impact["impact"], "impact_score": impact["score"], "impact_factors": impact["factors"]}


def filings_recent(
    ticker: str,
    forms: list[str] | None = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> list[dict[str, Any]]:
    """SEC EDGAR 최근 공시 목록 (기본 8-K/10-Q/10-K). 미국 종목 전용입니다."""
    from mcp_server.tools.filings import fetch_recent_filings

    return fetch_recent_filings(ticker, forms=forms, limit=limit)


def finnhub_data(
    kind: Literal["summary", "news", "insider", "analyst", "earnings", "financials"],
    symbol: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    """Finnhub 데이터 (FINNHUB_API_KEY 필요, 미국 종목).

    - summary: 뉴스+내부자+애널리스트+재무 종합 신호
    - news: 회사 뉴스(감성 포함, from_date/to_date) · insider: 내부자 거래 · analyst: 추천 등급 추이
    - earnings: 실적 발표 일정(symbol 생략 시 전체) · financials: P/E·ROE·성장률 등 기본 재무
    """
    from mcp_server.tools import finnhub_api as fh

    if kind == "earnings":
        return check(fh.get_earnings_calendar(symbol=symbol, from_date=from_date, to_date=to_date))
    if not symbol:
        raise ToolError(f"kind='{kind}'에는 symbol이 필요합니다.")
    handlers = {
        "summary": lambda: fh.get_finnhub_summary(symbol),
        "news": lambda: fh.get_company_news(symbol, from_date=from_date, to_date=to_date),
        "insider": lambda: fh.get_insider_transactions(symbol),
        "analyst": lambda: fh.get_analyst_recommendations(symbol),
        "financials": lambda: fh.get_basic_financials(symbol),
    }
    return check(handlers[kind]())


TOOLS: list[ToolSpec] = [
    (market_prices, WRITE_EXTERNAL),
    (news_search, READ_EXTERNAL),
    (news_sentiment, READ_EXTERNAL),
    (text_sentiment, READ_LOCAL),
    (filings_recent, READ_EXTERNAL),
    (finnhub_data, READ_EXTERNAL),
]
