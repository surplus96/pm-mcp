"""Read-only MCP resources and reusable prompts."""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError


def watchlist_resource() -> str:
    """스케줄러가 추적하는 워치리스트 (tickers, themes)."""
    from mcp_server.endpoints.portfolio import _read_watchlist

    return json.dumps(_read_watchlist(), ensure_ascii=False)


def portfolios_resource() -> str:
    """저장된 포트폴리오 이름 목록."""
    from mcp_server.tools.portfolio_manager import list_portfolios

    return json.dumps(list_portfolios(), ensure_ascii=False)


def portfolio_resource(name: str) -> str:
    """저장된 포트폴리오 한 개 (보유 종목, 현금, 생성/수정 시각)."""
    from mcp_server.tools.portfolio_manager import load_portfolio

    portfolio = load_portfolio(name)
    if not portfolio:
        raise ResourceNotFoundError(f"portfolio '{name}' not found")
    return json.dumps(portfolio.to_dict(), ensure_ascii=False)


def news_keywords_resource() -> str:
    """뉴스 감성·영향도 평가에 쓰는 키워드 사전."""
    from mcp_server.tools.news_sentiment import IMPACT_KEYWORDS, SENTIMENT_KEYWORDS

    return json.dumps(
        {
            "impact_keywords": {k: {"keywords": v["keywords"], "weight": v["weight"]} for k, v in IMPACT_KEYWORDS.items()},
            "sentiment_keywords": {
                k: {"keywords": v["keywords"], "score": v["score"]} for k, v in SENTIMENT_KEYWORDS.items()
            },
        },
        ensure_ascii=False,
    )


def sector_weights_resource() -> str:
    """고급 랭킹(stock_rank method="advanced")의 섹터별 팩터 가중치."""
    from mcp_server.tools.ranking_engine import DEFAULT_WEIGHTS, SECTOR_WEIGHTS

    return json.dumps({"default": DEFAULT_WEIGHTS, "sectors": SECTOR_WEIGHTS}, ensure_ascii=False)


def analyze_stock(ticker: str, market: str = "US") -> str:
    """단일 종목 심층 분석 워크플로."""
    return (
        f"{ticker} (market={market}) 종목을 분석해줘.\n"
        f"1. stock_snapshot(ticker='{ticker}', include_signal=True)로 종합 신호를 확인\n"
        f"2. stock_factors(ticker='{ticker}', market='{market}')로 팩터 점수와 해석 확인\n"
        "3. market_overview()로 시장 국면 확인\n"
        "핵심 결론 → 근거 지표 → 리스크 순으로 정리하고, 도구가 반환한 수치만 인용해. "
        "정보 제공 목적이며 투자 판단은 사용자 몫임을 마지막에 짧게 밝혀."
    )


def portfolio_checkup(holdings_text: str, cash: float = 0) -> str:
    """보유 포트폴리오 점검 워크플로."""
    return (
        f"내 포트폴리오를 점검해줘. 보유 종목: {holdings_text}, 현금: {cash}\n"
        "1. portfolio_analyze(aspect='comprehensive')로 건강도·손익·섹터·상관관계 확인\n"
        "2. 적신호 종목이 있으면 stock_snapshot으로 원인 확인\n"
        "3. market_overview()로 시장 국면을 반영해 조정 제안\n"
        "핵심 결론 → 근거 → 리스크 → 제안 순으로 정리해."
    )


def discover_themes(lookback_days: int = 7) -> str:
    """테마 발굴 → 종목 선정 워크플로."""
    return (
        f"최근 {lookback_days}일 뉴스 기준으로 유망 투자 테마를 찾아줘.\n"
        f"1. theme_propose(lookback_days={lookback_days})로 테마 후보 제시\n"
        "2. 내가 테마를 고르면 theme_explore로 대표 종목과 개요 확인\n"
        "3. theme_analyze로 팩터 기반 상위 종목 선정, 필요하면 dip_candidates로 저점 후보 확인"
    )


def register(mcp: MCPServer) -> None:
    mcp.resource("pm://watchlist", name="watchlist", mime_type="application/json")(watchlist_resource)
    mcp.resource("pm://portfolios", name="portfolios", mime_type="application/json")(portfolios_resource)
    mcp.resource("pm://portfolios/{name}", name="portfolio", mime_type="application/json")(portfolio_resource)
    mcp.resource("pm://reference/news-keywords", name="news-keywords", mime_type="application/json")(
        news_keywords_resource
    )
    mcp.resource("pm://reference/sector-weights", name="sector-weights", mime_type="application/json")(
        sector_weights_resource
    )
    for prompt in (analyze_stock, portfolio_checkup, discover_themes):
        mcp.prompt()(prompt)
