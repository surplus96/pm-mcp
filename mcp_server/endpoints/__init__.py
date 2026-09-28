"""MCP interface layer: tool, resource and prompt registration."""
from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = """\
PM-MCP: 미국·한국 주식 포트폴리오 매니저 도구 모음.

티커 규칙: 미국은 심볼(AAPL), 한국은 6자리 종목코드(005930) + market="KR".
보유 종목 문자열: 'TICKER:SHARES@ENTRY_PRICE, ...' (portfolio_quick_check는 느슨한 형식 허용).

의도별 시작 도구:
- 단일 종목: stock_snapshot → 필요 시 stock_factors (팩터 상세), chart(kind="dashboard")
- 여러 종목 비교/순위: stock_compare (2~5개) / stock_rank (method=factor|advanced|fundamental)
- 테마 발굴: theme_propose → theme_explore → theme_analyze, 저점 후보는 dip_candidates
- 포트폴리오: portfolio_analyze (aspect로 세부 분석) / portfolio_quick_check (매수일·가 포함 자유 입력)
- 뉴스: news_sentiment (view=detail|compare|timeline), 원문은 news_search
- 매수/매도 판단에는 market_overview로 시장 국면을 함께 확인
결과는 도구가 반환한 수치만 인용하고, 정보 제공 목적임을 밝힌다.
"""


@asynccontextmanager
async def _lifespan(_: MCPServer) -> AsyncIterator[dict]:
    # Runs after the stdio transport has claimed the real stdout. Some data
    # libraries (e.g. pykrx) print() diagnostics; send those to stderr so a
    # late buffer flush can never land on the JSON-RPC wire.
    original = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield {}
    finally:
        sys.stdout = original


def create_server() -> MCPServer:
    from mcp_server.endpoints import analysis, data, discovery, ops, output, portfolio, resources
    from mcp_server.endpoints._common import register

    mcp = MCPServer("PM-MCP", instructions=INSTRUCTIONS, version="2.0.0", lifespan=_lifespan)
    for module in (discovery, analysis, data, portfolio, output, ops):
        register(mcp, module.TOOLS)
    resources.register(mcp)
    return mcp
