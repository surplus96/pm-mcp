"""Theme discovery and candidate screening tools."""
from __future__ import annotations

from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_server.endpoints._common import READ_EXTERNAL, WRITE_EXTERNAL, Market, ToolSpec, check


def theme_propose(
    lookback_days: Annotated[int, Field(ge=1, le=90)] = 7,
    max_themes: Annotated[int, Field(ge=1, le=20)] = 5,
) -> list[str]:
    """최근 뉴스 밀도를 기준으로 투자 테마를 추천합니다."""
    from mcp_server.tools.interaction import propose_themes

    return propose_themes(lookback_days=lookback_days, max_themes=max_themes)


def theme_explore(theme: str, lookback_days: Annotated[int, Field(ge=1, le=90)] = 7) -> dict[str, Any]:
    """테마의 대표 종목 후보와 뉴스/가격 개요(마크다운)를 반환합니다. 종목 선정 전 탐색 단계에 사용합니다."""
    from mcp_server.tools.interaction import propose_tickers
    from mcp_server.tools.presenter import present_theme_overview

    tickers = propose_tickers(theme)
    overview = present_theme_overview(theme, tickers, lookback_days=lookback_days, with_images=False)
    return {"theme": theme, "tickers": tickers, "overview_md": overview}


async def theme_analyze(
    theme: str,
    ctx: Context,
    market: Market = "US",
    top_n: Annotated[int, Field(ge=1, le=20)] = 5,
    include_sentiment: bool = True,
    include_backtest: bool = False,
    rerank_by_backtest: bool = False,
    backtest_start: str = "2024-01-01",
    backtest_end: str = "2024-12-31",
    factor_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """테마 → 종목 발굴 → 팩터 점수 → (선택) 백테스트까지 한 번에 수행하는 테마 종합 분석.

    한국 테마는 market="KR"과 한글 테마명(예: "반도체")을 사용합니다.
    include_backtest=True는 종목당 백테스트를 돌리므로 수십 초 이상 걸릴 수 있습니다.
    """
    from mcp_server.tools.theme_factor_integrator import ThemeFactorIntegrator

    await ctx.report_progress(0, 1, f"'{theme}' 테마 분석 중")
    result = await anyio.to_thread.run_sync(
        lambda: ThemeFactorIntegrator.analyze_theme(
            theme=theme,
            top_n=top_n,
            include_backtest=include_backtest,
            include_sentiment=include_sentiment,
            rerank_by_backtest=rerank_by_backtest,
            market=market,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            factor_weights=factor_weights,
        )
    )
    await ctx.report_progress(1, 1, "완료")
    return check(result)


def dip_candidates(
    theme: str,
    tickers: list[str] | None = None,
    top_n: Annotated[int, Field(ge=1, le=20)] = 5,
    drawdown_min: Annotated[float, Field(ge=0, le=1, description="180일 고점 대비 최소 낙폭 (0.2 = 20%)")] = 0.2,
    ret10_min: float = 0.0,
    event_min: float = 0.5,
    save_csv: bool = True,
) -> dict[str, Any]:
    """낙폭이 크지만 펀더멘털·이벤트 점수가 양호한 저점 매수 후보를 선별합니다.

    tickers를 생략하면 테마 대표 종목을 자동으로 사용합니다. save_csv=True면 data/processed/dip에 CSV를 남깁니다.
    """
    from mcp_server.pipelines.dip_candidates import run_dip_candidates

    return run_dip_candidates(
        theme,
        tickers=tickers or None,
        top_n=top_n,
        drawdown_min=drawdown_min,
        ret10_min=ret10_min,
        event_min=event_min,
        save=save_csv,
    )


TOOLS: list[ToolSpec] = [
    (theme_propose, READ_EXTERNAL),
    (theme_explore, READ_EXTERNAL),
    (theme_analyze, READ_EXTERNAL),
    (dip_candidates, WRITE_EXTERNAL),
]
