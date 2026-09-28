"""Portfolio analysis, storage and watchlist tools."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.endpoints._common import (
    READ_EXTERNAL,
    WRITE_LOCAL,
    Period,
    ToolSpec,
    check,
    parse_holdings,
)

HOLDINGS_HELP = "보유 종목 'TICKER:SHARES@ENTRY_PRICE, ...' (예: 'AAPL:10@150, MSFT:5@400')"


def _parse_target_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in text.replace(" ", "").split(","):
        if ":" in part:
            ticker, weight = part.split(":", 1)
            try:
                weights[ticker.upper()] = float(weight) / 100
            except ValueError:
                raise ToolError(f"목표 비중을 파싱할 수 없습니다: '{part}' (형식: TICKER:WEIGHT%)") from None
    if not weights:
        raise ToolError("target_weights_text가 필요합니다. 형식: 'AAPL:30, MSFT:25, CASH:20'")
    return weights


def _apply_price_targets(holdings: list, text: str) -> None:
    for part in text.replace(" ", "").split(","):
        parts = part.split(":")
        if len(parts) < 2:
            continue
        for h in holdings:
            if h.ticker == parts[0].upper():
                try:
                    if parts[1]:
                        h.target_price = float(parts[1])
                    if len(parts) > 2 and parts[2]:
                        h.stop_loss = float(parts[2])
                except ValueError:
                    raise ToolError(f"목표가/손절가를 파싱할 수 없습니다: '{part}'") from None


def portfolio_analyze(
    holdings_text: Annotated[str, Field(description=HOLDINGS_HELP)],
    aspect: Literal["comprehensive", "pnl", "rebalance", "dividends", "alerts", "correlation", "sectors"] = "comprehensive",
    cash: Annotated[float, Field(ge=0)] = 0,
    target_weights_text: Annotated[str | None, Field(description="rebalance용 목표 비중 'TICKER:WEIGHT%, ...'")] = None,
    threshold_pct: Annotated[float, Field(gt=0, le=100, description="rebalance 편차 임계값(%)")] = 5.0,
    targets_text: Annotated[str | None, Field(description="alerts용 'TICKER:TARGET:STOP, ...'")] = None,
    days_ahead: Annotated[int, Field(ge=1, le=365, description="dividends 조회 기간(일)")] = 90,
    period: Period = "1y",
) -> dict[str, Any]:
    """포트폴리오 분석.

    - comprehensive: 건강도 점수·손익·리밸런싱·배당·알림·상관관계·섹터 통합 (기본)
    - pnl: 종목별/전체 손익 · rebalance: 목표 비중 대비 편차와 조정 수량 (target_weights_text 필요)
    - dividends: 배당 캘린더 · alerts: 목표가/손절가 도달 (targets_text)
    - correlation: 보유 종목 간 상관관계와 분산 점수 (period) · sectors: 섹터 비중과 집중도(HHI)
    """
    from mcp_server.tools import portfolio_manager as pm

    holdings = parse_holdings(holdings_text)

    if aspect == "pnl":
        result = pm.get_portfolio_summary(holdings, cash)
    elif aspect == "rebalance":
        weights = _parse_target_weights(target_weights_text or "")
        for h in holdings:
            if h.ticker in weights:
                h.target_weight = weights[h.ticker]
        result = pm.check_rebalancing(holdings, cash, threshold_pct / 100)
    elif aspect == "dividends":
        result = pm.get_dividend_calendar(holdings, days_ahead)
    elif aspect == "alerts":
        if targets_text:
            _apply_price_targets(holdings, targets_text)
        result = pm.check_price_alerts(holdings)
    elif aspect == "correlation":
        tickers = [h.ticker for h in holdings]
        if len(tickers) < 2:
            raise ToolError("상관관계 분석에는 최소 2개 종목이 필요합니다.")
        result = pm.analyze_correlation(tickers, period)
    elif aspect == "sectors":
        result = pm.analyze_sector_exposure(holdings)
    else:
        result = pm.analyze_portfolio_comprehensive(holdings, cash)
    return check(result)


def _close(ticker: str, start: str | None = None) -> float | None:
    import yfinance as yf

    from mcp_server.tools.market_data import _yf_symbol
    from mcp_server.tools.yf_utils import normalize_yf_columns

    try:
        if start:
            end = (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
            df = yf.download(_yf_symbol(ticker), start=start, end=end, progress=False, auto_adjust=True)
        else:
            df = yf.download(_yf_symbol(ticker), period="5d", progress=False, auto_adjust=True)
        closes = normalize_yf_columns(df).get("Close")
        if closes is None:
            return None
        closes = closes.dropna()
        if closes.empty:
            return None
        return float(closes.iloc[0] if start else closes.iloc[-1])
    except Exception:  # noqa: BLE001
        return None


def portfolio_quick_check(
    holdings_text: Annotated[str, Field(description="느슨한 형식 허용: 'AAPL@2024-10-01:185, LLY 2024-09-15 520, NVO'")],
) -> dict[str, Any]:
    """느슨하게 적힌 보유 종목(매수일·매수가 선택)을 받아 페이즈(상승/유지/불안정/적신호), 모멘텀·변동성·낙폭·SPY 상관,
    손익, 펀더멘털 점수를 행 데이터와 마크다운 표로 반환합니다. 매수가가 없으면 매수일 종가를 사용합니다.
    """
    from mcp_server.tools.analytics import rank_tickers_with_fundamentals
    from mcp_server.tools.collect import compute_basic_metrics
    from mcp_server.tools.parse import parse_holdings_text
    from mcp_server.tools.portfolio import evaluate_holdings

    parsed = [p for p in parse_holdings_text(holdings_text) if p.get("ticker")]
    if not parsed:
        raise ToolError("보유 종목을 인식하지 못했습니다. 예: 'AAPL@2024-10-01:185, NVO'")
    tickers = [p["ticker"] for p in parsed]

    phases = {e["ticker"]: e for e in evaluate_holdings(tickers)}
    ranks = {r["ticker"]: r for r in rank_tickers_with_fundamentals(tickers, dip_weight=0.12, use_dip_bonus=True)}

    rows = []
    for p in parsed:
        t = p["ticker"]
        metrics = compute_basic_metrics(t)
        entry_date = p.get("entry_date")
        entry_price = p.get("entry_price")
        if entry_price is None and entry_date:
            entry_price = _close(t, entry_date)
        last = _close(t)
        pnl = (last - float(entry_price)) / float(entry_price) if entry_price and last else None
        rank = ranks.get(t, {})
        rows.append({
            "ticker": t,
            "phase": phases.get(t, {}).get("phase"),
            **{k: metrics.get(k) for k in ("ret20", "mom3", "mom6", "mom12", "dd180", "vol30", "corr_spy")},
            "entry_date": entry_date,
            "entry_price": entry_price,
            "last": last,
            "pnl": round(pnl, 4) if pnl is not None else None,
            "base_score": rank.get("base_score"),
            "dip_bonus": rank.get("dip_bonus"),
            "score": rank.get("score"),
        })

    headers = list(rows[0].keys())

    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(_fmt(r[h]) for h in headers) + " |" for r in rows]
    return {"rows": rows, "markdown": "\n".join(lines)}


def portfolio_store(
    action: Literal["save", "load", "list"],
    name: str | None = None,
    holdings_text: Annotated[str | None, Field(description=HOLDINGS_HELP)] = None,
    cash: Annotated[float, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """포트폴리오를 이름으로 저장(save, 같은 이름은 덮어씀)/불러오기(load)/목록(list) 합니다."""
    from mcp_server.tools import portfolio_manager as pm

    if action == "list":
        names = pm.list_portfolios()
        return {"portfolios": names, "count": len(names)}
    if not name:
        raise ToolError(f"action='{action}'에는 name이 필요합니다.")
    if action == "load":
        portfolio = pm.load_portfolio(name)
        if not portfolio:
            raise ToolError(f"포트폴리오 '{name}'을 찾을 수 없습니다.")
        return portfolio.to_dict()
    holdings = parse_holdings(holdings_text or "")
    path = pm.save_portfolio(pm.Portfolio(name=name, holdings=holdings, cash=cash), name)
    return {"saved": True, "name": name, "filepath": path, "holdings_count": len(holdings), "cash": cash}


def _read_watchlist() -> dict[str, Any]:
    from mcp_server.config import WATCHLIST_PATH

    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tickers": [], "themes": []}


def watchlist(
    action: Literal["get", "update"] = "get",
    tickers: list[str] | None = None,
    themes: list[str] | None = None,
) -> dict[str, Any]:
    """스케줄러가 추적하는 워치리스트 조회(get) 또는 교체(update: 전달한 목록으로 덮어씀)."""
    from mcp_server.config import WATCHLIST_PATH

    data = _read_watchlist()
    if action == "get":
        return data
    if tickers is None and themes is None:
        raise ToolError("update에는 tickers 또는 themes가 필요합니다.")
    if tickers is not None:
        data["tickers"] = [t.strip().upper() for t in tickers if t.strip()]
    if themes is not None:
        data["themes"] = themes
    data["updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


TOOLS: list[ToolSpec] = [
    (portfolio_analyze, READ_EXTERNAL),
    (portfolio_quick_check, READ_EXTERNAL),
    (portfolio_store, WRITE_LOCAL),
    (watchlist, WRITE_LOCAL),
]
