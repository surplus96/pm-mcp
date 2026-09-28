from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math
import yfinance as yf

from mcp_server.tools.yf_utils import normalize_yf_columns
from mcp_server.config import (
    PRESENT_THEME_CHART_DAYS,
    PRESENT_PORTFOLIO_HISTORY_DAYS,
    PRESENT_YSCALE,
    PRESENT_MA_WINDOWS,
    PRESENT_COLORS,
    PRESENT_NEWS_MAX,
    PRESENT_FILINGS_MAX,
)
from .news_search import search_news
from .filings import fetch_recent_filings
from .analytics import rank_tickers_with_fundamentals
from .portfolio import evaluate_holdings
from .renderer import render_multi_price_chart

_SPARKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float]) -> str:
    if not values:
        return ""
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return _SPARKS[len(_SPARKS)//2] * len(values)
    out = []
    for v in values:
        pos = (v - vmin) / (vmax - vmin) if vmax != vmin else 0
        idx = min(len(_SPARKS) - 1, max(0, int(round(pos * (len(_SPARKS) - 1)))))
        out.append(_SPARKS[idx])
    return "".join(out)


def _mk_table(headers: List[str], rows: List[List[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---" for _ in headers]) + "|"
    body = ["| " + " | ".join(str(c) if c is not None else "" for c in r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def present_theme_overview(
    theme: str,
    tickers: List[str],
    lookback_days: int = 7,
    max_items: int = PRESENT_NEWS_MAX,
    with_images: bool = False,
    chart_days: int = PRESENT_THEME_CHART_DAYS,
    yscale: str = PRESENT_YSCALE,
    ma_windows: Tuple[int, ...] = PRESENT_MA_WINDOWS,
    colors: Optional[List[str]] = PRESENT_COLORS,
) -> str:
    queries = [f"{theme} stocks", f"{theme} demand", f"{theme} regulation"]
    news = search_news(queries, lookback_days=lookback_days, max_results=max_items)
    news_lines = []
    for n in news:
        for h in n.get("hits", [])[:max_items]:
            title = h.get('title') or ''
            src = h.get('source') or ''
            url = h.get('url') or ''
            news_lines.append(f"- {title} ({src}) — {url}")

    filings_all: List[Dict] = []
    for t in tickers:
        filings_all.extend(fetch_recent_filings(t, forms=["8-K","10-Q","10-K"], limit=PRESENT_FILINGS_MAX))
    filing_lines = [f"- {f.get('form')} | {f.get('filingDate')} | {f.get('title') or ''} — {f.get('url')}" for f in filings_all[:PRESENT_FILINGS_MAX]]

    # Fundamentals + momentum + events 기반 랭킹
    ranked = rank_tickers_with_fundamentals(tickers, dip_weight=0.12, use_dip_bonus=True)
    rows = [[
        r["ticker"],
        f"{r.get('base_score',0):.3f}",
        f"{r.get('dip_bonus',0):.3f}",
        f"{r.get('score',0):.3f}",
        r.get("pe", ""),
        r.get("pb", ""),
        r.get("eps", ""),
    ] for r in ranked]

    images = []
    if with_images:
        images.append(render_multi_price_chart(tickers, days=chart_days, yscale=yscale, ma_windows=ma_windows, colors=colors))
    md = []
    md.append(f"## {theme} Theme Overview")
    if images:
        for p in images:
            md.append(f"![chart]({p})")
    md.append("")
    md.append("### Top News")
    md.extend(news_lines or ["_No news found_"])
    md.append("")
    md.append("### Recent SEC Filings")
    md.extend(filing_lines or ["_No filings found_"])
    md.append("")
    md.append("### Scores (with key metrics)")
    md.append(_mk_table(["Ticker","Base","Dip","Total","PE","PB","EPS"], rows))
    return "\n".join(md)


def present_portfolio_overview(
    tickers: List[str],
    history_days: int = PRESENT_PORTFOLIO_HISTORY_DAYS,
    with_images: bool = False,
    yscale: str = PRESENT_YSCALE,
    ma_windows: Tuple[int, ...] = (),
    colors: Optional[List[str]] = PRESENT_COLORS,
) -> str:
    evals = evaluate_holdings(tickers)
    rows = []
    for e in evals:
        t = e["ticker"]
        hist = normalize_yf_columns(
            yf.download(t, period=f"{history_days}d", interval="1d", progress=False, auto_adjust=True)
        )
        if not hist.empty and "Close" in hist.columns:
            close_series = hist["Close"]
            try:
                values = list(close_series.tolist())
            except Exception:
                values = list(close_series.values) if hasattr(close_series, "values") else []
            values = values[-min(len(values), history_days):]
        else:
            values = []
        rows.append([t, e.get("phase"), e.get("ret20"), _sparkline(values)])
    images = []
    if with_images:
        images.append(render_multi_price_chart(tickers, days=history_days, yscale=yscale, ma_windows=ma_windows, colors=colors))
    md = []
    md.append("## Portfolio Overview")
    if images:
        for p in images:
            md.append(f"![chart]({p})")
    md.append("")
    md.append(_mk_table(["Ticker","Phase","ret20","Trend"], rows))
    return "\n".join(md)
