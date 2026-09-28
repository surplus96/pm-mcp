from __future__ import annotations
from datetime import datetime
from typing import List
from mcp_server.tools.news_search import search_news
from mcp_server.tools.analytics import rank_tickers_with_fundamentals
from mcp_server.tools.reports import generate_report
from mcp_server.tools.llm import summarize_items
from mcp_server.tools.filings import fetch_recent_filings, summarize_filings_items


def _fallback_bullets(lines: List[str]) -> str:
    if not lines:
        return ""
    return "\n".join(f"- {ln}" for ln in lines)


def run_theme_report(theme: str, tickers: List[str]) -> str:
    queries = [f"{theme} stocks", f"{theme} demand", f"{theme} regulation"]
    news = search_news(queries, lookback_days=7, max_results=5)
    news_lines: List[str] = []
    for n in news:
        for h in n.get("hits", [])[:3]:
            title = h.get('title') or ''
            src = h.get('source') or ''
            url = h.get('url') or ''
            news_lines.append(f"{title} | {src} | {url}")
    news_summary = summarize_items(news_lines, max_sentences=6) if news_lines else ""
    if not news_summary:
        news_summary = _fallback_bullets(news_lines)

    filings_all = []
    for t in tickers:
        filings_all.extend(fetch_recent_filings(t, forms=["8-K", "10-Q", "10-K"], limit=3))
    filings_summary = summarize_filings_items(filings_all, max_items=6)
    if not filings_summary and filings_all:
        filing_lines = [f"{f.get('form')} | {f.get('filingDate')} | {f.get('title') or ''} | {f.get('url')}" for f in filings_all[:6]]
        filings_summary = _fallback_bullets(filing_lines)

    ranked = rank_tickers_with_fundamentals(tickers, dip_weight=0.12, use_dip_bonus=True)

    payload = {
        "title": f"{theme} Theme Snapshot",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "tickers": tickers,
        "summary": f"Condensed news and ranking for {theme}.",
        "news_summary": news_summary,
        "filings_summary": filings_summary,
        "scores": ranked,
    }
    return generate_report(payload)
