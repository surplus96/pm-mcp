from __future__ import annotations
from datetime import datetime
from typing import List
from mcp_server.tools.portfolio import evaluate_holdings
from mcp_server.tools.analytics import rank_tickers_with_fundamentals
from mcp_server.tools.reports import generate_report


def run_portfolio_report(tickers: List[str]) -> str:
    evals = evaluate_holdings(tickers)
    # 펀더멘털 랭킹으로 점수 채우기 + 페이즈/ret20 병합
    ranked = rank_tickers_with_fundamentals(tickers, dip_weight=0.12, use_dip_bonus=True)
    rmap = {r["ticker"]: r for r in ranked}
    scores = []
    for e in evals:
        t = e["ticker"]
        base = rmap.get(t, {}).get("base_score", 0.0)
        dip = rmap.get(t, {}).get("dip_bonus", 0.0)
        total = rmap.get(t, {}).get("score", float(base) + float(dip))
        item = {
            "ticker": t,
            "base_score": round(float(base), 4),
            "dip_bonus": round(float(dip), 4),
            "score": round(float(total), 4),
            "phase": e.get("phase"),
            "ret20": e.get("ret20"),
        }
        scores.append(item)

    payload = {
        "title": "Portfolio Phase Report",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "tickers": tickers,
        "summary": "Phase signals for current holdings.",
        "news_summary": "",
        "filings_summary": "",
        "scores": scores,
    }
    return generate_report(payload)
