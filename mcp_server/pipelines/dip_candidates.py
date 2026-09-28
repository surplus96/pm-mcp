from __future__ import annotations
from typing import List, Optional, Dict
import os, csv
from datetime import datetime

from mcp_server.config import PROCESSED_PATH, safe_filename
from mcp_server.tools.analytics import rank_tickers_with_fundamentals
from mcp_server.tools.market_data import get_prices
from mcp_server.tools.renderer import render_price_chart
from mcp_server.tools.interaction import propose_tickers


def _drawdown_180d(ticker: str) -> Optional[float]:
    try:
        df = get_prices(ticker)
        if df.empty or 'Close' not in df.columns:
            return None
        closes = df['Close']
        recent = closes.tail(180)
        if recent.empty:
            recent = closes
        rh = float(recent.max()) if len(recent) else None
        last = float(closes.iloc[-1]) if len(closes) else None
        if not rh or rh <= 0 or not last:
            return None
        return max(0.0, (rh - last) / rh)
    except Exception:
        return None


def run_dip_candidates(
    theme: str,
    tickers: Optional[List[str]] = None,
    top_n: int = 5,
    drawdown_min: float = 0.2,
    ret10_min: float = 0.0,
    event_min: float = 0.5,
    save: bool = True,
) -> Dict:
    tickers = tickers or propose_tickers(theme)
    ranked = rank_tickers_with_fundamentals(tickers)
    enriched = []
    for r in ranked:
        dd = _drawdown_180d(r['ticker'])
        r['drawdown180'] = dd
        enriched.append(r)
    # 필터: 딥, 모멘텀(간접: mom3), 이벤트
    filtered = [r for r in enriched if (r.get('drawdown180') or 0) >= drawdown_min and (r.get('mom3') or 0) >= ret10_min and (r.get('eventScore') or 0) >= event_min]
    top = filtered[:top_n] if filtered else ranked[:top_n]

    # 저장 경로
    date_str = datetime.now().strftime('%Y-%m-%d')
    theme_dir = os.path.join(PROCESSED_PATH, 'dip')
    os.makedirs(theme_dir, exist_ok=True)
    csv_path = os.path.join(theme_dir, f"dip_candidates_{safe_filename(theme)}_{date_str}.csv")

    # CSV 저장(핵심 열만)
    if save:
        cols = ['ticker','sector','score','base_score','dip_bonus','valuation','growth','profitability','quality','drawdown180','mom1','mom3','mom6','mom12','pe','pb','eps','returnOnEquity','revenueGrowth','profitMargins','eventScore']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in top:
                w.writerow([r.get(c) for c in cols])

    # 차트 이미지 생성 링크만 반환
    image_paths = []
    for r in top:
        try:
            img = render_price_chart(r['ticker'], days=180)
            image_paths.append(img)
        except Exception:
            pass

    # 요약 마크다운(요약 + 링크)
    md = [f"# Dip Candidates - {theme}", "", "## Top Candidates"]
    for i, r in enumerate(top, 1):
        md.append(f"{i}. {r['ticker']} — score={r['score']:.3f}, dip={r['dip_bonus']:.3f}, dd180={ (r.get('drawdown180') or 0):.2f}")
    md.append("")
    md.append("## Links")
    if save:
        md.append(f"- CSV: {csv_path}")
    for p in image_paths:
        md.append(f"- ![chart]({p})")
    # Claude 반환은 링크/요약만
    return {
        "theme": theme,
        "top": [{"ticker": r['ticker'], "score": r['score'], "dd180": r.get('drawdown180'), "event": r.get('eventScore')} for r in top],
        "csv_path": csv_path if save else None,
        "report_md": "\n".join(md),
        "images": image_paths,
    }
