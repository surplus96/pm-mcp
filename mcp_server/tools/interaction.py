from __future__ import annotations
from typing import List, Dict

from .news_search import search_news
from .analytics import rank_tickers_with_fundamentals
from .filings import fetch_recent_filings
import yfinance as yf

ETF_MAP = {
    "AI": ["BOTZ","AIQ"],
    "gen ai": ["BOTZ","AIQ"],
    "machine learning": ["BOTZ","AIQ"],
    "semiconductor": ["SMH","SOXX"],
    "semis": ["SMH","SOXX"],
    "cloud": ["CLOU","WCLD"],
    "cybersecurity": ["HACK","CIBR"],
    "renewable energy": ["ICLN","TAN"],
    "solar": ["TAN","ICLN"],
    "wind": ["FAN","ICLN"],
    "biotech": ["XBI","IBB"],
    "healthcare": ["XLV","VHT","PPH"],
    "pharma": ["PPH","IHE"],
    "pharmaceuticals": ["PPH","IHE"],
    "fintech": ["FINX","ARKF","IPAY"],
    "payments": ["IPAY","FINX","ARKF"],
    "payment processing": ["IPAY","FINX","ARKF"],
    "digital banking": ["FINX","ARKF","IPAY"],
    "neobank": ["FINX","ARKF","IPAY"],
    "EV": ["DRIV","KARS"],
    # synonyms
    "rare disease": ["XBI","IBB","PPH"],
    "orphan drugs": ["XBI","IBB","PPH"],
}

# Thematic seed lists (domain-curated fallbacks)
THEME_SEEDS: Dict[str, List[str]] = {
    "AI": ["NVDA","MSFT","GOOGL","AMD","AVGO","META","AMZN"],
    "gen ai": ["NVDA","MSFT","GOOGL","AMD","AVGO","META","AMZN"],
    "machine learning": ["NVDA","MSFT","GOOGL","AMD","AVGO","META","AMZN"],
    "semiconductor": ["NVDA","AMD","AVGO","TSM","ASML","MU","TXN","AMAT","LRCX"],
    "semis": ["NVDA","AMD","AVGO","TSM","ASML","MU","TXN","AMAT","LRCX"],
    "cloud": ["AMZN","MSFT","GOOGL","CRM","NOW","SNOW","MDB","DDOG"],
    "cybersecurity": ["PANW","FTNT","CRWD","ZS","OKTA","S","SPLK"],
    "renewable energy": ["ENPH","FSLR","NEE","RUN","SEDG"],
    "solar": ["ENPH","FSLR","RUN","SEDG"],
    "wind": ["VWS","ENPH","NEE"],
    "biotech": ["VRTX","REGN","AMGN","GILD","SRPT","BIIB","LLY"],
    "healthcare": ["UNH","JNJ","PFE","MRK","ABT","ABBV","TMO","LLY"],
    "pharma": ["LLY","JNJ","PFE","MRK","ABBV","BMY","AZN","NVO"],
    "pharmaceuticals": ["LLY","JNJ","PFE","MRK","ABBV","BMY","AZN","NVO"],
    "rare disease": ["VRTX","BMRN","REGN","SRPT","RARE","IONS","ALNY","FOLD","KRYS","PTCT","UTHR"],
    "orphan drugs": ["VRTX","BMRN","REGN","SRPT","RARE","IONS","ALNY","FOLD","KRYS","PTCT","UTHR"],
    "fintech": ["V","MA","PYPL","SQ","GPN","FIS","FI","AFRM","SOFI","NU","COIN","GDRX","FOUR","TOST","UPST"],
    "payments": ["V","MA","PYPL","SQ","GPN","FIS","FI","AFRM","FOUR","TOST"],
    "digital banking": ["SOFI","NU","ALLY","AX","LYG","SCHW","HOOD"],
}

# Category-specific absolute fallbacks
DEFAULT_FALLBACK_BY_KEY: Dict[str, List[str]] = {
    "AI": ["NVDA","MSFT","GOOGL","AMD","AVGO","META","AMZN"],
    "semiconductor": ["NVDA","AMD","AVGO","TSM","ASML","MU","TXN"],
    "cloud": ["AMZN","MSFT","GOOGL","CRM","SNOW","NOW","MDB"],
    "cybersecurity": ["PANW","CRWD","FTNT","ZS"],
    "renewable energy": ["ENPH","FSLR","NEE"],
    "biotech": ["VRTX","REGN","AMGN","GILD","SRPT"],
    "healthcare": ["UNH","JNJ","LLY","ABT","TMO"],
    "pharma": ["LLY","JNJ","PFE","MRK","ABBV","AZN"],
    "fintech": ["V","MA","PYPL","SQ","GPN","FIS","FI","AFRM","SOFI","NU"],
    "payments": ["V","MA","PYPL","SQ","GPN","FIS","FI"],
    "digital banking": ["SOFI","NU","ALLY","AX","SCHW"],
    "EV": ["TSLA","RIVN","NIO","LI","BYDDF"],
    "rare disease": ["VRTX","BMRN","REGN","SRPT","RARE","IONS","ALNY","FOLD","KRYS","PTCT","UTHR"],
}


def _top_holdings(etf: str, max_n: int = 10) -> List[tuple[str,float]]:
    """ETF 보유종목과 비중(가능 시)을 반환 [(symbol, weight0..1)]. 비중 미제공시 1.0/순위 부여.
    """
    try:
        tk = yf.Ticker(etf)
        df = getattr(tk, 'fund_holdings', None)
        if df is not None and hasattr(df, 'head'):
            syms = list((df.get('symbol') or [])[:max_n])
            w = list((df.get('holdingPercent') or [])[:max_n])
            out = []
            for i, s in enumerate(syms):
                if not s:
                    continue
                weight = float(w[i]) if i < len(w) and w[i] is not None else None
                out.append((str(s), weight if weight is not None else 1.0))
            if out:
                return out
        hd = getattr(tk, 'holdings', None)
        if isinstance(hd, dict):
            comps = (hd.get('holdings') or [])[:max_n]
            out = []
            for c in comps:
                s = c.get('symbol')
                if not s:
                    continue
                weight = c.get('holdingPercent')
                out.append((str(s), float(weight) if weight is not None else 1.0))
            if out:
                return out
    except Exception:
        pass
    return []


def propose_themes(lookback_days: int = 7, max_themes: int = 5) -> List[str]:
    candidates = list(ETF_MAP.keys())
    density = []
    for c in candidates:
        res = search_news([f"{c} stocks"], lookback_days=lookback_days, max_results=5)
        hits = sum(len(x.get("hits", []) or []) for x in res)
        density.append((c, hits))
    density.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in density[:max_themes]]


def _theme_key(theme: str) -> str:
    t = (theme or "").strip().lower()
    # Korean synonyms handling
    if "희귀" in theme or "희귀질환" in theme or "고아의약" in theme or "고아 의약" in theme:
        return "rare disease"
    if "바이오" in theme or "생명공학" in theme:
        return "biotech"
    if "제약" in theme or "의약" in theme:
        return "pharma"
    if "핀테크" in theme or "결제" in theme or "디지털 뱅킹" in theme or "네오뱅크" in theme:
        return "fintech"
    if "반도체" in theme or "칩" in theme:
        return "semiconductor"
    if "클라우드" in theme or "saas" in t:
        return "cloud"
    if "사이버" in theme or "보안" in theme:
        return "cybersecurity"
    if "재생에너지" in theme or "태양광" in theme or "풍력" in theme or "그린" in theme:
        return "renewable energy"
    if "전기차" in theme or "배터리" in theme:
        return "EV"
    if "헬스케어" in theme or "의료" in theme or "메드텍" in theme:
        return "healthcare"
    if "인공지능" in theme or "생성형" in theme:
        return "AI"
    if "rare" in t or "orphan" in t:
        return "rare disease"
    if "biotech" in t or "bio" in t:
        return "biotech"
    if "pharma" in t or "pharmaceutical" in t:
        return "pharma"
    if ("fintech" in t
        or "payment" in t
        or "payments" in t
        or "processing" in t
        or "digital bank" in t
        or "digital banking" in t
        or "neobank" in t
        or "neo bank" in t):
        return "fintech"
    if "semiconductor" in t or "semis" in t or "chip" in t:
        return "semiconductor"
    if "cloud" in t or "saas" in t:
        return "cloud"
    if "cyber" in t or "security" in t:
        return "cybersecurity"
    if "renewable" in t or "green" in t or "solar" in t or "wind" in t:
        return "renewable energy"
    if "ev" in t or "electric vehicle" in t or "battery" in t:
        return "EV"
    if "healthcare" in t or "health care" in t or "medtech" in t:
        return "healthcare"
    if "ai" in t or "gen ai" in t or "machine learning" in t:
        return "AI"
    return t


def propose_tickers(theme: str) -> List[str]:
    key = _theme_key(theme)
    t = (theme or "").strip().lower()
    # 1) direct key map
    etfs = list(ETF_MAP.get(key, []))
    # 2) substring-based match across known ETF keys (e.g., "payment processing digital banking")
    if not etfs:
        seen = set()
        for k, v in ETF_MAP.items():
            if k in t:
                for e in v:
                    if e not in seen:
                        etfs.append(e)
                        seen.add(e)
    # 3) generic fallbacks by original strings
    if not etfs:
        etfs = ETF_MAP.get(theme, ETF_MAP.get(theme.lower(), []))
    score: dict[str,float] = {}
    order: dict[str,int] = {}
    for e in etfs:
        holdings = _top_holdings(e, max_n=15)
        for rank, (sym, w) in enumerate(holdings):
            # 비중 가중 + 순위 감쇠(상위 종목 가중)
            add = (w or 1.0) * (1.0 / (1 + rank))
            score[sym] = score.get(sym, 0.0) + add
            if sym not in order:
                order[sym] = len(order)
    # Add curated seeds when available (ensures domain relevance even if ETF holdings absent)
    # Merge multiple possible seed lists if multiple keywords present
    seed_keys = [key]
    for k in ["fintech","payments","digital banking","rare disease","orphan drugs","biotech","pharma","AI","semiconductor","cloud","cybersecurity","renewable energy","EV","healthcare"]:
        if k in t and k not in seed_keys:
            seed_keys.append(k)
    for sk in seed_keys:
        for s in THEME_SEEDS.get(sk, []):
            if s not in score:
                score[s] = score.get(s, 0.0) + 0.5  # small base weight for seeds
                if s not in order:
                    order[s] = len(order)
    if score:
        # 점수 내림차순, 동점은 등장 순서 유지
        sorted_syms = sorted(score.keys(), key=lambda s: (-score[s], order[s]))
        return [sym for sym in sorted_syms if sym][:10]
    # Absolute fallback: use category-specific list when available; else neutral large-cap tech basket
    if key in DEFAULT_FALLBACK_BY_KEY:
        return DEFAULT_FALLBACK_BY_KEY[key][:10]
    # last resort, neutral popular large-cap tickers (avoid biotech bias)
    return ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","TSLA","CRM","AMD"]


def analyze_selection(theme: str, tickers: List[str]) -> str:
    ranked = rank_tickers_with_fundamentals(tickers, dip_weight=0.12, use_dip_bonus=True)
    lines = ["## Analysis Summary"]
    lines.append("\n### Rank (Top 5)")
    for r in ranked[:5]:
        lines.append(
            f"- {r['ticker']}: score={r['score']:.3f} (base={r['base_score']:.3f}, dip={r['dip_bonus']:.3f}) "
            f"| PE={r.get('pe')}, PB={r.get('pb')}, EPS={r.get('eps')}"
        )
    lines.append("\n### Recent Filings Counts")
    for t in tickers[:5]:
        f = fetch_recent_filings(t, limit=5)
        lines.append(f"- {t}: {len(f)} filings in recent submissions")
    lines.append("\n> Next: call report_create(kind='theme_overview', with_images=True) for charts and a full overview.")
    return "\n".join(lines)
