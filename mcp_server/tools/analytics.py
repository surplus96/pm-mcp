from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from .market_data import get_fundamentals_snapshot, get_momentum_metrics
from .yf_utils import normalize_yf_columns
from mcp_server.config import SCORE_WEIGHTS, SCORE_SECTOR_NEUTRAL, SECTOR_FACTOR_WEIGHTS

def _parse_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {"growth":0.25,"profitability":0.25,"valuation":0.25,"quality":0.25}
    try:
        for part in s.split(','):
            if '=' in part:
                k,v = part.split('=',1)
                out[k.strip()] = float(v)
    except Exception:
        pass
    return out

DEFAULT_WEIGHTS = _parse_weights(SCORE_WEIGHTS)


def _to_float(x) -> float:
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return 0.0


def compute_dip_bonus_by_prices(ticker: str, lookback_days: int = 180) -> float:
    try:
        hist = normalize_yf_columns(
            yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        )
        if hist.empty:
            return 0.0
        closes = hist["Close"]
        window = closes.tail(min(len(closes), lookback_days))
        recent_high = _to_float(window.max())
        last = _to_float(closes.iloc[-1])
        ret10 = _to_float(closes.pct_change(10).iloc[-1])
        if recent_high <= 0:
            return 0.0
        drawdown = max(0.0, (recent_high - last) / recent_high)
        dd_score = min(drawdown / 0.30, 1.0)
        mom_score = max(0.0, min((ret10 + 0.05) / 0.10, 1.0))
        bonus = 0.5 * dd_score + 0.5 * (dd_score * mom_score)
        return round(bonus, 4)
    except Exception:
        return 0.0


def _rank_normalized(values: List[Optional[float]], higher_is_better: bool = True) -> List[float]:
    idx_vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not idx_vals:
        return [0.5] * len(values)
    sorted_idx = sorted(idx_vals, key=lambda x: x[1], reverse=higher_is_better)
    ranks = [None] * len(values)
    n = len(sorted_idx)
    for rank_pos, (i, _) in enumerate(sorted_idx):
        ranks[i] = (n - rank_pos - 1) / (n - 1) if n > 1 else 0.5
    return [r if r is not None else 0.5 for r in ranks]


def _rank_normalized_by_group(values: List[Optional[float]], groups: List[Optional[str]], higher_is_better: bool = True) -> List[float]:
    # 그룹(섹터)별로 0..1 정규화 후 전체 리스트 순서대로 결합
    from collections import defaultdict
    group_to_indices = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)
    out = [0.5] * len(values)
    for g, idxs in group_to_indices.items():
        sub_vals = [values[i] for i in idxs]
        sub_scores = _rank_normalized(sub_vals, higher_is_better=higher_is_better)
        for j, i in enumerate(idxs):
            out[i] = sub_scores[j]
    return out


def _combine_scores(parts: List[Tuple[float, float]]) -> float:
    # parts: [(score, weight)]
    num = sum(s * w for s, w in parts)
    den = sum(w for _, w in parts) or 1.0
    return round(num / den, 4)


def _parse_sector_weights(s: Optional[str]) -> Dict[str, Dict[str, float]]:
    import json
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def rank_tickers_with_fundamentals(
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
    dip_weight: float = 0.12,
    use_dip_bonus: bool = True,
) -> List[Dict]:
    """PER/PBR/EPS 등 펀더멘털을 활용해 growth/profitability/valuation/quality를 산출 후 랭킹.
    - valuation: 낮을수록 좋은 PER, PBR (역순 정규화)
    - growth: 높은 EPS(또는 EPS 성장 대용), revenueGrowth
    - profitability: profitMargins, returnOnEquity
    - quality: profitability와 동일 기준을 기본으로 사용(간단화)
    """
    weights = weights or DEFAULT_WEIGHTS
    fundamentals = [get_fundamentals_snapshot(t) for t in tickers]
    momentum = [get_momentum_metrics(t) for t in tickers]
    sector_weights_map = _parse_sector_weights(SECTOR_FACTOR_WEIGHTS)

    pe = [f.get("pe") for f in fundamentals]
    pb = [f.get("pb") for f in fundamentals]
    eps = [f.get("eps") for f in fundamentals]
    rev_g = [f.get("revenueGrowth") for f in fundamentals]
    pm = [f.get("profitMargins") for f in fundamentals]
    roe = [f.get("returnOnEquity") for f in fundamentals]
    sectors = [f.get("sector") for f in fundamentals]

    # Normalize to 0..1 (optionally sector-neutral)
    if SCORE_SECTOR_NEUTRAL:
        val_pe = _rank_normalized_by_group(pe, sectors, higher_is_better=False)
        val_pb = _rank_normalized_by_group(pb, sectors, higher_is_better=False)
        grow_eps = _rank_normalized_by_group(eps, sectors, higher_is_better=True)
        grow_rev = _rank_normalized_by_group(rev_g, sectors, higher_is_better=True)
        prof_pm = _rank_normalized_by_group(pm, sectors, higher_is_better=True)
        prof_roe = _rank_normalized_by_group(roe, sectors, higher_is_better=True)
    else:
        val_pe = _rank_normalized(pe, higher_is_better=False)
        val_pb = _rank_normalized(pb, higher_is_better=False)
        grow_eps = _rank_normalized(eps, higher_is_better=True)
        grow_rev = _rank_normalized(rev_g, higher_is_better=True)
        prof_pm = _rank_normalized(pm, higher_is_better=True)
        prof_roe = _rank_normalized(roe, higher_is_better=True)
    valuation = [_combine_scores([(val_pe[i], 0.6), (val_pb[i], 0.4)]) for i in range(len(tickers))]

    grow_eps = _rank_normalized(eps, higher_is_better=True)
    grow_rev = _rank_normalized(rev_g, higher_is_better=True)
    growth = [_combine_scores([(grow_eps[i], 0.5), (grow_rev[i], 0.5)]) for i in range(len(tickers))]

    prof_pm = _rank_normalized(pm, higher_is_better=True)
    prof_roe = _rank_normalized(roe, higher_is_better=True)
    profitability = [_combine_scores([(prof_pm[i], 0.5), (prof_roe[i], 0.5)]) for i in range(len(tickers))]

    # Momentum composite (higher better)
    mom_raw = [
        ((m.get("mom3") or 0.0) + (m.get("mom6") or 0.0) + (m.get("mom12") or 0.0)) / 3.0 if isinstance(m, dict) else 0.0
        for m in momentum
    ]
    mom_score = _rank_normalized(mom_raw, higher_is_better=True)

    # Event score (EDGAR 제목 키워드)
    try:
        from .filings import keyword_event_score
        ev_raw = [keyword_event_score(t) for t in tickers]
    except Exception:
        ev_raw = [0.5] * len(tickers)
    ev_score = _rank_normalized(ev_raw, higher_is_better=True)

    # Quality: profitability 50% + momentum 30% + event 20%
    quality = [_combine_scores([(profitability[i], 0.5), (mom_score[i], 0.3), (ev_score[i], 0.2)]) for i in range(len(tickers))]

    ranked: List[Dict] = []
    for i, t in enumerate(tickers):
        sector = fundamentals[i].get("sector")
        w = dict(weights)
        if sector and isinstance(sector_weights_map.get(sector), dict):
            w.update({k: float(v) for k, v in sector_weights_map[sector].items() if k in w})
        base = (
            float(valuation[i]) * w.get("valuation", 0.25)
            + float(growth[i]) * w.get("growth", 0.25)
            + float(profitability[i]) * w.get("profitability", 0.25)
            + float(quality[i]) * w.get("quality", 0.25)
        )
        dip_bonus = 0.0
        if use_dip_bonus:
            dip_bonus = dip_weight * compute_dip_bonus_by_prices(t)
        item = {
            "ticker": t,
            "sector": sector,
            "valuation": round(valuation[i], 4),
            "growth": round(growth[i], 4),
            "profitability": round(profitability[i], 4),
            "quality": round(quality[i], 4),
            "dip_bonus": round(dip_bonus, 4),
            "base_score": round(base, 4),
        }
        item["score"] = round(item["base_score"] + item["dip_bonus"], 4)
        # raw metrics for transparency
        m = momentum[i] if isinstance(momentum[i], dict) else {}
        item.update({
            "pe": pe[i], "pb": pb[i], "eps": eps[i],
            "revenueGrowth": rev_g[i], "profitMargins": pm[i], "returnOnEquity": roe[i],
            "mom1": m.get("mom1"), "mom3": m.get("mom3"), "mom6": m.get("mom6"), "mom12": m.get("mom12"),
            "mom": ((m.get("mom3") or 0.0) + (m.get("mom6") or 0.0) + (m.get("mom12") or 0.0)) / 3.0 if m else None,
            "eventScore": ev_raw[i]
        })
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ===== 비동기 버전 함수들 =====

async def rank_tickers_with_fundamentals_async(
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
    dip_weight: float = 0.12,
    use_dip_bonus: bool = True,
    max_concurrent: int = 5,
) -> List[Dict]:
    """rank_tickers_with_fundamentals의 비동기 버전 - 병렬 데이터 수집으로 속도 향상

    Args:
        tickers: 분석할 티커 리스트
        weights: 팩터 가중치
        dip_weight: 딥 보너스 가중치
        use_dip_bonus: 딥 보너스 사용 여부
        max_concurrent: 최대 동시 요청 수

    Returns:
        랭킹된 티커 리스트 (score 내림차순)
    """
    from mcp_server.tools.async_utils import parallel_map

    weights = weights or DEFAULT_WEIGHTS
    sector_weights_map = _parse_sector_weights(SECTOR_FACTOR_WEIGHTS)

    # 병렬로 데이터 수집
    fundamentals, momentum = await asyncio.gather(
        parallel_map(get_fundamentals_snapshot, tickers, max_concurrent),
        parallel_map(get_momentum_metrics, tickers, max_concurrent),
    )

    # 이벤트 스코어 병렬 수집
    try:
        from .filings import keyword_event_score
        ev_raw = await parallel_map(keyword_event_score, tickers, max_concurrent=3)
    except Exception:
        ev_raw = [0.5] * len(tickers)

    # 딥 보너스 병렬 계산
    if use_dip_bonus:
        dip_bonuses = await parallel_map(compute_dip_bonus_by_prices, tickers, max_concurrent)
    else:
        dip_bonuses = [0.0] * len(tickers)

    # 나머지 계산 로직 (동기 - CPU bound)
    pe = [f.get("pe") for f in fundamentals]
    pb = [f.get("pb") for f in fundamentals]
    eps = [f.get("eps") for f in fundamentals]
    rev_g = [f.get("revenueGrowth") for f in fundamentals]
    pm = [f.get("profitMargins") for f in fundamentals]
    roe = [f.get("returnOnEquity") for f in fundamentals]
    sectors = [f.get("sector") for f in fundamentals]

    # Normalize to 0..1
    if SCORE_SECTOR_NEUTRAL:
        val_pe = _rank_normalized_by_group(pe, sectors, higher_is_better=False)
        val_pb = _rank_normalized_by_group(pb, sectors, higher_is_better=False)
    else:
        val_pe = _rank_normalized(pe, higher_is_better=False)
        val_pb = _rank_normalized(pb, higher_is_better=False)

    valuation = [_combine_scores([(val_pe[i], 0.6), (val_pb[i], 0.4)]) for i in range(len(tickers))]

    grow_eps = _rank_normalized(eps, higher_is_better=True)
    grow_rev = _rank_normalized(rev_g, higher_is_better=True)
    growth = [_combine_scores([(grow_eps[i], 0.5), (grow_rev[i], 0.5)]) for i in range(len(tickers))]

    prof_pm = _rank_normalized(pm, higher_is_better=True)
    prof_roe = _rank_normalized(roe, higher_is_better=True)
    profitability = [_combine_scores([(prof_pm[i], 0.5), (prof_roe[i], 0.5)]) for i in range(len(tickers))]

    # Momentum composite
    mom_raw = [
        ((m.get("mom3") or 0.0) + (m.get("mom6") or 0.0) + (m.get("mom12") or 0.0)) / 3.0 if isinstance(m, dict) else 0.0
        for m in momentum
    ]
    mom_score = _rank_normalized(mom_raw, higher_is_better=True)
    ev_score = _rank_normalized(ev_raw, higher_is_better=True)

    # Quality
    quality = [_combine_scores([(profitability[i], 0.5), (mom_score[i], 0.3), (ev_score[i], 0.2)]) for i in range(len(tickers))]

    # 최종 랭킹 계산
    ranked: List[Dict] = []
    for i, t in enumerate(tickers):
        sector = fundamentals[i].get("sector")
        w = dict(weights)
        if sector and isinstance(sector_weights_map.get(sector), dict):
            w.update({k: float(v) for k, v in sector_weights_map[sector].items() if k in w})

        base = (
            float(valuation[i]) * w.get("valuation", 0.25)
            + float(growth[i]) * w.get("growth", 0.25)
            + float(profitability[i]) * w.get("profitability", 0.25)
            + float(quality[i]) * w.get("quality", 0.25)
        )

        dip_bonus = dip_weight * dip_bonuses[i] if use_dip_bonus else 0.0

        m = momentum[i] if isinstance(momentum[i], dict) else {}
        item = {
            "ticker": t,
            "sector": sector,
            "valuation": round(valuation[i], 4),
            "growth": round(growth[i], 4),
            "profitability": round(profitability[i], 4),
            "quality": round(quality[i], 4),
            "dip_bonus": round(dip_bonus, 4),
            "base_score": round(base, 4),
            "score": round(base + dip_bonus, 4),
            "pe": pe[i], "pb": pb[i], "eps": eps[i],
            "revenueGrowth": rev_g[i], "profitMargins": pm[i], "returnOnEquity": roe[i],
            "mom1": m.get("mom1"), "mom3": m.get("mom3"), "mom6": m.get("mom6"), "mom12": m.get("mom12"),
            "mom": ((m.get("mom3") or 0.0) + (m.get("mom6") or 0.0) + (m.get("mom12") or 0.0)) / 3.0 if m else None,
            "eventScore": ev_raw[i]
        }
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# asyncio import (비동기 함수용)
import asyncio
