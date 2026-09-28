"""
고급 랭킹 엔진 - 섹터별 가중치, 시장 상황 반영, Z-score 정규화

Features:
- 섹터별 동적 가중치 (Technology vs Utilities 등 다른 기준 적용)
- 6개 팩터: growth, profitability, valuation, quality, momentum, volatility
- Z-score 정규화 + 윈저화 (이상치 처리)
- 시장 상황(강세/약세) 반영 가중치 조정
- 섹터 내 상대 비교 옵션
"""
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import numpy as np
from mcp_server.tools.yf_utils import normalize_yf_columns

logger = logging.getLogger(__name__)


# ===== 섹터별 가중치 설정 =====
SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Technology": {
        "growth": 0.30,
        "profitability": 0.20,
        "valuation": 0.15,
        "quality": 0.15,
        "momentum": 0.15,
        "volatility": 0.05,
    },
    "Communication Services": {
        "growth": 0.28,
        "profitability": 0.22,
        "valuation": 0.18,
        "quality": 0.15,
        "momentum": 0.12,
        "volatility": 0.05,
    },
    "Consumer Cyclical": {
        "growth": 0.25,
        "profitability": 0.22,
        "valuation": 0.18,
        "quality": 0.15,
        "momentum": 0.15,
        "volatility": 0.05,
    },
    "Consumer Defensive": {
        "growth": 0.15,
        "profitability": 0.25,
        "valuation": 0.25,
        "quality": 0.20,
        "momentum": 0.10,
        "volatility": 0.05,
    },
    "Healthcare": {
        "growth": 0.28,
        "profitability": 0.22,
        "valuation": 0.18,
        "quality": 0.17,
        "momentum": 0.10,
        "volatility": 0.05,
    },
    "Financial Services": {
        "growth": 0.18,
        "profitability": 0.25,
        "valuation": 0.22,
        "quality": 0.20,
        "momentum": 0.10,
        "volatility": 0.05,
    },
    "Industrials": {
        "growth": 0.22,
        "profitability": 0.23,
        "valuation": 0.20,
        "quality": 0.18,
        "momentum": 0.12,
        "volatility": 0.05,
    },
    "Basic Materials": {
        "growth": 0.18,
        "profitability": 0.22,
        "valuation": 0.22,
        "quality": 0.18,
        "momentum": 0.15,
        "volatility": 0.05,
    },
    "Energy": {
        "growth": 0.15,
        "profitability": 0.25,
        "valuation": 0.25,
        "quality": 0.15,
        "momentum": 0.15,
        "volatility": 0.05,
    },
    "Utilities": {
        "growth": 0.10,
        "profitability": 0.25,
        "valuation": 0.30,
        "quality": 0.20,
        "momentum": 0.08,
        "volatility": 0.07,
    },
    "Real Estate": {
        "growth": 0.15,
        "profitability": 0.23,
        "valuation": 0.27,
        "quality": 0.18,
        "momentum": 0.10,
        "volatility": 0.07,
    },
}

# 기본 가중치 (섹터 미확인 시)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "growth": 0.22,
    "profitability": 0.22,
    "valuation": 0.20,
    "quality": 0.18,
    "momentum": 0.12,
    "volatility": 0.06,
}

# 시장 상황별 가중치 조정 배수
MARKET_CONDITION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "bull": {  # 강세장: 성장/모멘텀 중시
        "growth": 1.2,
        "profitability": 1.0,
        "valuation": 0.8,
        "quality": 0.9,
        "momentum": 1.3,
        "volatility": 0.7,
    },
    "bear": {  # 약세장: 밸류에이션/안정성 중시
        "growth": 0.8,
        "profitability": 1.2,
        "valuation": 1.3,
        "quality": 1.2,
        "momentum": 0.7,
        "volatility": 1.3,
    },
    "neutral": {  # 횡보장: 균형
        "growth": 1.0,
        "profitability": 1.0,
        "valuation": 1.0,
        "quality": 1.0,
        "momentum": 1.0,
        "volatility": 1.0,
    },
}


# ===== Z-score 정규화 + 윈저화 =====
def zscore_normalize(
    values: List[Optional[float]],
    winsorize_percentile: float = 0.05,
    higher_is_better: bool = True,
) -> List[float]:
    """Z-score 정규화 + 윈저화

    Args:
        values: 원본 값 리스트
        winsorize_percentile: 윈저화 백분위 (양쪽 끝 제거)
        higher_is_better: True면 높은 값이 좋음

    Returns:
        0~1 범위로 정규화된 점수 리스트
    """
    # None 값 처리
    valid_values = [v for v in values if v is not None]
    if len(valid_values) < 2:
        return [0.5] * len(values)

    arr = np.array(valid_values, dtype=float)

    # 윈저화 (극단치 제거)
    lower = np.percentile(arr, winsorize_percentile * 100)
    upper = np.percentile(arr, (1 - winsorize_percentile) * 100)
    arr_winsorized = np.clip(arr, lower, upper)

    # Z-score 계산
    mean = np.mean(arr_winsorized)
    std = np.std(arr_winsorized)
    if std < 1e-10:
        return [0.5] * len(values)

    # 원본 값들에 대해 Z-score 계산
    result = []
    for v in values:
        if v is None:
            result.append(0.5)
        else:
            # 윈저화 적용
            v_clipped = max(lower, min(v, upper))
            z = (v_clipped - mean) / std
            # Z-score를 0~1로 변환 (CDF 근사)
            # 표준정규분포에서 -3 ~ +3 범위를 0~1로 매핑
            score = (z + 3) / 6
            score = max(0.0, min(1.0, score))
            if not higher_is_better:
                score = 1.0 - score
            result.append(round(score, 4))

    return result


def zscore_normalize_by_group(
    values: List[Optional[float]],
    groups: List[Optional[str]],
    winsorize_percentile: float = 0.05,
    higher_is_better: bool = True,
) -> List[float]:
    """그룹(섹터)별 Z-score 정규화"""
    from collections import defaultdict

    group_to_indices = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g or "Unknown"].append(i)

    result = [0.5] * len(values)

    for group, indices in group_to_indices.items():
        sub_values = [values[i] for i in indices]
        sub_scores = zscore_normalize(sub_values, winsorize_percentile, higher_is_better)
        for j, idx in enumerate(indices):
            result[idx] = sub_scores[j]

    return result


# ===== 시장 상황 감지 =====
def detect_market_condition(benchmark: str = "SPY", lookback_days: int = 60) -> str:
    """시장 상황 감지 (강세/약세/횡보)

    기준:
    - 60일 수익률 > 10%: 강세장
    - 60일 수익률 < -10%: 약세장
    - 그 외: 횡보장
    """
    try:
        import yfinance as yf

        hist = normalize_yf_columns(
            yf.download(benchmark, period="6mo", interval="1d", progress=False, auto_adjust=True)
        )
        if hist.empty or len(hist) < lookback_days:
            return "neutral"

        close = hist["Close"].tail(lookback_days)

        first_val = float(close.iloc[0])
        last_val = float(close.iloc[-1])

        if first_val <= 0:
            return "neutral"

        returns = (last_val / first_val) - 1

        if returns > 0.10:
            return "bull"
        elif returns < -0.10:
            return "bear"
        else:
            return "neutral"
    except Exception as e:
        logger.warning(f"Failed to detect market condition: {e}")
        return "neutral"


def get_market_volatility(benchmark: str = "SPY", lookback_days: int = 30) -> float:
    """시장 변동성 계산 (연환산)"""
    try:
        import yfinance as yf

        hist = normalize_yf_columns(
            yf.download(benchmark, period="3mo", interval="1d", progress=False, auto_adjust=True)
        )
        if hist.empty or len(hist) < lookback_days:
            return 0.2  # 기본값 20%

        returns = hist["Close"].pct_change().dropna().tail(lookback_days)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        return float(annual_vol)
    except Exception:
        return 0.2


# ===== 팩터 계산 =====
@dataclass
class FactorScores:
    """팩터 점수 데이터 클래스"""
    ticker: str
    sector: Optional[str]
    growth: float
    profitability: float
    valuation: float
    quality: float
    momentum: float
    volatility: float
    raw_metrics: Dict


def calculate_factors(
    ticker: str,
    fundamentals: Dict,
    momentum_data: Dict,
    event_score: float = 0.5,
) -> FactorScores:
    """개별 종목의 6개 팩터 원시 값 계산

    Returns:
        FactorScores: 정규화 전 원시 팩터 값
    """
    # Growth 팩터 원시값
    eps = fundamentals.get("eps")
    rev_growth = fundamentals.get("revenueGrowth")
    earnings_growth = fundamentals.get("earningsQuarterlyGrowth")

    growth_raw = []
    if eps is not None:
        growth_raw.append(eps)
    if rev_growth is not None:
        growth_raw.append(rev_growth)
    if earnings_growth is not None:
        growth_raw.append(earnings_growth)
    growth_value = np.mean(growth_raw) if growth_raw else None

    # Profitability 팩터 원시값
    profit_margin = fundamentals.get("profitMargins")
    roe = fundamentals.get("returnOnEquity")
    roa = fundamentals.get("returnOnAssets")
    roic = fundamentals.get("roic")

    prof_raw = []
    if profit_margin is not None:
        prof_raw.append(profit_margin)
    if roe is not None:
        prof_raw.append(roe)
    if roa is not None:
        prof_raw.append(roa)
    if roic is not None:
        prof_raw.append(roic)
    profitability_value = np.mean(prof_raw) if prof_raw else None

    # Valuation 팩터 원시값 (낮을수록 좋음)
    pe = fundamentals.get("pe")
    pb = fundamentals.get("pb")

    val_raw = []
    if pe is not None and pe > 0:
        val_raw.append(pe)
    if pb is not None and pb > 0:
        val_raw.append(pb)
    valuation_value = np.mean(val_raw) if val_raw else None

    # Quality 팩터 (profitability + event)
    quality_value = None
    if profitability_value is not None:
        quality_value = profitability_value * 0.7 + event_score * 0.3

    # Momentum 팩터
    mom1 = momentum_data.get("mom1") or 0
    mom3 = momentum_data.get("mom3") or 0
    mom6 = momentum_data.get("mom6") or 0
    mom12 = momentum_data.get("mom12") or 0
    # 가중 평균 (최근 모멘텀에 더 높은 가중치)
    momentum_value = mom1 * 0.4 + mom3 * 0.3 + mom6 * 0.2 + mom12 * 0.1

    # Volatility 팩터 (낮을수록 좋음)
    volatility_value = _calculate_volatility(ticker)

    return FactorScores(
        ticker=ticker,
        sector=fundamentals.get("sector"),
        growth=growth_value,
        profitability=profitability_value,
        valuation=valuation_value,
        quality=quality_value,
        momentum=momentum_value,
        volatility=volatility_value,
        raw_metrics={
            "pe": pe,
            "pb": pb,
            "eps": eps,
            "revenueGrowth": rev_growth,
            "earningsGrowth": earnings_growth,
            "profitMargins": profit_margin,
            "roe": roe,
            "roa": roa,
            "roic": roic,
            "mom1": mom1,
            "mom3": mom3,
            "mom6": mom6,
            "mom12": mom12,
            "volatility": volatility_value,
            "eventScore": event_score,
        }
    )


def _calculate_volatility(ticker: str, lookback_days: int = 60) -> Optional[float]:
    """개별 종목 변동성 계산"""
    try:
        import yfinance as yf

        hist = normalize_yf_columns(
            yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        )
        if hist.empty or len(hist) < lookback_days:
            return None

        returns = hist["Close"].pct_change().dropna().tail(lookback_days)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        return float(annual_vol)
    except Exception:
        return None


# ===== 고급 랭킹 엔진 =====
class AdvancedRankingEngine:
    """고급 랭킹 엔진

    사용 예시:
        engine = AdvancedRankingEngine()
        engine.detect_market_condition()  # 시장 상황 감지

        results = await engine.rank_async(
            tickers=["AAPL", "MSFT", "GOOGL"],
            use_sector_weights=True,
            use_market_adjustment=True
        )
    """

    def __init__(self):
        self.market_condition: str = "neutral"
        self.market_volatility: float = 0.2

    def detect_market(self) -> Dict:
        """시장 상황 감지 및 저장"""
        self.market_condition = detect_market_condition()
        self.market_volatility = get_market_volatility()
        return {
            "condition": self.market_condition,
            "volatility": round(self.market_volatility, 4)
        }

    def get_weights(
        self,
        sector: Optional[str],
        use_sector_weights: bool = True,
        use_market_adjustment: bool = True,
    ) -> Dict[str, float]:
        """섹터 및 시장 상황을 반영한 가중치 계산"""
        # 기본 가중치
        if use_sector_weights and sector in SECTOR_WEIGHTS:
            weights = dict(SECTOR_WEIGHTS[sector])
        else:
            weights = dict(DEFAULT_WEIGHTS)

        # 시장 상황 반영
        if use_market_adjustment:
            multipliers = MARKET_CONDITION_MULTIPLIERS.get(self.market_condition, {})
            for factor in weights:
                if factor in multipliers:
                    weights[factor] *= multipliers[factor]

            # 정규화 (합이 1이 되도록)
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return weights

    def rank_sync(
        self,
        tickers: List[str],
        use_sector_weights: bool = True,
        use_market_adjustment: bool = True,
        sector_neutral: bool = False,
        dip_weight: float = 0.12,
        use_dip_bonus: bool = True,
    ) -> List[Dict]:
        """동기 버전 랭킹"""
        from .market_data import get_fundamentals_snapshot, get_momentum_metrics
        from .filings import keyword_event_score
        from .analytics import compute_dip_bonus_by_prices

        # 데이터 수집
        fundamentals = [get_fundamentals_snapshot(t) for t in tickers]
        momentum = [get_momentum_metrics(t) for t in tickers]

        try:
            event_scores = [keyword_event_score(t) for t in tickers]
        except Exception:
            event_scores = [0.5] * len(tickers)

        # 팩터 계산
        factors = [
            calculate_factors(t, f, m, e)
            for t, f, m, e in zip(tickers, fundamentals, momentum, event_scores)
        ]

        # 팩터 값 추출
        sectors = [f.sector for f in factors]
        growth_raw = [f.growth for f in factors]
        prof_raw = [f.profitability for f in factors]
        val_raw = [f.valuation for f in factors]
        qual_raw = [f.quality for f in factors]
        mom_raw = [f.momentum for f in factors]
        vol_raw = [f.volatility for f in factors]

        # Z-score 정규화
        if sector_neutral:
            growth_scores = zscore_normalize_by_group(growth_raw, sectors, higher_is_better=True)
            prof_scores = zscore_normalize_by_group(prof_raw, sectors, higher_is_better=True)
            val_scores = zscore_normalize_by_group(val_raw, sectors, higher_is_better=False)
            qual_scores = zscore_normalize_by_group(qual_raw, sectors, higher_is_better=True)
            mom_scores = zscore_normalize_by_group(mom_raw, sectors, higher_is_better=True)
            vol_scores = zscore_normalize_by_group(vol_raw, sectors, higher_is_better=False)
        else:
            growth_scores = zscore_normalize(growth_raw, higher_is_better=True)
            prof_scores = zscore_normalize(prof_raw, higher_is_better=True)
            val_scores = zscore_normalize(val_raw, higher_is_better=False)
            qual_scores = zscore_normalize(qual_raw, higher_is_better=True)
            mom_scores = zscore_normalize(mom_raw, higher_is_better=True)
            vol_scores = zscore_normalize(vol_raw, higher_is_better=False)

        # 최종 점수 계산
        results = []
        for i, ticker in enumerate(tickers):
            weights = self.get_weights(
                sectors[i],
                use_sector_weights=use_sector_weights,
                use_market_adjustment=use_market_adjustment
            )

            base_score = (
                growth_scores[i] * weights.get("growth", 0.22) +
                prof_scores[i] * weights.get("profitability", 0.22) +
                val_scores[i] * weights.get("valuation", 0.20) +
                qual_scores[i] * weights.get("quality", 0.18) +
                mom_scores[i] * weights.get("momentum", 0.12) +
                vol_scores[i] * weights.get("volatility", 0.06)
            )

            # Dip 보너스
            dip_bonus = 0.0
            if use_dip_bonus:
                dip_bonus = dip_weight * compute_dip_bonus_by_prices(ticker)

            final_score = base_score + dip_bonus

            results.append({
                "ticker": ticker,
                "sector": sectors[i],
                "score": round(final_score, 4),
                "base_score": round(base_score, 4),
                "dip_bonus": round(dip_bonus, 4),
                # 팩터 점수
                "growth": round(growth_scores[i], 4),
                "profitability": round(prof_scores[i], 4),
                "valuation": round(val_scores[i], 4),
                "quality": round(qual_scores[i], 4),
                "momentum": round(mom_scores[i], 4),
                "volatility": round(vol_scores[i], 4),
                # 적용된 가중치
                "weights_applied": weights,
                # 원시 메트릭
                **factors[i].raw_metrics,
            })

        # 점수 내림차순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    async def rank_async(
        self,
        tickers: List[str],
        use_sector_weights: bool = True,
        use_market_adjustment: bool = True,
        sector_neutral: bool = False,
        dip_weight: float = 0.12,
        use_dip_bonus: bool = True,
        max_concurrent: int = 5,
    ) -> List[Dict]:
        """비동기 버전 랭킹 (병렬 데이터 수집)"""
        import asyncio
        from .async_utils import parallel_map
        from .market_data import get_fundamentals_snapshot, get_momentum_metrics
        from .filings import keyword_event_score
        from .analytics import compute_dip_bonus_by_prices

        # 병렬 데이터 수집
        fundamentals, momentum = await asyncio.gather(
            parallel_map(get_fundamentals_snapshot, tickers, max_concurrent),
            parallel_map(get_momentum_metrics, tickers, max_concurrent),
        )

        try:
            event_scores = await parallel_map(keyword_event_score, tickers, max_concurrent=3)
        except Exception:
            event_scores = [0.5] * len(tickers)

        # 팩터 계산
        factors = [
            calculate_factors(t, f, m, e)
            for t, f, m, e in zip(tickers, fundamentals, momentum, event_scores)
        ]

        # 팩터 값 추출
        sectors = [f.sector for f in factors]
        growth_raw = [f.growth for f in factors]
        prof_raw = [f.profitability for f in factors]
        val_raw = [f.valuation for f in factors]
        qual_raw = [f.quality for f in factors]
        mom_raw = [f.momentum for f in factors]
        vol_raw = [f.volatility for f in factors]

        # Z-score 정규화
        if sector_neutral:
            growth_scores = zscore_normalize_by_group(growth_raw, sectors, higher_is_better=True)
            prof_scores = zscore_normalize_by_group(prof_raw, sectors, higher_is_better=True)
            val_scores = zscore_normalize_by_group(val_raw, sectors, higher_is_better=False)
            qual_scores = zscore_normalize_by_group(qual_raw, sectors, higher_is_better=True)
            mom_scores = zscore_normalize_by_group(mom_raw, sectors, higher_is_better=True)
            vol_scores = zscore_normalize_by_group(vol_raw, sectors, higher_is_better=False)
        else:
            growth_scores = zscore_normalize(growth_raw, higher_is_better=True)
            prof_scores = zscore_normalize(prof_raw, higher_is_better=True)
            val_scores = zscore_normalize(val_raw, higher_is_better=False)
            qual_scores = zscore_normalize(qual_raw, higher_is_better=True)
            mom_scores = zscore_normalize(mom_raw, higher_is_better=True)
            vol_scores = zscore_normalize(vol_raw, higher_is_better=False)

        # Dip 보너스 병렬 계산
        if use_dip_bonus:
            dip_bonuses = await parallel_map(compute_dip_bonus_by_prices, tickers, max_concurrent)
        else:
            dip_bonuses = [0.0] * len(tickers)

        # 최종 점수 계산
        results = []
        for i, ticker in enumerate(tickers):
            weights = self.get_weights(
                sectors[i],
                use_sector_weights=use_sector_weights,
                use_market_adjustment=use_market_adjustment
            )

            base_score = (
                growth_scores[i] * weights.get("growth", 0.22) +
                prof_scores[i] * weights.get("profitability", 0.22) +
                val_scores[i] * weights.get("valuation", 0.20) +
                qual_scores[i] * weights.get("quality", 0.18) +
                mom_scores[i] * weights.get("momentum", 0.12) +
                vol_scores[i] * weights.get("volatility", 0.06)
            )

            dip_bonus = dip_weight * dip_bonuses[i] if use_dip_bonus else 0.0
            final_score = base_score + dip_bonus

            results.append({
                "ticker": ticker,
                "sector": sectors[i],
                "score": round(final_score, 4),
                "base_score": round(base_score, 4),
                "dip_bonus": round(dip_bonus, 4),
                # 팩터 점수
                "growth": round(growth_scores[i], 4),
                "profitability": round(prof_scores[i], 4),
                "valuation": round(val_scores[i], 4),
                "quality": round(qual_scores[i], 4),
                "momentum": round(mom_scores[i], 4),
                "volatility": round(vol_scores[i], 4),
                # 적용된 가중치
                "weights_applied": weights,
                # 원시 메트릭
                **factors[i].raw_metrics,
            })

        # 점수 내림차순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


# ===== 글로벌 인스턴스 =====
_engine: Optional[AdvancedRankingEngine] = None


def get_ranking_engine() -> AdvancedRankingEngine:
    """랭킹 엔진 싱글톤 인스턴스"""
    global _engine
    if _engine is None:
        _engine = AdvancedRankingEngine()
    return _engine


# ===== 편의 함수 =====


async def rank_advanced_async(
    tickers: List[str],
    use_sector_weights: bool = True,
    use_market_adjustment: bool = True,
    sector_neutral: bool = False,
    dip_weight: float = 0.12,
    use_dip_bonus: bool = True,
    max_concurrent: int = 5,
) -> List[Dict]:
    """고급 랭킹 (비동기 버전)"""
    engine = get_ranking_engine()
    return await engine.rank_async(
        tickers,
        use_sector_weights=use_sector_weights,
        use_market_adjustment=use_market_adjustment,
        sector_neutral=sector_neutral,
        dip_weight=dip_weight,
        use_dip_bonus=use_dip_bonus,
        max_concurrent=max_concurrent,
    )
