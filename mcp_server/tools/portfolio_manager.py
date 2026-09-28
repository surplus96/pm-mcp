"""
포트폴리오 관리 모듈
- 리밸런싱 알림
- 손익 추적
- 배당 캘린더
- 목표가 알림
- 상관관계 분석
- 섹터 익스포저
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import yfinance as yf
import pandas as pd

from mcp_server.config import safe_filename
from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.yf_utils import normalize_yf_columns

logger = logging.getLogger(__name__)

# 포트폴리오 데이터 저장 경로
PORTFOLIO_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "portfolio"
)
os.makedirs(PORTFOLIO_DATA_DIR, exist_ok=True)


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class Holding:
    """보유 종목"""
    ticker: str
    shares: float
    entry_price: float
    entry_date: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_weight: Optional[float] = None  # 목표 비중 (0~1)
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Holding":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Portfolio:
    """포트폴리오"""
    name: str
    holdings: List[Holding]
    cash: float = 0.0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "holdings": [h.to_dict() for h in self.holdings],
            "cash": self.cash,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Portfolio":
        holdings = [Holding.from_dict(h) for h in data.get("holdings", [])]
        return cls(
            name=data.get("name", "default"),
            holdings=holdings,
            cash=data.get("cash", 0.0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


# ============================================================
# 포트폴리오 저장/로드
# ============================================================

def save_portfolio(portfolio: Portfolio, filename: str = "default") -> str:
    """포트폴리오 저장"""
    filepath = os.path.join(PORTFOLIO_DATA_DIR, f"{safe_filename(filename)}.json")
    portfolio.updated_at = datetime.now().isoformat()
    if not portfolio.created_at:
        portfolio.created_at = portfolio.updated_at

    with open(filepath, 'w') as f:
        json.dump(portfolio.to_dict(), f, indent=2)

    return filepath


def load_portfolio(filename: str = "default") -> Optional[Portfolio]:
    """포트폴리오 로드"""
    filepath = os.path.join(PORTFOLIO_DATA_DIR, f"{safe_filename(filename)}.json")
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)
    return Portfolio.from_dict(data)


def list_portfolios() -> List[str]:
    """저장된 포트폴리오 목록"""
    files = os.listdir(PORTFOLIO_DATA_DIR)
    return [f.replace(".json", "") for f in files if f.endswith(".json")]


# ============================================================
# 가격 조회 헬퍼
# ============================================================

def _get_current_price(ticker: str) -> Optional[float]:
    """현재가 조회"""
    cache_key = f"current_price_{ticker}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        cache_manager.set(cache_key, price, TTL.REALTIME)
        return price
    except Exception as e:
        logger.warning(f"Failed to get price for {ticker}: {e}")
        return None


def _get_ticker_info(ticker: str) -> Dict:
    """종목 정보 조회 (섹터, 배당 등)"""
    cache_key = f"ticker_info_{ticker}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info
        result = {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "name": info.get("shortName", ticker),
            "currency": info.get("currency", "USD"),
            "dividend_yield": info.get("dividendYield", 0),
            "dividend_rate": info.get("dividendRate", 0),
            "ex_dividend_date": info.get("exDividendDate"),
            "market_cap": info.get("marketCap", 0),
            "beta": info.get("beta", 1.0)
        }
        cache_manager.set(cache_key, result, TTL.FUNDAMENTAL)
        return result
    except Exception as e:
        logger.warning(f"Failed to get info for {ticker}: {e}")
        return {"sector": "Unknown", "name": ticker}


# ============================================================
# 손익 추적
# ============================================================

def calculate_pnl(holdings: List[Holding]) -> List[Dict]:
    """
    보유 종목별 손익 계산

    Returns:
        종목별 손익 정보 리스트
    """
    results = []

    # 병렬로 현재가 조회
    tickers = [h.ticker for h in holdings]
    current_prices = {}

    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        futures = {executor.submit(_get_current_price, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                current_prices[ticker] = future.result()
            except Exception:
                current_prices[ticker] = None

    for holding in holdings:
        current = current_prices.get(holding.ticker)
        if current is None:
            results.append({
                "ticker": holding.ticker,
                "error": "Price unavailable"
            })
            continue

        # 손익 계산
        cost_basis = holding.shares * holding.entry_price
        market_value = holding.shares * current
        pnl_amount = market_value - cost_basis
        pnl_percent = (current - holding.entry_price) / holding.entry_price * 100

        # 일일 손익 (간단 계산)
        daily_return = None
        try:
            hist = yf.Ticker(holding.ticker).history(period="2d")
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                daily_return = (current - prev_close) / prev_close * 100
        except Exception:
            pass

        results.append({
            "ticker": holding.ticker,
            "shares": holding.shares,
            "entry_price": holding.entry_price,
            "entry_date": holding.entry_date,
            "current_price": round(current, 2),
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2),
            "pnl_amount": round(pnl_amount, 2),
            "pnl_percent": round(pnl_percent, 2),
            "daily_return": round(daily_return, 2) if daily_return else None,
            "target_price": holding.target_price,
            "stop_loss": holding.stop_loss,
            "target_reached": current >= holding.target_price if holding.target_price else None,
            "stop_triggered": current <= holding.stop_loss if holding.stop_loss else None
        })

    return results


def get_portfolio_summary(holdings: List[Holding], cash: float = 0) -> Dict:
    """
    포트폴리오 전체 요약

    Returns:
        총 가치, 손익, 수익률 등
    """
    pnl_data = calculate_pnl(holdings)

    total_cost = 0
    total_value = 0
    total_pnl = 0
    winners = 0
    losers = 0

    valid_holdings = []
    for item in pnl_data:
        if "error" in item:
            continue
        valid_holdings.append(item)
        total_cost += item["cost_basis"]
        total_value += item["market_value"]
        total_pnl += item["pnl_amount"]
        if item["pnl_percent"] > 0:
            winners += 1
        elif item["pnl_percent"] < 0:
            losers += 1

    total_portfolio_value = total_value + cash
    total_return_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # 최고/최저 성과
    if valid_holdings:
        sorted_by_pnl = sorted(valid_holdings, key=lambda x: x["pnl_percent"], reverse=True)
        best = sorted_by_pnl[0]
        worst = sorted_by_pnl[-1]
    else:
        best = worst = None

    return {
        "total_cost": round(total_cost, 2),
        "total_market_value": round(total_value, 2),
        "cash": round(cash, 2),
        "total_portfolio_value": round(total_portfolio_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_percent": round(total_return_pct, 2),
        "holdings_count": len(valid_holdings),
        "winners": winners,
        "losers": losers,
        "win_rate": round(winners / len(valid_holdings) * 100, 1) if valid_holdings else 0,
        "best_performer": {"ticker": best["ticker"], "return": best["pnl_percent"]} if best else None,
        "worst_performer": {"ticker": worst["ticker"], "return": worst["pnl_percent"]} if worst else None,
        "holdings": pnl_data,
        "as_of": datetime.now().isoformat()
    }


# ============================================================
# 리밸런싱 알림
# ============================================================

def check_rebalancing(
    holdings: List[Holding],
    cash: float = 0,
    threshold: float = 0.05
) -> Dict:
    """
    리밸런싱 필요 여부 확인

    Args:
        holdings: 보유 종목 리스트
        cash: 현금
        threshold: 편차 임계값 (기본 5%)

    Returns:
        리밸런싱 필요 여부 및 조정 내역
    """
    # 목표 비중이 설정된 종목만 필터링
    target_holdings = [h for h in holdings if h.target_weight is not None]

    if not target_holdings:
        return {
            "needs_rebalancing": False,
            "message": "목표 비중이 설정된 종목이 없습니다.",
            "deviations": []
        }

    # 현재 가치 계산
    current_prices = {}
    for h in holdings:
        current_prices[h.ticker] = _get_current_price(h.ticker)

    total_value = cash
    for h in holdings:
        price = current_prices.get(h.ticker)
        if price:
            total_value += h.shares * price

    if total_value <= 0:
        return {"error": "포트폴리오 가치를 계산할 수 없습니다."}

    # 현재 비중 vs 목표 비중
    deviations = []
    for h in target_holdings:
        price = current_prices.get(h.ticker)
        if not price:
            continue

        current_value = h.shares * price
        current_weight = current_value / total_value
        target_weight = h.target_weight
        deviation = current_weight - target_weight

        if abs(deviation) > threshold:
            # 조정 필요 수량 계산
            target_value = total_value * target_weight
            value_diff = target_value - current_value
            shares_diff = value_diff / price

            deviations.append({
                "ticker": h.ticker,
                "current_weight": round(current_weight * 100, 2),
                "target_weight": round(target_weight * 100, 2),
                "deviation": round(deviation * 100, 2),
                "current_value": round(current_value, 2),
                "target_value": round(target_value, 2),
                "action": "매도" if deviation > 0 else "매수",
                "shares_to_trade": abs(round(shares_diff, 2)),
                "value_to_trade": abs(round(value_diff, 2))
            })

    needs_rebalancing = len(deviations) > 0
    total_deviation = sum(abs(d["deviation"]) for d in deviations)

    # 현금 비중 체크
    cash_weight = cash / total_value if total_value > 0 else 0

    return {
        "needs_rebalancing": needs_rebalancing,
        "total_deviation": round(total_deviation, 2),
        "threshold": threshold * 100,
        "deviations": sorted(deviations, key=lambda x: abs(x["deviation"]), reverse=True),
        "portfolio_value": round(total_value, 2),
        "cash_weight": round(cash_weight * 100, 2),
        "checked_at": datetime.now().isoformat()
    }


# ============================================================
# 배당 캘린더
# ============================================================

def get_dividend_calendar(holdings: List[Holding], days_ahead: int = 90) -> Dict:
    """
    보유 종목 배당 일정

    Args:
        holdings: 보유 종목 리스트
        days_ahead: 향후 조회 기간 (일)

    Returns:
        배당 일정 및 예상 배당금
    """
    tickers = [h.ticker for h in holdings]
    shares_map = {h.ticker: h.shares for h in holdings}

    upcoming = []
    total_annual_dividend = 0
    cutoff = datetime.now() + timedelta(days=days_ahead)

    for ticker in tickers:
        info = _get_ticker_info(ticker)
        shares = shares_map[ticker]

        dividend_rate = info.get("dividend_rate", 0) or 0
        dividend_yield = info.get("dividend_yield", 0) or 0
        ex_date_ts = info.get("ex_dividend_date")

        annual_dividend = shares * dividend_rate
        total_annual_dividend += annual_dividend

        # 배당락일 처리
        if ex_date_ts:
            try:
                ex_date = datetime.fromtimestamp(ex_date_ts)
                if datetime.now() <= ex_date <= cutoff:
                    upcoming.append({
                        "ticker": ticker,
                        "name": info.get("name", ticker),
                        "ex_dividend_date": ex_date.strftime("%Y-%m-%d"),
                        "days_until": (ex_date - datetime.now()).days,
                        "dividend_rate": dividend_rate,
                        "shares": shares,
                        "expected_payment": round(shares * dividend_rate / 4, 2)  # 분기 배당 가정
                    })
            except Exception:
                pass

        # 배당 정보가 없는 경우에도 기록
        if dividend_yield > 0:
            upcoming.append({
                "ticker": ticker,
                "name": info.get("name", ticker),
                "dividend_yield": round(dividend_yield * 100, 2),
                "annual_dividend": round(annual_dividend, 2),
                "shares": shares
            })

    # 날짜순 정렬
    dated = [u for u in upcoming if "ex_dividend_date" in u]
    undated = [u for u in upcoming if "ex_dividend_date" not in u]
    dated.sort(key=lambda x: x.get("days_until", 999))

    return {
        "upcoming_dividends": dated,
        "dividend_stocks": undated,
        "total_annual_dividend": round(total_annual_dividend, 2),
        "days_ahead": days_ahead,
        "as_of": datetime.now().isoformat()
    }


# ============================================================
# 목표가 알림
# ============================================================

def check_price_alerts(holdings: List[Holding]) -> Dict:
    """
    목표가/손절가 도달 여부 확인

    Returns:
        알림 목록
    """
    alerts = []

    for holding in holdings:
        current = _get_current_price(holding.ticker)
        if not current:
            continue

        alert_item = {
            "ticker": holding.ticker,
            "current_price": round(current, 2),
            "entry_price": holding.entry_price,
            "alerts": []
        }

        # 목표가 체크
        if holding.target_price:
            pct_to_target = (holding.target_price - current) / current * 100
            alert_item["target_price"] = holding.target_price
            alert_item["pct_to_target"] = round(pct_to_target, 2)

            if current >= holding.target_price:
                alert_item["alerts"].append({
                    "type": "TARGET_REACHED",
                    "message": f"목표가 ${holding.target_price} 도달! 현재가 ${current:.2f}",
                    "severity": "success"
                })
            elif pct_to_target < 5:
                alert_item["alerts"].append({
                    "type": "NEAR_TARGET",
                    "message": f"목표가까지 {pct_to_target:.1f}% 남음",
                    "severity": "info"
                })

        # 손절가 체크
        if holding.stop_loss:
            pct_to_stop = (current - holding.stop_loss) / current * 100
            alert_item["stop_loss"] = holding.stop_loss
            alert_item["pct_to_stop"] = round(pct_to_stop, 2)

            if current <= holding.stop_loss:
                alert_item["alerts"].append({
                    "type": "STOP_TRIGGERED",
                    "message": f"손절가 ${holding.stop_loss} 도달! 현재가 ${current:.2f}",
                    "severity": "danger"
                })
            elif pct_to_stop < 5:
                alert_item["alerts"].append({
                    "type": "NEAR_STOP",
                    "message": f"손절가까지 {pct_to_stop:.1f}% 남음",
                    "severity": "warning"
                })

        if alert_item["alerts"] or holding.target_price or holding.stop_loss:
            alerts.append(alert_item)

    # 심각도 순 정렬
    severity_order = {"danger": 0, "warning": 1, "success": 2, "info": 3}
    alerts.sort(key=lambda x: min(
        [severity_order.get(a["severity"], 4) for a in x["alerts"]]
        if x["alerts"] else [4]
    ))

    triggered = [a for a in alerts if any(
        al["type"] in ["TARGET_REACHED", "STOP_TRIGGERED"] for al in a["alerts"]
    )]

    return {
        "total_alerts": sum(len(a["alerts"]) for a in alerts),
        "triggered_count": len(triggered),
        "triggered": triggered,
        "all_alerts": alerts,
        "checked_at": datetime.now().isoformat()
    }


# ============================================================
# 상관관계 분석
# ============================================================

def analyze_correlation(
    tickers: List[str],
    period: str = "1y"
) -> Dict:
    """
    종목 간 상관관계 분석

    Args:
        tickers: 종목 리스트
        period: 분석 기간

    Returns:
        상관관계 매트릭스 및 분석
    """
    cache_key = f"correlation_{'-'.join(sorted(tickers))}_{period}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    # 가격 데이터 수집
    prices_dict = {}
    for ticker in tickers:
        try:
            data = normalize_yf_columns(
                yf.download(ticker, period=period, progress=False)
            )
            if not data.empty and "Close" in data.columns:
                prices_dict[ticker] = data["Close"]
        except Exception as e:
            logger.warning(f"Failed to get data for {ticker}: {e}")

    if len(prices_dict) < 2:
        return {"error": "상관관계 분석을 위해 최소 2개 종목이 필요합니다."}

    # DataFrame 생성
    df = pd.DataFrame(prices_dict)
    df = df.dropna()

    if len(df) < 30:
        return {"error": "데이터가 충분하지 않습니다."}

    # 수익률 계산
    returns = df.pct_change().dropna()

    # 상관관계 매트릭스
    corr_matrix = returns.corr()

    # 상관관계 쌍 추출
    pairs = []
    for i, t1 in enumerate(corr_matrix.columns):
        for j, t2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.loc[t1, t2]
                pairs.append({
                    "pair": f"{t1}-{t2}",
                    "correlation": round(float(corr), 3),
                    "relationship": _interpret_correlation(float(corr))
                })

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # 평균 상관관계
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    # 다각화 점수 (상관관계가 낮을수록 좋음)
    diversification_score = max(0, min(100, (1 - avg_corr) * 100))

    result = {
        "tickers": list(corr_matrix.columns),
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "pairs": pairs,
        "average_correlation": round(float(avg_corr), 3),
        "diversification_score": round(diversification_score, 1),
        "diversification_rating": _rate_diversification(diversification_score),
        "period": period,
        "data_points": len(returns),
        "analyzed_at": datetime.now().isoformat()
    }

    cache_manager.set(cache_key, result, TTL.DAILY)
    return result


def _interpret_correlation(corr: float) -> str:
    """상관관계 해석"""
    if corr >= 0.8:
        return "매우 강한 양의 상관"
    elif corr >= 0.5:
        return "강한 양의 상관"
    elif corr >= 0.3:
        return "약한 양의 상관"
    elif corr >= -0.3:
        return "상관관계 없음"
    elif corr >= -0.5:
        return "약한 음의 상관"
    elif corr >= -0.8:
        return "강한 음의 상관"
    else:
        return "매우 강한 음의 상관"


def _rate_diversification(score: float) -> str:
    """다각화 등급"""
    if score >= 80:
        return "우수"
    elif score >= 60:
        return "양호"
    elif score >= 40:
        return "보통"
    elif score >= 20:
        return "미흡"
    else:
        return "불량"


# ============================================================
# 섹터 익스포저
# ============================================================

def analyze_sector_exposure(holdings: List[Holding]) -> Dict:
    """
    섹터별 비중 분석

    Returns:
        섹터별 비중 및 집중도 분석
    """
    # 현재 가치 계산
    sector_values = defaultdict(float)
    sector_holdings = defaultdict(list)
    total_value = 0

    for holding in holdings:
        price = _get_current_price(holding.ticker)
        if not price:
            continue

        value = holding.shares * price
        total_value += value

        info = _get_ticker_info(holding.ticker)
        sector = info.get("sector", "Unknown")

        sector_values[sector] += value
        sector_holdings[sector].append({
            "ticker": holding.ticker,
            "name": info.get("name", holding.ticker),
            "value": round(value, 2),
            "shares": holding.shares
        })

    if total_value <= 0:
        return {"error": "포트폴리오 가치를 계산할 수 없습니다."}

    # 섹터별 비중 계산
    sectors = []
    for sector, value in sector_values.items():
        weight = value / total_value
        sectors.append({
            "sector": sector,
            "value": round(value, 2),
            "weight": round(weight * 100, 2),
            "holdings_count": len(sector_holdings[sector]),
            "holdings": sector_holdings[sector]
        })

    sectors.sort(key=lambda x: x["weight"], reverse=True)

    # 집중도 분석 (허핀달 지수)
    weights = [s["weight"] / 100 for s in sectors]
    hhi = sum(w ** 2 for w in weights)
    concentration = "높음" if hhi > 0.25 else "보통" if hhi > 0.15 else "낮음"

    # 상위 섹터
    top_sector = sectors[0] if sectors else None
    top_sector_warning = top_sector and top_sector["weight"] > 40

    return {
        "sectors": sectors,
        "sector_count": len(sectors),
        "total_value": round(total_value, 2),
        "concentration_index": round(hhi, 4),
        "concentration_level": concentration,
        "top_sector": top_sector["sector"] if top_sector else None,
        "top_sector_weight": top_sector["weight"] if top_sector else 0,
        "concentration_warning": top_sector_warning,
        "recommendation": _get_sector_recommendation(sectors, hhi),
        "analyzed_at": datetime.now().isoformat()
    }


def _get_sector_recommendation(sectors: List[Dict], hhi: float) -> str:
    """섹터 배분 추천"""
    if hhi > 0.25:
        return "섹터 집중도가 높습니다. 다른 섹터로 분산 투자를 고려하세요."
    elif len(sectors) < 3:
        return "섹터 다각화가 부족합니다. 최소 3개 이상 섹터로 분산을 권장합니다."
    else:
        return "섹터 배분이 양호합니다."


# ============================================================
# 종합 포트폴리오 분석
# ============================================================

def analyze_portfolio_comprehensive(
    holdings: List[Holding],
    cash: float = 0
) -> Dict:
    """
    포트폴리오 종합 분석

    Returns:
        손익, 리밸런싱, 배당, 알림, 상관관계, 섹터 분석 통합
    """
    tickers = [h.ticker for h in holdings]

    # 병렬로 각 분석 실행
    results = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(get_portfolio_summary, holdings, cash): "summary",
            executor.submit(check_rebalancing, holdings, cash): "rebalancing",
            executor.submit(get_dividend_calendar, holdings): "dividends",
            executor.submit(check_price_alerts, holdings): "alerts",
            executor.submit(analyze_correlation, tickers): "correlation",
            executor.submit(analyze_sector_exposure, holdings): "sectors"
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"error": str(e)}

    # 전체 건강도 점수 계산
    health_score = _calculate_health_score(results)

    return {
        "portfolio_name": "My Portfolio",
        "health_score": health_score,
        "summary": results.get("summary", {}),
        "rebalancing": results.get("rebalancing", {}),
        "dividends": results.get("dividends", {}),
        "alerts": results.get("alerts", {}),
        "correlation": results.get("correlation", {}),
        "sectors": results.get("sectors", {}),
        "analyzed_at": datetime.now().isoformat()
    }


def _calculate_health_score(results: Dict) -> Dict:
    """포트폴리오 건강도 점수"""
    scores = []
    details = []

    # 수익률 점수
    summary = results.get("summary", {})
    total_return = summary.get("total_return_percent", 0)
    if total_return > 20:
        scores.append(100)
        details.append("수익률 우수")
    elif total_return > 0:
        scores.append(70)
        details.append("수익률 양호")
    elif total_return > -10:
        scores.append(50)
        details.append("수익률 보통")
    else:
        scores.append(30)
        details.append("수익률 부진")

    # 다각화 점수
    correlation = results.get("correlation", {})
    div_score = correlation.get("diversification_score", 50)
    scores.append(div_score)
    if div_score >= 70:
        details.append("분산 투자 우수")
    elif div_score >= 50:
        details.append("분산 투자 양호")
    else:
        details.append("분산 투자 미흡")

    # 리밸런싱 점수
    rebalancing = results.get("rebalancing", {})
    if not rebalancing.get("needs_rebalancing", True):
        scores.append(100)
        details.append("비중 균형 유지")
    else:
        deviation = rebalancing.get("total_deviation", 0)
        if deviation < 10:
            scores.append(70)
            details.append("소폭 리밸런싱 필요")
        else:
            scores.append(40)
            details.append("리밸런싱 필요")

    # 알림 점수
    alerts = results.get("alerts", {})
    triggered = alerts.get("triggered_count", 0)
    if triggered == 0:
        scores.append(100)
    elif triggered == 1:
        scores.append(70)
        details.append("가격 알림 발생")
    else:
        scores.append(40)
        details.append(f"{triggered}개 가격 알림")

    avg_score = sum(scores) / len(scores) if scores else 50

    if avg_score >= 80:
        grade = "A"
        status = "매우 건강"
    elif avg_score >= 60:
        grade = "B"
        status = "건강"
    elif avg_score >= 40:
        grade = "C"
        status = "보통"
    else:
        grade = "D"
        status = "주의 필요"

    return {
        "score": round(avg_score, 1),
        "grade": grade,
        "status": status,
        "details": details
    }


# ============================================================
# 헬퍼 함수
# ============================================================

def create_holdings_from_text(text: str) -> List[Holding]:
    """
    텍스트에서 보유 종목 파싱

    Format: "TICKER:SHARES@PRICE, ..."
    Example: "AAPL:10@150, MSFT:5@400, GOOGL:3@140"
    """
    holdings = []
    parts = text.replace(" ", "").split(",")

    for part in parts:
        try:
            if ":" not in part:
                continue

            ticker_part, rest = part.split(":", 1)

            if "@" in rest:
                shares_str, price_str = rest.split("@", 1)
                shares = float(shares_str)
                entry_price = float(price_str)
            else:
                shares = float(rest)
                entry_price = _get_current_price(ticker_part) or 0

            holdings.append(Holding(
                ticker=ticker_part.upper(),
                shares=shares,
                entry_price=entry_price
            ))
        except Exception as e:
            logger.warning(f"Failed to parse holding: {part} - {e}")

    return holdings
