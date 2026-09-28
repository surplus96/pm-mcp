"""Data-quality checks and server operations (cache, circuit breakers, scheduler)."""
from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_server.endpoints._common import DESTRUCTIVE_LOCAL, READ_EXTERNAL, READ_LOCAL, Period, ToolSpec, check, clean_tickers


def _download(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    from mcp_server.tools.market_data import _yf_symbol
    from mcp_server.tools.yf_utils import normalize_yf_columns

    data = normalize_yf_columns(yf.download(_yf_symbol(ticker), period=period, progress=False))
    if data.empty:
        raise ToolError(f"{ticker}: 가격 데이터가 없습니다.")
    return data


def _date(idx: Any) -> str:
    return str(idx.date()) if hasattr(idx, "date") else str(idx)


def _outliers(ticker: str, period: str, threshold: float) -> dict[str, Any]:
    returns = _download(ticker, period)["Close"].pct_change().dropna()
    mean, std = returns.mean(), returns.std()
    if std == 0:
        return {"ticker": ticker, "outlier_count": 0, "message": "변동성이 없습니다."}
    z = (returns - mean) / std
    hits = returns[abs(z) > threshold]
    return {
        "ticker": ticker,
        "period": period,
        "threshold": threshold,
        "outlier_count": len(hits),
        "outliers": [
            {"date": _date(d), "return_pct": round(float(r) * 100, 2), "z_score": round(float(z[d]), 2)}
            for d, r in list(hits.items())[:20]
        ],
        "statistics": {
            "mean_return": round(float(mean) * 100, 4),
            "std_return": round(float(std) * 100, 4),
            "max_return": round(float(returns.max()) * 100, 2),
            "min_return": round(float(returns.min()) * 100, 2),
        },
    }


def _missing(ticker: str, period: str) -> dict[str, Any]:
    data = _download(ticker, period)
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in data.columns]
    by_col = {k: int(v) for k, v in data[cols].isna().sum().items()}
    total = sum(by_col.values())
    cells = len(data) * len(cols)
    gaps = data[cols][data[cols].isna().any(axis=1)]
    return {
        "ticker": ticker,
        "period": period,
        "total_rows": len(data),
        "missing_count": total,
        "missing_pct": round(total / cells * 100, 2) if cells else 0,
        "by_column": by_col,
        "missing_dates": [
            {"date": _date(idx), "missing_cols": [c for c in cols if pd.isna(row[c])]}
            for idx, row in gaps.head(20).iterrows()
        ],
    }


def data_quality(
    tickers: Annotated[list[str], Field(min_length=1)],
    check_type: Literal["validate", "clean", "outliers", "missing"] = "validate",
    period: Period = "1y",
    threshold: Annotated[float, Field(gt=0, description="outliers용 z-score 임계값")] = 3.0,
) -> dict[str, Any]:
    """가격 데이터 품질 점검. 분석 결과가 이상할 때 먼저 확인합니다.

    - validate: 품질 점수(0-100)·등급·검사 항목·권장사항 (여러 종목이면 요약)
    - clean: 검증 후 자동 정제(보간·윈저화)와 정제 전후 비교
    - outliers: 일간 수익률 z-score 이상치 · missing: 누락 값/날짜
    """
    from mcp_server.tools.data_validator import get_data_quality_summary, validate_and_clean

    symbols = clean_tickers(tickers)
    if check_type == "validate" and len(symbols) > 1:
        return get_data_quality_summary(symbols, period)

    def one(t: str) -> dict[str, Any]:
        if check_type == "outliers":
            return _outliers(t, period, threshold)
        if check_type == "missing":
            return _missing(t, period)
        return check(validate_and_clean(t, period, auto_clean=check_type == "clean"))

    if len(symbols) == 1:
        return one(symbols[0])
    return {t: one(t) for t in symbols}


def ops_status(history_limit: Annotated[int, Field(ge=0, le=100)] = 10) -> dict[str, Any]:
    """서버 운영 상태: 캐시 통계, 외부 API 서킷 브레이커 상태, 스케줄러 작업과 최근 실행 이력."""
    from mcp_server.tools.cache_manager import cache_manager
    from mcp_server.tools.resilience import get_all_circuit_status
    from mcp_server.tools.scheduler import get_scheduler

    scheduler = get_scheduler()
    return {
        "cache": cache_manager.stats(),
        "circuits": get_all_circuit_status(),
        "scheduler": scheduler.get_status(),
        "job_history": scheduler.get_job_history(None, history_limit) if history_limit else [],
    }


def ops_action(
    action: Literal["cache_clear", "cache_expire", "circuit_reset", "scheduler_start", "scheduler_stop", "scheduler_run_job"],
    target: Annotated[
        str | None,
        Field(description="circuit_reset: 서킷 이름(생략 시 전체) / scheduler_run_job: market_refresh, news_scan, "
                          "filings_check, weekly_report, cache_cleanup, metrics_precompute"),
    ] = None,
) -> dict[str, Any]:
    """운영 작업 실행. cache_clear는 모든 캐시를 지우므로 이후 요청이 느려집니다."""
    if action in ("cache_clear", "cache_expire"):
        from mcp_server.tools.cache_manager import cache_manager

        count = cache_manager.clear() if action == "cache_clear" else cache_manager.expire()
        return {"action": action, "items": count}

    if action == "circuit_reset":
        from mcp_server.tools.resilience import CircuitBreaker, reset_all_circuits

        if target:
            cb = CircuitBreaker._instances.get(target)
            if not cb:
                raise ToolError(f"서킷 '{target}'을 찾을 수 없습니다. 이름: {sorted(CircuitBreaker._instances)}")
            cb.reset()
        else:
            reset_all_circuits()
        return {"action": action, "reset": target or "all"}

    from mcp_server.tools.scheduler import get_scheduler

    scheduler = get_scheduler()
    if action == "scheduler_start":
        scheduler.start()
    elif action == "scheduler_stop":
        scheduler.stop()
    else:
        if not target:
            raise ToolError("scheduler_run_job에는 target(job_id)이 필요합니다.")
        return scheduler.run_job_now(target)
    return {"action": action, "status": scheduler.get_status()}


TOOLS: list[ToolSpec] = [
    (data_quality, READ_EXTERNAL),
    (ops_status, READ_LOCAL),
    (ops_action, DESTRUCTIVE_LOCAL),
]
