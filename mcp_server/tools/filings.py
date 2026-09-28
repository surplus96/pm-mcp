from __future__ import annotations
from typing import List, Dict, Optional
import os
import logging
import requests
from .llm import summarize_items
from .cache_manager import cache_manager, TTL
from .resilience import (
    retry_with_backoff, Timeout, RetryConfig,
    circuit_sec, CircuitOpenError
)

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_session = requests.Session()
_ticker_cache: Dict[str, str] = {}  # 인메모리 캐시 (CIK 매핑용)


def _ua() -> str:
    ua = os.getenv("SEC_EDGAR_USER_AGENT") or "contact@example.com PM-MCP"
    return ua


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": _ua(),
        "Accept-Encoding": "gzip, deflate",
    }


@retry_with_backoff(
    attempts=RetryConfig.SEC_EDGAR["attempts"],
    min_wait=RetryConfig.SEC_EDGAR["min_wait"],
    max_wait=RetryConfig.SEC_EDGAR["max_wait"]
)
def _fetch_sec_tickers() -> dict:
    """SEC 티커 목록 조회 (재시도 + 서킷 브레이커)"""
    def _do_request():
        resp = _session.get(SEC_TICKERS_URL, headers=_headers(), timeout=Timeout.SEC_EDGAR)
        resp.raise_for_status()
        return resp.json()
    return circuit_sec.call(_do_request)


def get_cik_from_ticker(ticker: str) -> Optional[str]:
    t = (ticker or "").upper().strip()
    if not t:
        return None
    if t in _ticker_cache:
        return _ticker_cache[t]
    try:
        data = _fetch_sec_tickers()
        for _, row in data.items():
            if (row.get("ticker") or "").upper() == t:
                cik_str = str(row.get("cik_str"))
                _ticker_cache[t] = cik_str
                return cik_str
    except CircuitOpenError:
        logger.warning(f"SEC circuit open, cannot fetch CIK for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to fetch CIK for {ticker}: {e}")
    return None


def _zero_pad_10(cik_str: str) -> str:
    return str(cik_str).zfill(10)


@retry_with_backoff(
    attempts=RetryConfig.SEC_EDGAR["attempts"],
    min_wait=RetryConfig.SEC_EDGAR["min_wait"],
    max_wait=RetryConfig.SEC_EDGAR["max_wait"]
)
def _fetch_sec_submissions(cik: str) -> dict:
    """SEC 제출물 조회 (재시도 + 서킷 브레이커)"""
    def _do_request():
        resp = _session.get(
            SEC_SUBMISSIONS_URL.format(cik=_zero_pad_10(cik)),
            headers=_headers(),
            timeout=Timeout.SEC_EDGAR
        )
        resp.raise_for_status()
        return resp.json()
    return circuit_sec.call(_do_request)


def fetch_recent_filings(ticker: str, forms: Optional[List[str]] = None, limit: int = 10, use_cache: bool = True) -> List[Dict]:
    """SEC 공시 조회 (6시간 캐시, 재시도 + 서킷 브레이커 적용)"""
    forms = forms or ["8-K", "10-Q", "10-K"]

    # 캐시 키 생성
    forms_key = ",".join(sorted(forms))
    cache_key = f"filings:{ticker}:{forms_key}:{limit}"

    if use_cache:
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    cik_str = get_cik_from_ticker(ticker)
    if not cik_str:
        return []

    try:
        j = _fetch_sec_submissions(cik_str)
        recent = j.get("filings", {}).get("recent", {})
        out: List[Dict] = []
        for i, form in enumerate(recent.get("form", [])):
            if form not in forms:
                continue
            acc = (recent.get("accessionNumber", [""])[i] or "").replace("-", "")
            doc = recent.get("primaryDocument", [""])[i]
            filing_date = recent.get("filingDate", [""])[i]
            report_date = recent.get("reportDate", [""])[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_str)}/{acc}/{doc}"
            out.append({
                "ticker": ticker,
                "cik": cik_str,
                "form": form,
                "filingDate": filing_date,
                "reportDate": report_date,
                "accessionNumber": recent.get("accessionNumber", [""])[i],
                "primaryDocument": doc,
                "url": url,
                "title": recent.get("primaryDocDescription", [""])[i],
            })
            if len(out) >= limit:
                break

        # 결과 캐싱
        if use_cache and out:
            cache_manager.set(cache_key, out, TTL.FILING)

        logger.debug(f"Fetched {len(out)} filings for {ticker}")
        return out
    except CircuitOpenError:
        logger.warning(f"SEC circuit open for {ticker}, using cached data")
        stale_data = cache_manager.get(cache_key)
        if stale_data:
            return stale_data
        return []
    except Exception as e:
        # 캐시된 데이터 폴백
        logger.warning(f"Failed to fetch filings for {ticker}: {e}, using cached data")
        stale_data = cache_manager.get(cache_key)
        if stale_data:
            return stale_data
        return []


from mcp_server.config import EVENT_WEIGHTS_PATH
import json

_EVENT_WEIGHTS_CACHE: Dict[str, float] | None = None

def _load_event_weights() -> Dict[str, float]:
    global _EVENT_WEIGHTS_CACHE
    if _EVENT_WEIGHTS_CACHE is not None:
        return _EVENT_WEIGHTS_CACHE
    try:
        with open(EVENT_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            _EVENT_WEIGHTS_CACHE = json.load(f)
    except Exception:
        _EVENT_WEIGHTS_CACHE = {"guidance":1.0,"partnership":0.8,"acquisition":1.0,"litigation":-0.8,"fda":1.0,"recall":-1.0}
    return _EVENT_WEIGHTS_CACHE


def keyword_event_score(ticker: str, limit: int = 10) -> float:
    weights = _load_event_weights()
    filings = fetch_recent_filings(ticker, limit=limit)
    if not filings:
        return 0.5
    score = 0.0
    cnt = 0
    for f in filings:
        title = (f.get("title") or "").lower()
        for k, w in weights.items():
            if k in title:
                score += float(w)
        cnt += 1
    if cnt == 0:
        return 0.5
    raw = score / max(1, cnt)
    norm = max(0.0, min((raw + 1.0) / 2.0, 1.0))
    return round(norm, 3)

# --- Added back: summarize_filings_items for theme_report compatibility ---

def summarize_filings_items(filings: List[Dict], max_items: int = 6) -> str:
    """최근 공시 목록을 요약 문자열로 반환. Perplexity 요약 실패 시 불릿 목록 폴백.
    입력: [{form, filingDate, title, url}, ...]
    """
    if not filings:
        return ""
    lines = []
    for f in filings[:max_items]:
        lines.append(f"{f.get('form')} | {f.get('filingDate')} | {f.get('title') or ''} | {f.get('url')}")
    # 시도: LLM 요약
    try:
        summary = summarize_items(lines, max_sentences=6)
        if summary:
            return summary
    except Exception:
        pass
    # 폴백: 불릿 나열
    return "\n".join(f"- {ln}" for ln in lines)
