"""DART (금융감독원 전자공시) OPEN API adapter (FR-K10/K11/K15).

Wraps the ``OpenDartReader`` Python library to surface:
    * Korean corporate filings for a stock code (``get_filings``)
    * Key financial metrics from the latest annual K-IFRS report
      (``get_financials`` — ROE / ROA / operating margin / net margin
      / revenue growth) using XBRL extraction under the hood.

The module is safe to import even without ``OpenDartReader`` installed
— every public call returns an empty container so the wider pipeline
degrades gracefully instead of crashing. When the dependency *is*
available but the API key is missing, the same graceful-empty path is
taken and a single warning is logged.

Cache layer: short TTLs are fine here because DART updates are not
real-time — fundamentals change ~quarterly, filings at most daily.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from mcp_server.tools.cache_manager import TTL, cached

logger = logging.getLogger(__name__)


def _normalize_stock_code(ticker: str) -> str:
    """Strip ``.KS`` / ``.KQ`` / whitespace and return a 6-digit code."""
    t = (ticker or "").strip().upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        t = t[:-3]
    return t


class DartClient:
    """Thin wrapper around OpenDartReader with defensive fallbacks."""

    def __init__(self) -> None:
        self._reader = None
        self._key = os.getenv("DART_API_KEY", "").strip()
        if not self._key:
            logger.warning("DART_API_KEY not set; DartClient will return empty data.")
            return
        try:
            import OpenDartReader  # type: ignore
            self._reader = OpenDartReader(self._key)
            logger.info("DartClient initialised (OpenDartReader).")
        except ImportError:
            logger.warning("OpenDartReader not installed; `pip install OpenDartReader` to enable.")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to init OpenDartReader: %s", e)

    @property
    def ready(self) -> bool:
        return self._reader is not None

    # ----------------------------------------------------------------- filings

    @cached(ttl=TTL.DAILY, prefix="dart_filings")
    def get_filings(self, ticker: str, days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
        """Recent filings for a 6-digit KRX stock code.

        Returns a list of ``{date, title, type, url}`` dicts sorted
        newest-first. Empty list on any failure (missing key, network,
        unknown code).
        """
        if not self.ready:
            return []
        code = _normalize_stock_code(ticker)
        if not (code.isdigit() and len(code) == 6):
            return []
        try:
            from datetime import datetime, timedelta
            start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            end = datetime.now().strftime("%Y%m%d")
            df = self._reader.list(code, start=start, end=end)  # type: ignore[attr-defined]
            if df is None or getattr(df, "empty", True):
                return []
            records = df.head(limit).to_dict(orient="records")
            out: list[dict[str, Any]] = []
            for r in records:
                rcp = r.get("rcept_no") or r.get("rceptNo") or ""
                out.append({
                    "date": str(r.get("rcept_dt", "")),
                    "title": str(r.get("report_nm", "")),
                    "type": str(r.get("pblntf_ty", "")),
                    "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}" if rcp else "",
                })
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("DART filings failed for %s: %s", ticker, e)
            return []

    # -------------------------------------------------------------- financials

    @cached(ttl=TTL.FUNDAMENTAL, prefix="dart_fin")
    def get_financials(self, ticker: str, year: int | None = None) -> dict[str, Any]:
        """Pull a compact set of ratios from the latest annual report.

        Returns a dict shaped like the yfinance-equivalent fundamentals
        block so ``financial_factors.py`` can consume it interchangeably:
            {"source": "dart", "year": 2025, "ROE": 0.18, "ROA": 0.09,
             "Operating_Margin": 0.22, "Net_Margin": 0.15,
             "Revenue_Growth": 0.08, "raw": {...}}

        ``{"source": "dart"}`` with the rest missing means we could reach
        DART but couldn't extract numeric values (e.g. non-calendar FY).
        """
        if not self.ready:
            return {}
        code = _normalize_stock_code(ticker)
        if not (code.isdigit() and len(code) == 6):
            return {}
        try:
            from datetime import datetime
            use_year = int(year) if year else datetime.now().year - 1
            df = self._reader.finstate_all(code, use_year)  # type: ignore[attr-defined]
            if df is None or getattr(df, "empty", True):
                return {"source": "dart", "year": use_year}

            # OpenDartReader returns rows keyed by ``account_nm`` (Korean
            # account names). Convert to a flat ``{account: value}`` map.
            amt_col = next((c for c in ("thstrm_amount", "thstrm_add_amount")
                            if c in df.columns), None)
            if not amt_col:
                return {"source": "dart", "year": use_year}
            # ``frmtrm_amount`` is the same field for the immediately
            # preceding fiscal year, used to compute Revenue_Growth /
            # EPS_Growth without a second API round trip.
            prev_col = next((c for c in ("frmtrm_amount", "frmtrm_add_amount")
                             if c in df.columns), None)

            def _pick_value(keys: list[str], col: str) -> float | None:
                for k in keys:
                    rows = df[df.get("account_nm", "").astype(str).str.contains(k, na=False, regex=False)]
                    if not rows.empty:
                        try:
                            raw = str(rows.iloc[0][col]).replace(",", "").strip()
                            return float(raw) if raw and raw not in ("nan", "-") else None
                        except Exception:  # noqa: BLE001
                            continue
                return None

            def _pick(keys: list[str]) -> float | None:
                return _pick_value(keys, amt_col)

            def _pick_prev(keys: list[str]) -> float | None:
                return _pick_value(keys, prev_col) if prev_col else None

            revenue = _pick(["매출액", "영업수익"])
            op_income = _pick(["영업이익"])
            net_income = _pick(["당기순이익"])
            equity = _pick(["자본총계"])
            assets = _pick(["자산총계"])
            liabilities = _pick(["부채총계"])

            out: dict[str, Any] = {"source": "dart", "year": use_year}
            if equity and net_income:
                out["ROE"] = net_income / equity
            if assets and net_income:
                out["ROA"] = net_income / assets
            if revenue and op_income:
                out["Operating_Margin"] = op_income / revenue
            if revenue and net_income:
                out["Net_Margin"] = net_income / revenue

            # Solvency — derive from balance-sheet identity when the
            # explicit liability row is absent so we don't need a second
            # filing call.
            if liabilities is None and assets and equity:
                liabilities = assets - equity
            if liabilities is not None and equity:
                out["Debt_to_Equity"] = liabilities / equity
            if liabilities is not None and assets:
                out["Debt_to_Asset"] = liabilities / assets

            # Efficiency — coarse but useful when yfinance is rate-limited.
            if revenue and assets:
                out["Asset_Turnover"] = revenue / assets

            # Growth — proper YoY using ``frmtrm_amount`` (DART's own
            # previous-period column on every row), not a separate
            # ``매출액(전기)`` lookup which doesn't exist as an account.
            prev_rev = _pick_prev(["매출액", "영업수익"])
            if revenue and prev_rev:
                out["Revenue_Growth"] = (revenue - prev_rev) / prev_rev
            prev_net = _pick_prev(["당기순이익"])
            if net_income and prev_net:
                out["EPS_Growth"] = (net_income - prev_net) / prev_net

            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("DART financials failed for %s: %s", ticker, e)
            return {}


_client: DartClient | None = None


def get_dart_client() -> DartClient:
    """Module-level singleton. Cheap when DART_API_KEY is absent."""
    global _client
    if _client is None:
        _client = DartClient()
    return _client
