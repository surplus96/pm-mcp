"""SEC EDGAR — US issuer fundamentals via the XBRL Company Facts API.

Replaces yfinance for the eight standard ratios we surface on US tickers
in ``financial_factors``. Yahoo's quote endpoints rate-limit cluster
IPs aggressively (HF Spaces in particular), so a full Stock Analyzer
session on five symbols often comes back with every single ratio blank.
EDGAR is free, has no auth, and serves the same data straight from
the XBRL filings — typically 1–3 s per ticker.

Output shape matches ``dart_rest.get_financials`` so the
``financial_factors`` integration is symmetric across markets.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import requests

from mcp_server.config import sec_user_agent

logger = logging.getLogger(__name__)


SEC_BASE_URL = "https://data.sec.gov"
CIK_MAP_PATH = Path(__file__).parent.parent / "data" / "sec_cik_map.json"

# SEC requires a User-Agent identifying the caller. They publish the
# header rule at https://www.sec.gov/os/accessing-edgar-data — failing
# to set it gets us a 403. The contact comes from SEC_EDGAR_USER_AGENT.
def _headers() -> Dict[str, str]:
    return {
        "User-Agent": sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
        "Host": "data.sec.gov",
    }

# XBRL concept names we read. Each list is tried in order — issuers
# don't all tag identically (NetIncomeLoss vs ProfitLoss for IFRS
# filers, RevenueFromContractWithCustomerExcludingAssessedTax for the
# new ASC 606 tag set, etc.).
_REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
_OPERATING_INCOME_CONCEPTS = ("OperatingIncomeLoss",)
_NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
_ASSETS_CONCEPTS = ("Assets",)
_LIABILITIES_CONCEPTS = ("Liabilities",)
_EQUITY_CONCEPTS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_EPS_CONCEPTS = ("EarningsPerShareBasic", "EarningsPerShareDiluted")
# ROIC inputs.
_PRETAX_INCOME_CONCEPTS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems",
)
_TAX_EXPENSE_CONCEPTS = ("IncomeTaxExpenseBenefit",)
_LONG_TERM_DEBT_CONCEPTS = (
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
)


@lru_cache(maxsize=1)
def _load_cik_map() -> dict[str, dict[str, str]]:
    """Load the pre-baked ticker → {cik, name} mapping (~10,000 entries)."""
    if not CIK_MAP_PATH.exists():
        logger.warning("sec_cik_map.json missing at %s", CIK_MAP_PATH)
        return {}
    try:
        with CIK_MAP_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("SEC CIK mapping loaded: %d entries.", len(data))
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load sec_cik_map.json: %s", e)
        return {}


def _lookup_cik(ticker: str) -> str | None:
    t = str(ticker or "").strip().upper()
    if not t:
        return None
    # SEC normalises ``BRK.B`` style as ``BRK-B`` in their tickers file —
    # try both shapes so we don't surprise the caller.
    for candidate in (t, t.replace(".", "-")):
        entry = _load_cik_map().get(candidate)
        if entry:
            return entry.get("cik")
    return None


def _latest_annual(units: list[dict]) -> tuple[float | None, float | None, int | None]:
    """From a list of XBRL fact entries, return ``(current_year_value,
    prior_year_value, fy)`` using the most-recent ``fp == 'FY'`` row.

    Each entry looks like
    ``{"end": "2024-09-28", "val": 391035000000, "fy": 2024, "fp": "FY",
       "form": "10-K", "filed": "..."}``. Amended filings (10-K/A) are
    preferred over the original when they share a fiscal year, so the
    most-recently-filed restated number wins.
    """
    if not units:
        return None, None, None
    by_year: dict[int, dict] = {}
    for e in units:
        if e.get("fp") != "FY":
            continue
        if not str(e.get("form", "")).startswith("10-K"):
            continue
        fy = e.get("fy")
        if fy is None:
            continue
        prev = by_year.get(fy)
        if prev is None or str(e.get("filed", "")) > str(prev.get("filed", "")):
            by_year[fy] = e
    if not by_year:
        return None, None, None
    years_sorted = sorted(by_year.keys(), reverse=True)
    cur_fy = years_sorted[0]
    cur_val = by_year[cur_fy].get("val")
    prev_val = by_year[years_sorted[1]].get("val") if len(years_sorted) >= 2 else None
    try:
        cur = float(cur_val) if cur_val is not None else None
        prev = float(prev_val) if prev_val is not None else None
    except (TypeError, ValueError):
        cur, prev = None, None
    return cur, prev, cur_fy


def _is_single_quarter(e: dict) -> bool:
    """True iff a fact entry represents exactly one fiscal quarter.

    Duration concepts (IS, CF) come with ``start`` and ``end``; a real
    quarterly entry spans ~85-95 days. The naive ``fp in ('Q1','Q2','Q3')``
    check is unreliable because EDGAR also tags YTD-cumulative rows
    that way — Q2 sometimes carries the H1 sum, Q3 the 9-month sum.
    Filtering on the duration removes the YTD twins and lets a TTM
    over the right four entries actually equal the trailing year.
    """
    start = e.get("start")
    end = e.get("end")
    if not start or not end:
        return False
    try:
        from datetime import date
        s = date.fromisoformat(start)
        d = date.fromisoformat(end)
        days = (d - s).days
    except Exception:  # noqa: BLE001
        return False
    return 80 <= days <= 100


def _latest_mrq(units: list[dict]) -> tuple[float | None, float | None, str | None, str | None]:
    """Return ``(mrq_val, prior_year_same_quarter_val, end, fp)``.

    Picks the freshest true single-quarter fact (using ``_is_single_quarter``)
    and finds the matching prior-year quarter for YoY comparison.
    For balance-sheet (instant) concepts that have no ``start`` field
    we fall back to the latest entry of any fp.
    """
    if not units:
        return None, None, None, None
    # Prefer single-quarter rows for duration concepts; if none qualify
    # (instant / BS concept) fall back to all rows.
    q_rows = [e for e in units if _is_single_quarter(e)]
    pool = q_rows or list(units)
    pool.sort(key=lambda r: str(r.get("end", "")), reverse=True)
    cur = pool[0]
    cur_end = str(cur.get("end", ""))
    cur_fp = cur.get("fp")
    try:
        cur_val = float(cur["val"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, None
    # Same quarter prior year — match fp and year-1.
    prev_val = None
    cur_fy = cur.get("fy")
    if cur_fy is not None and cur_fp is not None:
        for r in pool[1:]:
            if r.get("fy") == cur_fy - 1 and r.get("fp") == cur_fp:
                try:
                    prev_val = float(r["val"])
                except (KeyError, TypeError, ValueError):
                    pass
                break
    return cur_val, prev_val, cur_end, cur_fp


def _latest_pit(units: list[dict]) -> tuple[float | None, str | None]:
    """Latest point-in-time value (balance-sheet snapshot)."""
    if not units:
        return None, None
    rows = sorted(units, key=lambda r: str(r.get("end", "")), reverse=True)
    r0 = rows[0]
    try:
        return float(r0["val"]), str(r0.get("end"))
    except (TypeError, KeyError, ValueError):
        return None, None


def _pick_concept(
    facts_block: dict,
    candidates: tuple[str, ...],
    *,
    unit_key: str = "USD",
) -> tuple[float | None, float | None, int | None]:
    """Walk ``facts.us-gaap`` and return the candidate with the latest FY.

    First-match-wins was wrong: large companies migrated to the ASC 606
    revenue tag (``RevenueFromContractWithCustomerExcludingAssessedTax``)
    around 2018 and stopped reporting under the legacy ``Revenues``
    concept, but their old data still sits in the JSON. Picking the
    first non-empty concept therefore returned 2018-era figures for
    Apple (and 2010-era for Microsoft, which split to a different
    revenue tag entirely). We compare across all candidates and keep
    the freshest fiscal year.
    """
    us_gaap = facts_block.get("us-gaap") or {}
    best: tuple[float | None, float | None, int | None] = (None, None, None)
    best_fy = -1
    for name in candidates:
        node = us_gaap.get(name)
        if not node:
            continue
        units = (node.get("units") or {}).get(unit_key) or []
        if not units:
            continue
        cur, prev, fy = _latest_annual(units)
        if cur is None or fy is None:
            continue
        if fy > best_fy:
            best = (cur, prev, fy)
            best_fy = fy
    return best


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    try:
        if den == 0:
            return None
        return num / den
    except (TypeError, ZeroDivisionError):
        return None


def get_financials(ticker: str, *, timeout: float = 8.0) -> dict[str, Any]:
    """Return 8 standard ratios for a US issuer in one EDGAR round-trip.

    Returns dict with keys ``source``, ``year``, ``cik``, and zero or
    more of ``ROE``, ``ROA``, ``Operating_Margin``, ``Net_Margin``,
    ``Debt_to_Equity``, ``Debt_to_Asset``, ``Asset_Turnover``,
    ``Revenue_Growth``, ``EPS_Growth``. Empty dict on hard failure.
    """
    cik = _lookup_cik(ticker)
    if not cik:
        logger.debug("SEC CIK lookup miss for %s", ticker)
        return {}

    url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_headers(), timeout=timeout)
        if r.status_code == 404:
            logger.debug("SEC companyfacts 404 for %s (cik=%s)", ticker, cik)
            return {}
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("SEC fetch failed for %s (cik=%s): %s", ticker, cik, e)
        return {}

    facts = data.get("facts") or {}

    revenue, prev_revenue, fy = _pick_concept(facts, _REVENUE_CONCEPTS)
    op_income, _, _ = _pick_concept(facts, _OPERATING_INCOME_CONCEPTS)
    net_income, prev_net, _ = _pick_concept(facts, _NET_INCOME_CONCEPTS)
    pretax_income, _, _ = _pick_concept(facts, _PRETAX_INCOME_CONCEPTS)
    tax_expense, _, _ = _pick_concept(facts, _TAX_EXPENSE_CONCEPTS)
    assets, _, _ = _pick_concept(facts, _ASSETS_CONCEPTS)
    liabilities, _, _ = _pick_concept(facts, _LIABILITIES_CONCEPTS)
    equity, _, _ = _pick_concept(facts, _EQUITY_CONCEPTS)
    long_term_debt, _, _ = _pick_concept(facts, _LONG_TERM_DEBT_CONCEPTS)
    eps, prev_eps, _ = _pick_concept(facts, _EPS_CONCEPTS, unit_key="USD/shares")

    if liabilities is None and assets is not None and equity is not None:
        liabilities = assets - equity

    out: dict[str, Any] = {
        "source": "sec_edgar",
        "year": fy,
        "cik": cik,
    }
    if (v := _safe_ratio(net_income, equity)) is not None:
        out["ROE"] = v
    if (v := _safe_ratio(net_income, assets)) is not None:
        out["ROA"] = v
    if (v := _safe_ratio(op_income, revenue)) is not None:
        out["Operating_Margin"] = v
    if (v := _safe_ratio(net_income, revenue)) is not None:
        out["Net_Margin"] = v
    if (v := _safe_ratio(liabilities, equity)) is not None:
        out["Debt_to_Equity"] = v
    if (v := _safe_ratio(liabilities, assets)) is not None:
        out["Debt_to_Asset"] = v
    if (v := _safe_ratio(revenue, assets)) is not None:
        out["Asset_Turnover"] = v
    if revenue is not None and prev_revenue:
        out["Revenue_Growth"] = (revenue - prev_revenue) / prev_revenue
    if eps is not None and prev_eps:
        out["EPS_Growth"] = (eps - prev_eps) / abs(prev_eps)
    elif net_income is not None and prev_net:
        out["EPS_Growth"] = (net_income - prev_net) / abs(prev_net)

    # ROIC — NOPAT / Invested Capital.
    # NOPAT = OperatingIncome × (1 - effective tax). Falls back to the
    # US federal statutory rate (21%) when the relevant XBRL tags are
    # absent. Long-term debt missing → IC = Equity only (slightly
    # inflates ROIC, conservative-bias).
    if op_income is not None and equity is not None:
        if pretax_income and tax_expense is not None and pretax_income != 0:
            effective_tax = max(0.0, min(0.5, tax_expense / pretax_income))
        else:
            effective_tax = 0.21
        nopat = op_income * (1 - effective_tax)
        invested_capital = equity + (long_term_debt or 0.0)
        if (v := _safe_ratio(nopat, invested_capital)) is not None:
            out["ROIC"] = v

    return out


def _mrq_pick(facts_block: dict, candidates: tuple[str, ...], unit_key: str = "USD") -> tuple[float | None, float | None, str | None, str | None]:
    """Freshest single-quarter value across candidates (with YoY prior)."""
    us_gaap = facts_block.get("us-gaap") or {}
    best: tuple[float | None, float | None, str | None, str | None] = (None, None, None, None)
    best_end = ""
    for name in candidates:
        node = us_gaap.get(name)
        if not node:
            continue
        units = (node.get("units") or {}).get(unit_key) or []
        if not units:
            continue
        cur, prev, end, fp = _latest_mrq(units)
        if cur is None or not end:
            continue
        if end > best_end:
            best = (cur, prev, end, fp)
            best_end = end
    return best


def _pit_pick(facts_block: dict, candidates: tuple[str, ...], unit_key: str = "USD") -> tuple[float | None, str | None]:
    """Latest point-in-time across candidates, freshest 'end' wins."""
    us_gaap = facts_block.get("us-gaap") or {}
    best: tuple[float | None, str | None] = (None, None)
    best_end = ""
    for name in candidates:
        node = us_gaap.get(name)
        if not node:
            continue
        units = (node.get("units") or {}).get(unit_key) or []
        if not units:
            continue
        val, end = _latest_pit(units)
        if val is None or not end:
            continue
        if end > best_end:
            best = (val, end)
            best_end = end
    return best


def get_quarterly_financials(ticker: str, *, timeout: float = 8.0) -> dict[str, Any]:
    """Return MRQ (most-recent-quarter) IS snapshot + latest BS snapshot.

    More current than the annual ``get_financials`` (which lags up to
    a year after fiscal close) — MRQ surfaces the latest single-quarter
    revenue / op income / net income directly from the 10-Q filing,
    together with the same-quarter-prior-year figure for YoY growth.

    Ratios computed here are quarter-level (Net_Margin = Q net / Q revenue
    etc.), not annualized — caller multiplies by 4 if an annual-equivalent
    figure is wanted. Balance-sheet items are the point-in-time snapshot
    at quarter-end so ROE / ROA blend single-quarter income against
    period-end equity / assets (i.e. quarterly returns, not annualised).
    """
    cik = _lookup_cik(ticker)
    if not cik:
        return {}

    url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_headers(), timeout=timeout)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("SEC quarterly fetch failed for %s (cik=%s): %s", ticker, cik, e)
        return {}

    facts = data.get("facts") or {}

    revenue, prev_revenue, rev_end, fp = _mrq_pick(facts, _REVENUE_CONCEPTS)
    op_income, _, _, _ = _mrq_pick(facts, _OPERATING_INCOME_CONCEPTS)
    net_income, prev_net, _, _ = _mrq_pick(facts, _NET_INCOME_CONCEPTS)
    pretax_income, _, _, _ = _mrq_pick(facts, _PRETAX_INCOME_CONCEPTS)
    tax_expense, _, _, _ = _mrq_pick(facts, _TAX_EXPENSE_CONCEPTS)
    eps_q, prev_eps_q, _, _ = _mrq_pick(facts, _EPS_CONCEPTS, unit_key="USD/shares")

    assets, _ = _pit_pick(facts, _ASSETS_CONCEPTS)
    liabilities, _ = _pit_pick(facts, _LIABILITIES_CONCEPTS)
    equity, _ = _pit_pick(facts, _EQUITY_CONCEPTS)
    long_term_debt, _ = _pit_pick(facts, _LONG_TERM_DEBT_CONCEPTS)

    if liabilities is None and assets is not None and equity is not None:
        liabilities = assets - equity

    out: dict[str, Any] = {
        "source": "sec_edgar",
        "period": "MRQ",
        "end": rev_end,
        "fp": fp,
        "cik": cik,
    }
    if (v := _safe_ratio(net_income, equity)) is not None:
        out["ROE_Q"] = v  # quarterly, not annualised
    if (v := _safe_ratio(net_income, assets)) is not None:
        out["ROA_Q"] = v
    if (v := _safe_ratio(op_income, revenue)) is not None:
        out["Operating_Margin"] = v
    if (v := _safe_ratio(net_income, revenue)) is not None:
        out["Net_Margin"] = v
    if (v := _safe_ratio(liabilities, equity)) is not None:
        out["Debt_to_Equity"] = v
    if (v := _safe_ratio(liabilities, assets)) is not None:
        out["Debt_to_Asset"] = v
    if revenue is not None and prev_revenue:
        out["Revenue_YoY"] = (revenue - prev_revenue) / abs(prev_revenue)
    if eps_q is not None and prev_eps_q:
        out["EPS_YoY"] = (eps_q - prev_eps_q) / abs(prev_eps_q)
    elif net_income is not None and prev_net:
        out["EPS_YoY"] = (net_income - prev_net) / abs(prev_net)
    out["Revenue_Q"] = revenue
    out["NetIncome_Q"] = net_income
    out["EPS_Q"] = eps_q
    return out
