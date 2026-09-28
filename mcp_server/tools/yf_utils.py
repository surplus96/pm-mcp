"""yfinance DataFrame column normalization utility.

yfinance >= 0.2.31 may return MultiIndex columns even for single tickers,
e.g. ("Close", "AAPL") instead of "Close".  This module provides a single
function that flattens columns so downstream code can always use df["Close"].
"""
from __future__ import annotations

import pandas as pd


def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex / comma-joined columns to simple names.

    Handles three cases:
    1. MultiIndex ("Close", "AAPL") -> "Close"
    2. Comma-joined "Close,AAPL"   -> "Close"
    3. Already flat "Close"        -> passthrough

    The function operates **in-place** on the column index and also returns
    the same DataFrame for convenience.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        return df

    new_cols = []
    changed = False
    for col in df.columns:
        if isinstance(col, str) and "," in col:
            new_cols.append(col.split(",")[0].strip())
            changed = True
        else:
            new_cols.append(col)
    if changed:
        df.columns = new_cols

    return df


def is_yfinance_supported(ticker: str, market: str = "US") -> bool:
    """False for KR special-listing codes Yahoo Finance doesn't index.

    Pattern-based, not market-arg-based — ``market`` is accepted for
    backwards compatibility but ignored. Empirically several callers
    (stock_report.py among them) hard-code ``market="US"`` regardless
    of the actual ticker, and ``yf_utils.detect_market`` itself can't
    classify a 6-char alphanumeric like ``0001A0`` so it returns ``US``
    too. Trusting either signal would let unsupported tickers slip
    through and re-trigger the yfinance 404 cascade we're trying to
    suppress, so the test runs purely on the ticker shape.

    Pass-through cases (return True):
        - Plain alpha (US): ``AAPL``, ``MSFT``, ``BRK.A``
        - 6-digit numeric (KR common stock — wraps to ``.KS`` / ``.KQ``)
        - Already-suffixed KR: ``005930.KS``, ``247540.KQ``

    Short-circuit cases (return False):
        - 6-char mixed alphanumeric KRX special listings (REIT, ETN,
          A-prefix stock-loan, ELW): ``0001A0`` (덕양에너젠), ``A12345``
        - Empty / None
    """
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    # Strip yfinance-style KR suffix before shape testing.
    bare = t.replace(".KS", "").replace(".KQ", "")
    if not bare:
        return False
    # 6-char ticker that mixes letters and digits = KRX special listing.
    # ``isdigit() == False and isalpha() == False and isalnum() == True``
    # is the precise mixed-class check; ``"AAPL"`` (5 alpha) and
    # ``"005930"`` (6 digit) both pass through, ``"0001A0"`` does not.
    if len(bare) == 6 and bare.isalnum() and not bare.isdigit() and not bare.isalpha():
        return False
    return True


def normalize_ticker_multi_market(ticker: str, market: str = "US") -> str:
    """Normalize ticker symbol for multi-market yfinance lookup.

    Rules:
    - US market: passthrough (uppercased, stripped)
    - KR market: append ``.KS`` (KOSPI) to 6-digit codes unless already suffixed.
      If the ticker already ends with ``.KS`` or ``.KQ``, it is returned as-is.

    This mirrors yfinance's convention where Korean equities are queried as
    ``005930.KS`` (Samsung Electronics, KOSPI) or ``247540.KQ`` (Ecopro BM, KOSDAQ).

    Args:
        ticker: Raw ticker symbol (e.g. ``"AAPL"``, ``"005930"``, ``"005930.KS"``)
        market: Market code, ``"US"`` (default) or ``"KR"``

    Returns:
        Normalized ticker ready for ``yfinance.Ticker()``.
    """
    if ticker is None:
        return ""
    t = str(ticker).strip().upper()
    if not t:
        return t

    m = (market or "US").strip().upper()
    if m == "KR":
        # Already suffixed — keep as-is.
        if t.endswith(".KS") or t.endswith(".KQ"):
            return t
        # 6-digit Korean code → ask the lookup helper which suffix
        # actually matches (KOSPI vs KOSDAQ). Defaulting to ``.KS`` for
        # everything used to break KOSDAQ-listed names like ``247540``
        # (에코프로비엠) because yfinance returns 404 for the wrong
        # suffix.
        if t.isdigit() and len(t) == 6:
            from mcp_server.tools.kr_market_lookup import kr_yfinance_symbol
            return kr_yfinance_symbol(t)
        return t

    # US / default → passthrough
    return t


def detect_market(ticker: str) -> str:
    """Classify a ticker as ``"KR"`` or ``"US"`` based on surface form.

    Rules (FR-K01):
      * ``.KS`` / ``.KQ`` suffix → KR
      * 6 consecutive digits → KR (Korean stock codes are exactly 6)
      * contains Hangul (``[가-힣]``) → KR (Korean company name input)
      * anything else → US (alphabetic tickers, ``BRK.A`` style, …)

    The heuristic is deliberately simple — ambiguous cases fall back to
    the US path which uses yfinance, matching pre-integration behaviour.
    """
    if not ticker:
        return "US"
    raw = str(ticker).strip()
    # Check Hangul before uppercasing because .upper() is a no-op on CJK
    # but lowercase/uppercase Korean literals compare equal anyway.
    import re as _re
    if _re.search(r"[가-힣]", raw):
        return "KR"
    t = raw.upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KR"
    if t.isdigit() and len(t) == 6:
        return "KR"
    # 6-char alphanumeric mixing letters and digits → KRX special
    # listing (REIT/ETN/A-prefix stock-loan/ELW), e.g. ``0001A0`` for
    # 덕양에너젠. US tickers are pure alpha (``AAPL``) or alpha + dot
    # (``BRK.A``) and never reach 6 chars by mixing classes, so this is
    # an unambiguous KR signal. Without this branch the whole KR routing
    # chain (KIS/PyKrx/DART) gets bypassed and the frontend shows N/A
    # everywhere because data goes to yfinance under "US".
    if len(t) == 6 and t.isalnum() and not t.isalpha():
        return "KR"
    return "US"
