"""Offline tests for the MCP server (no network access)."""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pandas as pd
import pytest
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from mcp_server import mcp_app
from mcp_server.config import safe_filename
from mcp_server.endpoints import analysis, data
from mcp_server.endpoints._common import check
from mcp_server.tools import market_data

EXPECTED_TOOLS = {
    "theme_propose", "theme_explore", "theme_analyze", "dip_candidates",
    "stock_snapshot", "stock_factors", "stock_compare", "stock_rank", "market_overview", "backtest_strategy",
    "market_prices", "news_search", "news_sentiment", "text_sentiment", "filings_recent", "finnhub_data",
    "portfolio_analyze", "portfolio_quick_check", "portfolio_store", "watchlist",
    "chart", "report_create",
    "data_quality", "ops_status", "ops_action",
}


def run(coro):
    return asyncio.run(coro)


def _ohlcv(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=rows, freq="B"),
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, rows),
    })


# ---- registration -------------------------------------------------------

def test_tool_set_and_metadata():
    tools = run(mcp_app.mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.description, t.name
        assert t.annotations is not None, t.name
        assert "ctx" not in t.input_schema.get("properties", {}), t.name


def test_resources_and_prompts_registered():
    uris = {str(r.uri) for r in run(mcp_app.mcp.list_resources())}
    assert {"pm://watchlist", "pm://portfolios", "pm://reference/sector-weights"} <= uris
    assert {p.name for p in run(mcp_app.mcp.list_prompts())} == {"analyze_stock", "portfolio_checkup", "discover_themes"}


# ---- helpers ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [("AAPL", "AAPL"), ("../../etc/passwd", "_.._etc_passwd"), ("AI Growth Stocks", "AI Growth Stocks"), ("..", "untitled")],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected
    assert os.sep not in safe_filename(raw)


def test_check_converts_error_only_results():
    with pytest.raises(ToolError, match="boom"):
        check({"error": "boom", "ticker": "AAPL"})
    with pytest.raises(ToolError):
        check([{"error": "boom"}])
    partial = {"error": "partial", "rows": [1]}
    assert check(partial) is partial


# ---- cache --------------------------------------------------------------

def test_cache_skips_empty_results(monkeypatch, tmp_path):
    from mcp_server.tools.cache_manager import CacheManager

    monkeypatch.setattr(CacheManager, "_instance", None)  # singleton: isolate from data/diskcache
    cm = CacheManager(cache_dir=str(tmp_path))
    calls = []

    @cm.cached(ttl=60, prefix="t")
    def fetch(n):
        calls.append(n)
        return _ohlcv(n) if n else pd.DataFrame()

    fetch(0); fetch(0)
    assert calls == [0, 0]  # empty frame is not cached
    fetch(5); fetch(5)
    assert calls == [0, 0, 5]  # non-empty frame is cached


# ---- SEC User-Agent ------------------------------------------------------

def test_sec_user_agent_comes_from_env(monkeypatch):
    from mcp_server.tools import filings, sec_edgar_fundamentals

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Tester tester@example.org")
    assert filings._headers()["User-Agent"] == "Tester tester@example.org"
    assert sec_edgar_fundamentals._headers()["User-Agent"] == "Tester tester@example.org"
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    assert "@" in sec_edgar_fundamentals._headers()["User-Agent"]


# ---- market data --------------------------------------------------------

def test_get_prices_accepts_market_and_period(monkeypatch):
    calls = {}

    def fake_download(ticker, start, end, interval):
        calls.update(ticker=ticker, start=start, end=end)
        return _ohlcv(5).set_index("Date")

    monkeypatch.setattr(market_data, "_download_prices", fake_download)
    df = market_data.get_prices.__wrapped__("AAPL", market="US", period="6mo")
    assert not df.empty
    assert (pd.Timestamp(calls["end"]) - pd.Timestamp(calls["start"])).days == 180


def test_market_prices_paginates_with_iso_dates(monkeypatch):
    monkeypatch.setattr(market_data, "get_prices", lambda *a, **k: _ohlcv(250))
    page = data.market_prices("AAPL", page_size=100, cursor=200)
    assert page["total_rows"] == 250 and len(page["rows"]) == 50 and page["next_cursor"] is None
    assert page["rows"][0]["Date"].startswith("2025-")


def test_empty_prices_surface_as_tool_error(monkeypatch):
    monkeypatch.setattr(market_data, "get_prices", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ToolError, match="가격 데이터가 없습니다"):
        run(mcp_app.mcp.call_tool("market_prices", {"ticker": "ZZZZ"}))


# ---- analysis -----------------------------------------------------------

def test_stock_factors_technical_only(monkeypatch):
    monkeypatch.setattr(market_data, "get_prices", lambda *a, **k: _ohlcv())
    result = analysis.stock_factors("AAPL", sections=["technical"])
    assert set(result["factors"]) == {"technical"}
    assert 0 <= result["composite_score"] <= 100
    assert result["recommendation"]
    assert "technical" in result["section_scores"]


def test_stock_factors_all_sections_failing_raises(monkeypatch):
    monkeypatch.setattr(market_data, "get_prices", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ToolError):
        analysis.stock_factors("AAPL", sections=["technical"])


def test_backtest_rejects_inverted_thresholds():
    with pytest.raises(ToolError, match="sell_threshold"):
        run(mcp_app.mcp.call_tool("backtest_strategy", {"ticker": "AAPL", "buy_threshold": 40, "sell_threshold": 60}))


# ---- portfolio / watchlist ----------------------------------------------

def test_portfolio_store_roundtrip_and_path_safety(monkeypatch, tmp_path):
    from mcp_server.endpoints.portfolio import portfolio_store
    from mcp_server.tools import portfolio_manager as pm

    monkeypatch.setattr(pm, "PORTFOLIO_DATA_DIR", str(tmp_path))
    saved = portfolio_store("save", name="../../escape", holdings_text="AAPL:10@150, MSFT:5@400", cash=100)
    assert os.path.dirname(saved["filepath"]) == str(tmp_path)
    assert portfolio_store("list")["count"] == 1
    loaded = portfolio_store("load", name="../../escape")
    assert [h["ticker"] for h in loaded["holdings"]] == ["AAPL", "MSFT"]
    with pytest.raises(ToolError):
        portfolio_store("load", name="missing")


def test_portfolio_analyze_rejects_bad_holdings():
    with pytest.raises(ToolError, match="파싱"):
        run(mcp_app.mcp.call_tool("portfolio_analyze", {"holdings_text": "???"}))


def test_watchlist_update(monkeypatch, tmp_path):
    from mcp_server import config
    from mcp_server.endpoints.portfolio import watchlist

    path = tmp_path / "watchlist.json"
    monkeypatch.setattr(config, "WATCHLIST_PATH", str(path))
    assert watchlist("get") == {"tickers": [], "themes": []}
    watchlist("update", tickers=["aapl", " nvda "])
    assert json.loads(path.read_text())["tickers"] == ["AAPL", "NVDA"]


# ---- resources / prompts ------------------------------------------------

def test_sector_weights_resource():
    contents = list(run(mcp_app.mcp.read_resource("pm://reference/sector-weights")))
    body = json.loads(contents[0].content)
    assert "default" in body and "sectors" in body


def test_missing_portfolio_resource(monkeypatch, tmp_path):
    from mcp_server.tools import portfolio_manager as pm

    monkeypatch.setattr(pm, "PORTFOLIO_DATA_DIR", str(tmp_path))
    with pytest.raises(ResourceError):
        run(mcp_app.mcp.read_resource("pm://portfolios/nope"))


def test_prompt_mentions_ticker():
    result = run(mcp_app.mcp.get_prompt("analyze_stock", {"ticker": "NVDA"}))
    assert "NVDA" in result.messages[0].content.text


# ---- HTTP transport -----------------------------------------------------

def _asgi_status(app, headers):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers, "query_string": b"",
             "server": ("127.0.0.1", 8010), "scheme": "http", "root_path": "", "http_version": "1.1"}
    asyncio.run(app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_http_requires_bearer_token():
    from mcp_server.mcp_app_http import build_app

    app = build_app(host="127.0.0.1", token="s3cret")
    assert _asgi_status(app, []) == 401
    assert _asgi_status(app, [(b"authorization", b"Bearer wrong")]) == 401


def test_http_refuses_public_bind_without_token():
    from mcp_server.mcp_app_http import build_app

    with pytest.raises(RuntimeError, match="PM_MCP_TOKEN"):
        build_app(host="0.0.0.0", token="")


def test_lifespan_routes_print_to_stderr(capsys):
    from mcp_server.endpoints import _lifespan

    async def go():
        async with _lifespan(None):
            print("library noise")

    run(go())
    captured = capsys.readouterr()
    assert "library noise" in captured.err
    assert "library noise" not in captured.out
