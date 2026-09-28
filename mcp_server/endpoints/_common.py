"""Shared helpers for the MCP tool layer."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

Market = Literal["US", "KR"]
Period = Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# Tool behaviour hints (MCP ToolAnnotations).
READ_EXTERNAL = ToolAnnotations(read_only_hint=True, open_world_hint=True)
READ_LOCAL = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE_LOCAL = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
WRITE_EXTERNAL = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DESTRUCTIVE_LOCAL = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

ToolSpec = tuple[Callable[..., Any], ToolAnnotations]

# Keys that may accompany an ``error`` without carrying any real payload.
_ERROR_ONLY_KEYS = {"error", "ticker", "symbol", "market", "theme", "note"}


def register(mcp: MCPServer, tools: Iterable[ToolSpec]) -> None:
    for fn, annotations in tools:
        mcp.add_tool(fn, annotations=annotations)


def check(result: Any) -> Any:
    """Turn the legacy ``{"error": ...}`` return convention into a ToolError.

    Results that carry data alongside a partial ``error`` are passed through.
    """
    if isinstance(result, dict) and result.get("error") and set(result) <= _ERROR_ONLY_KEYS:
        raise ToolError(str(result["error"]))
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        only = result[0]
        if only.get("error") and set(only) <= _ERROR_ONLY_KEYS:
            raise ToolError(str(only["error"]))
    return result


def clean_tickers(tickers: Iterable[str], *, limit: int | None = None, minimum: int = 1) -> list[str]:
    out = [t.strip().upper() for t in tickers if t and t.strip()]
    if len(out) < minimum:
        raise ToolError(f"최소 {minimum}개 종목이 필요합니다.")
    return out[:limit] if limit else out


def parse_holdings(holdings_text: str) -> list:
    from mcp_server.tools.portfolio_manager import create_holdings_from_text

    holdings = create_holdings_from_text(holdings_text)
    if not holdings:
        raise ToolError("보유 종목을 파싱할 수 없습니다. 형식: TICKER:SHARES@ENTRY_PRICE, ...")
    return holdings
