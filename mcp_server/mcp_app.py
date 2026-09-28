"""PM-MCP server entry point (stdio).

Run with ``python -m mcp_server.mcp_app``. Tools, resources and prompts are
defined in :mod:`mcp_server.endpoints`.
"""
from __future__ import annotations

from mcp_server.endpoints import create_server

mcp = create_server()


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
