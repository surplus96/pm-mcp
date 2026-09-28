"""PM-MCP over Streamable HTTP (endpoint: /mcp).

Environment:
    HOST / PORT     bind address (default 127.0.0.1:8010)
    PM_MCP_TOKEN    when set, every request must send ``Authorization: Bearer <token>``.
                    Required when HOST is not a loopback address.
    ALLOWED_HOSTS   comma-separated Host header values accepted when HOST is not
                    loopback (e.g. ``mcp.example.com,mcp.example.com:443``).
"""
from __future__ import annotations

import hmac
import os

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp_server.mcp_app import mcp

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class BearerTokenMiddleware:
    """Reject HTTP requests that do not carry the expected bearer token."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            supplied = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                response = JSONResponse({"error": "unauthorized"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(host: str | None = None, token: str | None = None) -> ASGIApp:
    host = host or os.getenv("HOST", "127.0.0.1")
    token = token if token is not None else os.getenv("PM_MCP_TOKEN", "")
    if host not in LOOPBACK and not token:
        raise RuntimeError("PM_MCP_TOKEN must be set when binding to a non-loopback address.")

    security = None
    if host not in LOOPBACK:
        allowed = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=bool(allowed),
            allowed_hosts=allowed,
        )

    app: Starlette = mcp.streamable_http_app(host=host, transport_security=security)
    return BearerTokenMiddleware(app, token) if token else app


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(build_app(host), host=host, port=int(os.getenv("PORT", "8010")), log_level="info")
