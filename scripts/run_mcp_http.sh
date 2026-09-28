#!/bin/bash
# PM-MCP over Streamable HTTP (endpoint: http://HOST:PORT/mcp).
# Defaults to 127.0.0.1:8010. Binding to a non-loopback HOST requires
# PM_MCP_TOKEN (clients send "Authorization: Bearer <token>") and
# ALLOWED_HOSTS for DNS-rebinding protection.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source ./.env; set +a
fi

exec .venv/bin/python -m mcp_server.mcp_app_http
