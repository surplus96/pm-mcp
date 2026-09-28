#!/bin/zsh
# Create/refresh .venv and run the MCP server over stdio.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source ./.env; set +a
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
exec .venv/bin/python -m mcp_server.mcp_app
