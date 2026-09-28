"""Regenerate the tool tables in skills/pm-mcp-director/references/tool-catalog.md.

Run from the repo root:  .venv/bin/python scripts/gen_tool_catalog.py
The tables come from the live server schema, so they always match the code.
Everything from "## 리소스" onward is kept as-is (hand-written).
"""
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp_server.endpoints import analysis, data, discovery, ops, output, portfolio  # noqa: E402
from mcp_server.mcp_app import mcp  # noqa: E402

CATALOG = os.path.join(ROOT, "skills", "pm-mcp-director", "references", "tool-catalog.md")
SECTIONS = [
    ("테마/아이디어 발굴", discovery),
    ("종목 분석·랭킹·백테스트", analysis),
    ("시장 데이터·뉴스·공시", data),
    ("포트폴리오·워치리스트", portfolio),
    ("차트·리포트", output),
    ("데이터 품질·운영", ops),
]
HEADER = """# pm-mcp 전체 도구 카탈로그 (v2, 도구 {count}개)

모든 도구는 `mcp__pm-mcp__<이름>` 형태로 호출한다. **굵게**는 필수, `이름(값)`은 기본값, `이름?`은 선택값, `∈`는 허용 값이다.
아래 표는 `scripts/gen_tool_catalog.py`가 서버 스키마에서 생성한다. 도구를 바꾸면 스크립트를 다시 실행한다.
"""


def _enum(schema: dict) -> list | None:
    for s in [schema, schema.get("items", {}), *schema.get("anyOf", [])]:
        if "enum" in s:
            return s["enum"]
        if "items" in s and "enum" in s["items"]:
            return s["items"]["enum"]
    return None


def _param(name: str, schema: dict, required: bool) -> str:
    enum = _enum(schema)
    if required:
        text = f"**{name}**"
    elif schema.get("default") is None:
        text = f"{name}?"
    else:
        text = f"{name}({schema['default']!r})"
    return f"{text} ∈ {'/'.join(map(str, enum))}" if enum else text


def _summary(description: str) -> str:
    """First paragraph of the docstring, joined onto one line."""
    return " ".join(description.strip().split("\n\n")[0].split()).replace("|", "\\|")


def build() -> str:
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    out = [HEADER.format(count=len(tools))]
    for title, module in SECTIONS:
        out += [f"## {title}", "| 도구 | 파라미터 | 설명 |", "|---|---|---|"]
        for fn, _ in module.TOOLS:
            tool = tools[fn.__name__]
            schema = tool.input_schema
            required = set(schema.get("required", []))
            params = ", ".join(_param(n, s, n in required) for n, s in schema.get("properties", {}).items())
            out.append(f"| `{tool.name}` | {params} | {_summary(tool.description)} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    with open(CATALOG, encoding="utf-8") as f:
        current = f.read()
    tail = current[current.index("## 리소스"):]
    with open(CATALOG, "w", encoding="utf-8") as f:
        f.write(build() + "\n" + tail)
    print(f"updated {os.path.relpath(CATALOG, ROOT)}")


if __name__ == "__main__":
    main()
