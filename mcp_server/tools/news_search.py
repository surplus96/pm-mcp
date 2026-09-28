from __future__ import annotations
from typing import List, Dict
from datetime import datetime, timedelta
import json
import hashlib
import re
import urllib.parse as urlparse
import logging
import feedparser

from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.resilience import circuit_rss, CircuitOpenError

logger = logging.getLogger(__name__)


# Pre-compiled HTML strippers — used before snippet truncation so callers never
# receive unterminated tags (truncation can otherwise slice through `<a href="..."`).
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_UNCLOSED_TAIL_RE = re.compile(r"<[^<>]*$")
_ENTITY_REPLACEMENTS = (
    ("&nbsp;", " "),
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&apos;", "'"),
)
_NUMERIC_ENTITY_RE = re.compile(r"&#\d+;")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags, decode common entities, collapse whitespace.

    Handles unterminated trailing tags (`<a href="...`) defensively so this
    function is safe to call *before* any truncation.
    """
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = _HTML_UNCLOSED_TAIL_RE.sub("", text)  # defensively strip any dangling `<...`
    for needle, repl in _ENTITY_REPLACEMENTS:
        text = text.replace(needle, repl)
    text = _NUMERIC_ENTITY_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _now_utc():
    return datetime.utcnow()


def _to_iso(dt):
    try:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _parse_published(entry) -> datetime | None:
    try:
        if getattr(entry, "published_parsed", None):
            import time
            return datetime.utcfromtimestamp(time.mktime(entry.published_parsed))
    except Exception:
        return None
    return None


def _fetch_rss_feed(url: str) -> dict:
    """RSS 피드 조회 (서킷 브레이커 + 타임아웃)"""
    try:
        return circuit_rss.call(
            lambda: feedparser.parse(url, request_headers={"User-Agent": "PM-MCP/1.0"})
        )
    except CircuitOpenError:
        logger.warning(f"RSS circuit open, skipping: {url}")
        return {"entries": []}
    except Exception as e:
        logger.warning(f"RSS fetch error: {e}")
        return {"entries": []}


def _search_news_rss(queries: List[str], lookback_days: int = 7, max_results: int = 10) -> List[Dict]:
    out: List[Dict] = []
    cutoff = _now_utc() - timedelta(days=lookback_days)
    for q in queries:
        q_enc = urlparse.quote(q)
        rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
        feed = _fetch_rss_feed(rss_url)
        hits = []
        for e in getattr(feed, "entries", [])[: max_results * 2]:
            pub = _parse_published(e)
            if pub and pub < cutoff:
                continue
            # Strip HTML BEFORE truncating, otherwise a 300-char cut can
            # slice through an <a href="..."> tag and leave an unterminated
            # `<a href="..."` that downstream HTML-strip regexes cannot match.
            raw_summary = getattr(e, "summary", "") or ""
            clean_summary = _strip_html(raw_summary)
            hits.append({
                "title": _strip_html(getattr(e, "title", "") or ""),
                "url": getattr(e, "link", ""),
                "published": _to_iso(pub) if pub else None,
                "source": getattr(getattr(e, "source", {}), "title", None) or getattr(e, "source", None),
                "snippet": clean_summary[:300],
            })
            if len(hits) >= max_results:
                break
        out.append({"query": q, "hits": hits})
    return out


def search_news(queries: List[str], lookback_days: int = 7, max_results: int = 10, use_cache: bool = True) -> List[Dict]:
    """뉴스 검색 (Google News RSS). 1시간 캐시 적용.
    반환 스키마: [{ query, hits: [{title, url, published, source, snippet}] }]
    """
    if use_cache:
        key_data = json.dumps({"queries": sorted(queries), "lookback": lookback_days, "max": max_results}, sort_keys=True)
        cache_key = f"news:{hashlib.md5(key_data.encode()).hexdigest()[:12]}"
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

    result = _search_news_rss(queries, lookback_days=lookback_days, max_results=max_results)

    if use_cache and result:
        cache_manager.set(cache_key, result, TTL.NEWS)

    return result
