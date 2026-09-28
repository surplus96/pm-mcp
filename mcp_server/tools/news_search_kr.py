"""Korean-language news search (FR-K12).

Uses Google News RSS with ``hl=ko&gl=KR&ceid=KR:ko`` so the returned
headlines are in Korean and skew toward domestic publishers (한경,
매경, 연합뉴스, …). Snippet HTML stripping reuses the ``_strip_html``
helper from :mod:`mcp_server.tools.news_search` so we don't duplicate
regex logic and so defenses against truncated ``<a href>`` tags apply
uniformly.

Why not scrape Naver Finance directly? robots.txt disallows most paths
and the HTML changes often — Google News is strictly more reliable and
attribution is already embedded per entry.
"""
from __future__ import annotations

import logging
import urllib.parse as urlparse
from datetime import timedelta
from typing import List, Dict


from mcp_server.tools.news_search import (
    _fetch_rss_feed,
    _now_utc,
    _parse_published,
    _strip_html,
    _to_iso,
)

logger = logging.getLogger(__name__)

_RSS_BASE = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def search_news_kr(
    queries: List[str], lookback_days: int = 7, max_results: int = 10,
) -> List[Dict]:
    """Korean-locale variant of ``news_search.search_news``.

    Returns the same shape — ``[{"query": str, "hits": [article, ...]}]``
    — so downstream code (``_collect_news``, flatten helpers, …) works
    without branching.
    """
    out: List[Dict] = []
    cutoff = _now_utc() - timedelta(days=lookback_days)
    for q in queries:
        q_enc = urlparse.quote(q)
        feed = _fetch_rss_feed(_RSS_BASE.format(q=q_enc))
        hits = []
        for e in getattr(feed, "entries", [])[: max_results * 2]:
            pub = _parse_published(e)
            if pub and pub < cutoff:
                continue
            raw_summary = getattr(e, "summary", "") or ""
            clean_summary = _strip_html(raw_summary)
            hits.append({
                "title": _strip_html(getattr(e, "title", "") or ""),
                "url": getattr(e, "link", ""),
                "published": _to_iso(pub) if pub else None,
                "source": getattr(getattr(e, "source", {}), "title", None)
                          or getattr(e, "source", None),
                "snippet": clean_summary[:300],
            })
            if len(hits) >= max_results:
                break
        out.append({"query": q, "hits": hits})
    return out
