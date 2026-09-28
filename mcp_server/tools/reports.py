from __future__ import annotations
from typing import Dict
from jinja2 import Template

DEFAULT_TEMPLATE = Template(
"""
# {{ title }}

**Date**: {{ date }}
**Tickers**: {{ tickers|join(', ') }}

## Summary
{{ summary }}

## News (Condensed)
{% if news_summary %}{{ news_summary }}{% else %}_No news summary_
{% endif %}

## SEC Filings (Condensed)
{% if filings_summary %}{{ filings_summary }}{% else %}_No filings summary_
{% endif %}

## Scores
| Ticker | Base | Dip bonus | Total |
|---|---:|---:|---:|
{% for t in scores %}| {{ t.ticker }} | {{ '%.3f'|format(t.base_score) }} | {{ '%.3f'|format(t.dip_bonus) }} | {{ '%.3f'|format(t.score) }} |
{% endfor %}

## Factor Evidence
| Ticker | Sector | PE | PB | EPS | ROE | RevG | ProfitM | Mom(3/6/12) | Event |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{% for t in scores %}| {{ t.ticker }} | {{ t.sector or '' }} | {{ t.pe if t.pe is not none else '' }} | {{ t.pb if t.pb is not none else '' }} | {{ t.eps if t.eps is not none else '' }} | {{ t.returnOnEquity if t.returnOnEquity is not none else '' }} | {{ t.revenueGrowth if t.revenueGrowth is not none else '' }} | {{ t.profitMargins if t.profitMargins is not none else '' }} | {{ '%.3f/%.3f/%.3f'|format(t.mom3 or 0.0, t.mom6 or 0.0, t.mom12 or 0.0) }} | {{ '%.3f'|format(t.eventScore or 0.0) }} |
{% endfor %}
"""
)


def generate_report(payload: Dict) -> str:
    return DEFAULT_TEMPLATE.render(**payload)
