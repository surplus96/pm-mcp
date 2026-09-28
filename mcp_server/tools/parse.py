from __future__ import annotations
from typing import List, Dict
import re


def parse_holdings_text(text: str) -> List[Dict]:
    """
    자연어에 가까운 간단한 보유주 입력을 파싱합니다.
    지원 예:
      - "AAPL LLY NVO"
      - "AAPL@2024-10-01:185, LLY@2024-09-15:520, NVO"
      - 줄바꿈/쉼표/공백 구분, 각 토큰은 다음 형태를 지원:
         TICKER
         TICKER DATE PRICE
         TICKER@DATE:PRICE
    반환: [{ticker, entry_date?, entry_price?}]
    """
    if not text:
        return []
    items: List[str] = []
    for part in re.split(r"[\n,]", text):
        part = part.strip()
        if part:
            items.extend(part.split()) if ("@" not in part and ":" not in part and len(part.split()) > 1) else [part]

    results: List[Dict] = []
    token_re = re.compile(r"^(?P<ticker>[A-Za-z\.\-]+)(?:@(?P<date>\d{4}-\d{2}-\d{2}))?(?::(?P<price>\d+(?:\.\d+)?))?$")

    i = 0
    while i < len(items):
        raw = items[i].strip()
        m = token_re.match(raw)
        if m:
            ticker = m.group("ticker").upper()
            entry_date = m.group("date")
            price_s = m.group("price")
            entry_price = float(price_s) if price_s else None
            results.append({"ticker": ticker, "entry_date": entry_date, "entry_price": entry_price})
            i += 1
            continue
        # 패턴 미일치: "AAPL 2024-10-01 185" 스타일 처리
        parts = raw.split()
        if len(parts) >= 3 and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[1]) and re.match(r"^\d+(?:\.\d+)?$", parts[2]):
            ticker = parts[0].upper()
            results.append({"ticker": ticker, "entry_date": parts[1], "entry_price": float(parts[2])})
            i += 1
            continue
        # 단일 토큰 티커로 간주
        results.append({"ticker": raw.upper(), "entry_date": None, "entry_price": None})
        i += 1
    # 중복 제거(앞쪽 우선)
    seen = set()
    dedup: List[Dict] = []
    for r in results:
        t = r.get("ticker")
        if t and t not in seen:
            dedup.append(r)
            seen.add(t)
    return dedup


