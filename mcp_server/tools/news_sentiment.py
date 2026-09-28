"""
뉴스 감성 분석 모듈
- 키워드 기반 감성 분류
- LLM 기반 고급 분석
- 중복 제거 및 클러스터링
- 영향도 평가
- 뉴스 타임라인
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal
from difflib import SequenceMatcher
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from mcp_server.tools.cache_manager import cache_manager, TTL

logger = logging.getLogger(__name__)


# ============================================================
# 감성 분석 키워드 사전
# ============================================================

SENTIMENT_KEYWORDS = {
    "strong_positive": {
        "keywords": [
            "surge", "soar", "skyrocket", "breakthrough", "record high", "beat expectations",
            "exceeds estimates", "outperform", "upgrade", "buy rating", "strong buy",
            "blockbuster", "explosive growth", "blowout earnings", "all-time high",
            "massive gains", "rally", "boom", "stellar", "exceptional"
        ],
        "score": 0.9
    },
    "positive": {
        "keywords": [
            "gain", "rise", "increase", "growth", "profit", "beat", "exceed",
            "bullish", "optimistic", "positive", "improve", "recovery", "upside",
            "momentum", "strength", "advance", "expand", "success", "win", "approval",
            "partnership", "deal", "contract", "launch", "innovation"
        ],
        "score": 0.6
    },
    "weak_positive": {
        "keywords": [
            "steady", "stable", "maintain", "hold", "in-line", "meet expectations",
            "modest", "slight gain", "gradual", "potential", "opportunity"
        ],
        "score": 0.3
    },
    "neutral": {
        "keywords": [
            "announce", "report", "update", "plan", "schedule", "file", "submit",
            "appoint", "change", "transition", "move", "consider", "evaluate"
        ],
        "score": 0.0
    },
    "weak_negative": {
        "keywords": [
            "concern", "caution", "uncertainty", "challenge", "headwind", "pressure",
            "slight decline", "modest loss", "below expectations", "underperform",
            "mixed results", "volatile", "risk"
        ],
        "score": -0.3
    },
    "negative": {
        "keywords": [
            "drop", "fall", "decline", "loss", "miss", "cut", "reduce", "layoff",
            "downgrade", "sell rating", "bearish", "pessimistic", "weak", "struggle",
            "disappointing", "shortfall", "deficit", "lawsuit", "investigation"
        ],
        "score": -0.6
    },
    "strong_negative": {
        "keywords": [
            "crash", "plunge", "collapse", "crisis", "bankruptcy", "fraud", "scandal",
            "massive loss", "severe", "catastrophic", "devastate", "default",
            "recall", "halt", "suspend", "terminate", "investigation", "sec probe"
        ],
        "score": -0.9
    }
}

# 주가 영향도 키워드
IMPACT_KEYWORDS = {
    "high_impact": {
        "keywords": [
            "earnings", "revenue", "guidance", "forecast", "fda", "approval",
            "merger", "acquisition", "buyout", "ipo", "split", "dividend",
            "ceo", "cfo", "executive", "sec", "lawsuit", "investigation",
            "recall", "bankruptcy", "default", "patent", "breakthrough"
        ],
        "weight": 1.0
    },
    "medium_impact": {
        "keywords": [
            "analyst", "rating", "upgrade", "downgrade", "price target",
            "contract", "partnership", "deal", "expansion", "launch",
            "product", "service", "market share", "competition"
        ],
        "weight": 0.7
    },
    "low_impact": {
        "keywords": [
            "conference", "presentation", "interview", "comment", "opinion",
            "speculation", "rumor", "report", "analysis"
        ],
        "weight": 0.4
    }
}


class NewsSentimentAnalyzer:
    """뉴스 감성 분석기"""

    def __init__(self):
        self.sentiment_cache = {}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트 감성 분석 (키워드 기반)

        Args:
            text: 분석할 텍스트

        Returns:
            감성 분석 결과
        """
        if not text:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        text_lower = text.lower()
        scores = []
        matched_keywords = []

        # 각 감성 카테고리에서 키워드 매칭
        for category, data in SENTIMENT_KEYWORDS.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    scores.append(data["score"])
                    matched_keywords.append({
                        "keyword": keyword,
                        "category": category,
                        "score": data["score"]
                    })

        # 점수 계산
        if scores:
            avg_score = sum(scores) / len(scores)
            # 매칭된 키워드 수에 따른 신뢰도
            confidence = min(1.0, len(scores) / 5)
        else:
            avg_score = 0.0
            confidence = 0.2  # 키워드 없으면 낮은 신뢰도

        # 감성 레이블 결정
        if avg_score > 0.5:
            sentiment = "positive"
        elif avg_score > 0.1:
            sentiment = "somewhat_positive"
        elif avg_score < -0.5:
            sentiment = "negative"
        elif avg_score < -0.1:
            sentiment = "somewhat_negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": round(avg_score, 3),
            "confidence": round(confidence, 2),
            "matched_keywords": matched_keywords[:10]  # 상위 10개만
        }

    def analyze_impact(self, text: str) -> Dict[str, Any]:
        """
        뉴스 영향도 평가

        Args:
            text: 뉴스 텍스트

        Returns:
            영향도 평가 결과
        """
        if not text:
            return {"impact": "low", "score": 0.0, "factors": []}

        text_lower = text.lower()
        impact_scores = []
        factors = []

        for level, data in IMPACT_KEYWORDS.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    impact_scores.append(data["weight"])
                    factors.append({
                        "keyword": keyword,
                        "level": level,
                        "weight": data["weight"]
                    })

        if impact_scores:
            max_score = max(impact_scores)
            avg_score = sum(impact_scores) / len(impact_scores)
            final_score = (max_score + avg_score) / 2
        else:
            final_score = 0.3

        # 영향도 레벨 결정
        if final_score >= 0.8:
            impact_level = "high"
        elif final_score >= 0.5:
            impact_level = "medium"
        else:
            impact_level = "low"

        return {
            "impact": impact_level,
            "score": round(final_score, 2),
            "factors": factors[:5]
        }

    def analyze_news_item(self, news: Dict) -> Dict[str, Any]:
        """
        단일 뉴스 아이템 종합 분석

        Args:
            news: 뉴스 아이템 (title, snippet 필드 포함)

        Returns:
            종합 분석 결과
        """
        title = news.get("title", "")
        snippet = news.get("snippet", "") or news.get("summary", "")
        combined_text = f"{title} {snippet}"

        # 감성 분석
        sentiment = self.analyze_text(combined_text)

        # 제목은 가중치 높게
        title_sentiment = self.analyze_text(title)
        if title_sentiment["confidence"] > 0:
            # 제목 감성에 가중치 부여
            combined_score = (sentiment["score"] + title_sentiment["score"] * 1.5) / 2.5
            sentiment["score"] = round(combined_score, 3)

        # 영향도 분석
        impact = self.analyze_impact(combined_text)

        # 종합 점수 (감성 * 영향도)
        composite_score = sentiment["score"] * impact["score"]

        return {
            **news,
            "sentiment": sentiment["sentiment"],
            "sentiment_score": sentiment["score"],
            "sentiment_confidence": sentiment["confidence"],
            "impact": impact["impact"],
            "impact_score": impact["score"],
            "composite_score": round(composite_score, 3),
            "keywords": sentiment.get("matched_keywords", [])[:5]
        }


# ============================================================
# 중복 제거 및 클러스터링
# ============================================================

class NewsDeduplicator:
    """뉴스 중복 제거 및 클러스터링"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.threshold = similarity_threshold

    def similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def deduplicate(self, news_items: List[Dict]) -> List[Dict]:
        """
        중복 뉴스 제거

        Args:
            news_items: 뉴스 리스트

        Returns:
            중복 제거된 뉴스 리스트
        """
        if not news_items:
            return []

        unique = []
        seen_titles = []

        for item in news_items:
            title = item.get("title", "")
            is_duplicate = False

            for seen_title in seen_titles:
                if self.similarity(title, seen_title) > self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(item)
                seen_titles.append(title)

        return unique

    def cluster_by_topic(self, news_items: List[Dict], num_clusters: int = 5) -> Dict[str, List[Dict]]:
        """
        뉴스를 주제별로 클러스터링 (간단한 키워드 기반)

        Args:
            news_items: 뉴스 리스트
            num_clusters: 최대 클러스터 수

        Returns:
            주제별 클러스터
        """
        # 토픽 키워드 정의
        topic_keywords = {
            "Earnings & Financials": ["earning", "revenue", "profit", "loss", "quarter", "fiscal", "eps", "guidance"],
            "M&A & Deals": ["merger", "acquisition", "buyout", "deal", "partnership", "joint venture"],
            "Products & Innovation": ["launch", "product", "innovation", "technology", "patent", "release"],
            "Regulatory & Legal": ["fda", "sec", "regulation", "lawsuit", "investigation", "approval", "compliance"],
            "Market & Trading": ["stock", "share", "price", "trading", "market", "investor", "analyst"],
            "Leadership & Operations": ["ceo", "executive", "layoff", "restructure", "expansion", "hire"]
        }

        clusters = defaultdict(list)

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            best_topic = "Other"
            best_score = 0

            for topic, keywords in topic_keywords.items():
                score = sum(1 for kw in keywords if kw in text)
                if score > best_score:
                    best_score = score
                    best_topic = topic

            clusters[best_topic].append(item)

        # 빈 클러스터 제거 및 정렬
        result = {k: v for k, v in clusters.items() if v}
        return dict(sorted(result.items(), key=lambda x: -len(x[1])))


# ============================================================
# LLM 기반 고급 분석
# ============================================================

class _LLMSentiment(BaseModel):
    overall_sentiment: Literal["bullish", "bearish", "neutral"]
    sentiment_score: float = Field(description="-1.0 (very bearish) to 1.0 (very bullish)")
    key_themes: List[str]
    summary: str = Field(description="1-2 sentence market impact summary")
    confidence: float = Field(description="0.0 to 1.0")


def _call_llm_sentiment(news_items: List[Dict]) -> Dict:
    """Claude를 사용한 뉴스 감성 분석 (구조화 출력)."""
    from mcp_server.tools.llm import call_llm_structured

    news_text = "\n".join(
        f"- {item.get('title', '')}: {item.get('snippet', '')[:200]}" for item in news_items[:10]
    )
    result = call_llm_structured(
        "You are a financial news analyst. Assess how these headlines affect the stock's near-term price.",
        f"News:\n{news_text}",
        _LLMSentiment,
        max_tokens=4096,
    )
    out = result.model_dump()
    out["sentiment_score"] = max(-1.0, min(1.0, out["sentiment_score"]))
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out


def analyze_with_llm(news_items: List[Dict]) -> Dict[str, Any]:
    """
    LLM 기반 고급 감성 분석

    Args:
        news_items: 뉴스 리스트

    Returns:
        LLM 분석 결과
    """
    try:
        return _call_llm_sentiment(news_items)
    except Exception as e:
        logger.warning(f"LLM sentiment analysis failed: {e}")
        return {
            "overall_sentiment": "unknown",
            "sentiment_score": 0.0,
            "key_themes": [],
            "summary": "LLM analysis unavailable",
            "confidence": 0.0,
            "error": str(e)
        }


# ============================================================
# 뉴스 타임라인
# ============================================================

def create_news_timeline(news_items: List[Dict]) -> List[Dict]:
    """
    뉴스를 시간순으로 정리

    Args:
        news_items: 뉴스 리스트

    Returns:
        날짜별 그룹핑된 타임라인
    """
    # 날짜별 그룹핑
    by_date = defaultdict(list)

    for item in news_items:
        published = item.get("published", "")
        if published:
            try:
                # ISO 형식 또는 기타 형식 파싱
                if "T" in published:
                    date_str = published.split("T")[0]
                else:
                    date_str = published[:10]
                by_date[date_str].append(item)
            except Exception:
                by_date["unknown"].append(item)
        else:
            by_date["unknown"].append(item)

    # 날짜순 정렬
    timeline = []
    for date in sorted(by_date.keys(), reverse=True):
        items = by_date[date]
        # 각 날짜 내에서 영향도/감성 점수로 정렬
        sorted_items = sorted(
            items,
            key=lambda x: (
                x.get("impact_score", 0) * abs(x.get("sentiment_score", 0))
            ),
            reverse=True
        )
        timeline.append({
            "date": date,
            "count": len(sorted_items),
            "items": sorted_items
        })

    return timeline


# ============================================================
# 통합 분석 함수
# ============================================================

def analyze_news_sentiment(
    news_items: List[Dict],
    deduplicate: bool = True,
    use_llm: bool = False,
    include_timeline: bool = True
) -> Dict[str, Any]:
    """
    뉴스 종합 감성 분석

    Args:
        news_items: 뉴스 리스트 (search_news 결과에서 hits 추출)
        deduplicate: 중복 제거 여부
        use_llm: LLM 분석 사용 여부
        include_timeline: 타임라인 포함 여부

    Returns:
        종합 분석 결과
    """
    if not news_items:
        return {
            "total": 0,
            "sentiment_distribution": {},
            "overall": "neutral",
            "score": 0.0,
            "items": []
        }

    # 중복 제거
    deduplicator = NewsDeduplicator()
    if deduplicate:
        news_items = deduplicator.deduplicate(news_items)

    # 개별 뉴스 분석
    analyzer = NewsSentimentAnalyzer()
    analyzed_items = []

    for item in news_items:
        analyzed = analyzer.analyze_news_item(item)
        analyzed_items.append(analyzed)

    # 감성 분포 계산
    sentiment_counts = defaultdict(int)
    total_score = 0
    total_impact_weighted_score = 0
    total_weight = 0

    for item in analyzed_items:
        sentiment_counts[item["sentiment"]] += 1
        total_score += item["sentiment_score"]

        # 영향도 가중 점수
        weight = item.get("impact_score", 0.5)
        total_impact_weighted_score += item["sentiment_score"] * weight
        total_weight += weight

    # 전체 감성 계산
    avg_score = total_score / len(analyzed_items) if analyzed_items else 0
    weighted_score = total_impact_weighted_score / total_weight if total_weight > 0 else 0

    if weighted_score > 0.3:
        overall = "bullish"
    elif weighted_score > 0.1:
        overall = "somewhat_bullish"
    elif weighted_score < -0.3:
        overall = "bearish"
    elif weighted_score < -0.1:
        overall = "somewhat_bearish"
    else:
        overall = "neutral"

    result = {
        "total": len(analyzed_items),
        "sentiment_distribution": dict(sentiment_counts),
        "overall": overall,
        "score": round(weighted_score, 3),
        "avg_score": round(avg_score, 3),
        "items": sorted(analyzed_items, key=lambda x: abs(x.get("composite_score", 0)), reverse=True)
    }

    # 클러스터링
    clusters = deduplicator.cluster_by_topic(analyzed_items)
    result["clusters"] = {k: len(v) for k, v in clusters.items()}

    # 타임라인
    if include_timeline:
        result["timeline"] = create_news_timeline(analyzed_items)

    # LLM 분석
    if use_llm:
        llm_result = analyze_with_llm(analyzed_items)
        result["llm_analysis"] = llm_result

    result["analyzed_at"] = datetime.now().isoformat()

    return result


def analyze_ticker_news(
    ticker: str,
    lookback_days: int = 7,
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    특정 종목 뉴스 감성 분석

    Args:
        ticker: 종목 심볼
        lookback_days: 검색 기간 (일)
        use_llm: LLM 분석 사용 여부

    Returns:
        종목별 뉴스 감성 분석 결과
    """
    cache_key = f"news_sentiment_{ticker}_{lookback_days}"
    cached = cache_manager.get(cache_key)
    if cached:
        return cached

    # FR-K12: KR 티커는 Finnhub 을 건너뛰고 곧바로 한국어 RSS (Google News KR)
    # 로 수집한다. Finnhub 은 국내 상장사를 커버하지 않아 매번 빈 결과가 나오며
    # fallback 검색 쿼리는 영어 페이지만 뽑아 KR 센티먼트가 왜곡되기 쉽다.
    from mcp_server.tools.yf_utils import detect_market
    is_kr = detect_market(ticker) == "KR"

    news_items: list = []
    if is_kr:
        try:
            from mcp_server.tools.news_search_kr import search_news_kr
            # 한글 검색이 더 잘 붙도록 ticker + (가능하면) 한글 종목명 병합 쿼리.
            query = ticker
            try:
                from mcp_server.tools.kr_market_data import get_kr_adapter
                nm = get_kr_adapter().get_ticker_name(ticker) or ""
                if nm:
                    query = nm  # 한글명을 우선 쓰면 매칭률이 훨씬 높다
            except Exception:  # noqa: BLE001
                pass
            search_result = search_news_kr([query], lookback_days=lookback_days, max_results=20)
            for block in search_result:
                news_items.extend(block.get("hits", []))
        except Exception:  # noqa: BLE001
            news_items = []
    else:
        # US 경로: Finnhub 우선 → 실패 시 일반 뉴스 검색으로 fallback
        try:
            from mcp_server.tools.finnhub_api import get_company_news
            finnhub_news = get_company_news(ticker)
            news_items = finnhub_news.get("news", [])
        except Exception:
            news_items = []
        if not news_items:
            from mcp_server.tools.news_search import search_news
            search_result = search_news([ticker], lookback_days=lookback_days, max_results=20)
            for block in search_result:
                news_items.extend(block.get("hits", []))

    # 분석 실행
    result = analyze_news_sentiment(
        news_items,
        deduplicate=True,
        use_llm=use_llm,
        include_timeline=True
    )

    result["ticker"] = ticker.upper()
    result["period_days"] = lookback_days

    # 투자 신호 생성
    score = result.get("score", 0)
    if score > 0.4:
        signal = "Strong Positive - News sentiment supports bullish outlook"
    elif score > 0.15:
        signal = "Positive - Generally favorable news coverage"
    elif score < -0.4:
        signal = "Strong Negative - News sentiment indicates caution"
    elif score < -0.15:
        signal = "Negative - Some concerning news coverage"
    else:
        signal = "Neutral - Mixed or neutral news sentiment"

    result["investment_signal"] = signal

    cache_manager.set(cache_key, result, TTL.NEWS)
    return result


def compare_tickers_sentiment(
    tickers: List[str],
    lookback_days: int = 7
) -> Dict[str, Any]:
    """
    여러 종목 뉴스 감성 비교

    Args:
        tickers: 종목 심볼 리스트
        lookback_days: 검색 기간

    Returns:
        종목별 감성 비교 결과
    """
    results = []

    # 병렬 분석
    with ThreadPoolExecutor(max_workers=min(len(tickers), 5)) as executor:
        futures = {
            executor.submit(analyze_ticker_news, ticker, lookback_days, False): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
                results.append({
                    "ticker": ticker.upper(),
                    "overall": data.get("overall", "neutral"),
                    "score": data.get("score", 0),
                    "news_count": data.get("total", 0),
                    "signal": data.get("investment_signal", "")
                })
            except Exception as e:
                results.append({
                    "ticker": ticker.upper(),
                    "error": str(e)
                })

    # 점수순 정렬
    results.sort(key=lambda x: x.get("score", -999), reverse=True)

    # 순위 추가
    for i, r in enumerate(results, 1):
        if "error" not in r:
            r["rank"] = i

    return {
        "comparison_date": datetime.now().isoformat(),
        "period_days": lookback_days,
        "tickers": results,
        "most_positive": results[0]["ticker"] if results and "score" in results[0] else None,
        "most_negative": results[-1]["ticker"] if results and "score" in results[-1] else None
    }


# 싱글톤 분석기
_analyzer = NewsSentimentAnalyzer()


def get_analyzer() -> NewsSentimentAnalyzer:
    return _analyzer
