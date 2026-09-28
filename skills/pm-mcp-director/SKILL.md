---
name: pm-mcp-director
description: pm-mcp(포트폴리오 매니저 MCP) 워크플로우 디렉터. 사용자가 주식/포트폴리오 관련 요청을 자연어로 하면 pm-mcp 도구들을 올바른 조합과 순서로 호출해 분석·리포트를 완성한다. 트리거 예시 — "AAPL 분석해줘", "내 포트폴리오 점검", "요즘 뜨는 테마 추천", "이 종목들 비교/랭킹", "뉴스 감성 어때", "백테스트 돌려줘", "차트 그려줘", "리밸런싱 필요한지 봐줘", "저점 매수 후보 찾아줘", stock analysis, portfolio check, theme ideas, ranking, backtest. 종목 티커(AAPL, 005930 등)·포트폴리오·워치리스트·테마·감성·기술적 지표가 언급되면 명시적으로 pm-mcp를 지칭하지 않아도 이 스킬을 사용한다.
---

# pm-mcp 워크플로우 디렉터

pm-mcp는 시장 데이터·기술적/재무/감성 분석·랭킹·백테스트·차트·리포트를 제공하는 포트폴리오 매니저 MCP 서버다(도구 25개, v2). 이 스킬은 사용자의 자연어 요청을 적절한 도구 시퀀스로 변환하는 방법을 안내한다.

전체 도구와 파라미터는 `references/tool-catalog.md`를 참조한다. 아래는 의도별 표준 워크플로우다.

## 핵심 원칙

1. **의도를 먼저 분류한다.** 단일 종목 분석인지, 다종목 비교인지, 포트폴리오 점검인지, 테마 발굴인지에 따라 시작 도구가 다르다.
2. **종합 도구를 우선 사용한다.** `stock_snapshot`, `stock_factors`, `portfolio_analyze(aspect="comprehensive")`가 기본이다. 특정 지표만 물으면 `stock_factors(sections=[...])`처럼 범위를 좁힌다.
3. **시장 컨텍스트를 곁들인다.** 매수/매도 판단이 걸린 요청이면 `market_overview`를 함께 호출해 강세/약세/횡보 맥락을 답변에 반영한다.
4. **오류는 그대로 전달된다.** 도구 실패는 `isError`와 사유 메시지로 돌아온다. 사유를 사용자에게 알리고 대체 경로를 시도한다.
5. **투자 조언 면책.** 결과는 항상 정보 제공 목적임을 밝히고, 최종 판단은 사용자 몫임을 답변 말미에 짧게 명시한다.

## 입력 형식 규칙

- **티커**: 미국 주식은 심볼(AAPL), 한국 주식은 종목코드(005930) + `market="KR"`.
- **tickers**: 문자열 배열 `["AAPL", "MSFT", "NVDA"]`.
- **holdings_text** (`portfolio_analyze`, `portfolio_store`, `chart`): `"TICKER:SHARES@ENTRY_PRICE, ..."` 예: `"AAPL:10@150, MSFT:5@400"`. 현금은 `cash`.
- **자연어 보유주** (`portfolio_quick_check`): `"AAPL@2024-10-01:185, LLY 2024-09-15 520, NVO"`처럼 느슨한 형식 허용.

## 의도별 워크플로우

### 1. 단일 종목 분석 — "AAPL 어때?", "삼성전자 분석해줘"
1. `stock_snapshot(ticker, include_signal=True)` — 가격/변동성 + 밸류에이션·수익성·성장성 + 애널리스트 + 뉴스 감성 + 기술적 신호 + Buy/Hold/Sell 신호. 단일 종목의 기본 선택(US/KR).
2. `stock_factors(ticker, market)` — 기술(10)·재무(20)·감성(10) 팩터와 섹션 점수, 종합 점수·추천 등급. 한국 주식은 `market="KR"`.
3. 시각화 요청 시 `chart(kind="dashboard", ticker)` (캔들+기술지표+수익률분포+상대강도).

기술적 지표만 빠르게: `stock_factors(ticker, sections=["technical"])`.

### 2. 다종목 비교/랭킹 — "이 중에 뭐가 제일 나아?"
- 2~5개 간단 비교: `stock_compare(tickers)`
- 팩터 기반 랭킹: `stock_rank(tickers, method="factor", market)` — 종합점수·추천 등급
- 섹터 가중치·시장 국면 반영: `stock_rank(tickers, method="advanced")`
- 펀더멘털 + 낙폭 보너스: `stock_rank(tickers, method="fundamental")`
- 차트 비교: `chart(kind="comparison", tickers, normalize=True)`

### 3. 테마 발굴 → 종목 선정 — "요즘 뜨는 테마 뭐야?"
1. `theme_propose(lookback_days, max_themes)` — 뉴스 기반 테마 추천
2. 사용자가 테마를 고르면 `theme_explore(theme)` — 대표 종목 후보 + 개요
3. `theme_analyze(theme, market, top_n)` — 팩터 점수로 상위 종목 선정(선택: `include_backtest=True`)
4. 리포트: `report_create(kind="theme", theme, tickers)` 또는 `kind="theme_overview"`(with_images 가능), 선택 종목 요약은 `kind="selection"`

### 4. 저점 매수 후보 — "물린 종목 중 반등할 만한 거", "딥 노리기"
`dip_candidates(theme, tickers?, drawdown_min, top_n)` — 낙폭·이벤트 점수 기반 후보와 요약 마크다운(`report_md`).

### 5. 포트폴리오 점검 — "내 계좌 점검해줘"
- 느슨한 보유 내역: `portfolio_quick_check(holdings_text)` — 페이즈·모멘텀·손익·점수 표
- 정형 입력: `portfolio_analyze(holdings_text, cash)` — 건강도·손익·리밸런싱·배당·알림·상관관계·섹터 통합
- 부분 질문은 `aspect`로: `pnl`, `rebalance`(+`target_weights_text`), `dividends`, `alerts`(+`targets_text`), `correlation`, `sectors`
- 저장/재사용: `portfolio_store(action="save"|"load"|"list", name, holdings_text)` — 저장된 목록은 리소스 `pm://portfolios`로도 읽을 수 있다
- 리포트: `report_create(kind="portfolio_phase" | "portfolio_overview", tickers)`
- 차트: `chart(kind="allocation" | "sectors", holdings_text)`

### 6. 뉴스·감성 — "요즘 NVDA 뉴스 분위기 어때?"
1. `news_sentiment(tickers, lookback_days)` — bullish/bearish 점수·분포·투자신호 (`view="detail"`)
2. 여러 종목 순위: `news_sentiment(tickers, view="compare")` / 날짜별 흐름: `view="timeline"`
3. 원문 뉴스: `news_search(queries, lookback_days)` / 임의 텍스트: `text_sentiment(text)`
4. Finnhub 심화(미국): `finnhub_data(kind="summary", symbol)` — 뉴스+내부자+애널리스트+재무 통합 신호

### 7. 백테스트 — "이 전략 과거에 먹혔어?"
`backtest_strategy(ticker, market, start_date, end_date, buy_threshold, sell_threshold)` — CAGR·MDD·Sharpe·승률·SPY 대비 성과. 결과 해석 시 과최적화 위험을 함께 언급한다.

### 8. 차트 — "차트 그려줘"
| 요청 | 호출 |
|---|---|
| 캔들차트 | `chart(kind="candlestick", ticker, period, ma_periods=[20,50])` |
| 기술지표 | `chart(kind="technical", ticker, indicators=["rsi","macd"])` |
| 종목 비교 | `chart(kind="comparison", tickers)` |
| 상대강도 | `chart(kind="relative_strength", ticker, benchmark="SPY")` |
| 수익률 분포 | `chart(kind="returns", ticker)` |
| 상관관계 | `chart(kind="correlation", tickers)` |
| 포트폴리오 비중 / 섹터 | `chart(kind="allocation" \| "sectors", holdings_text)` |
| 종합 대시보드 | `chart(kind="dashboard", ticker)` |

차트는 `chart_html`(대시보드는 `charts`)을 반환한다. `save_as`를 지정하면 서버의 `data/charts/<save_as>.html`에도 저장되고 경로가 `saved_path`(대시보드는 `saved_paths`)로 돌아온다. 사용자에게 보여줄 때는 이 경로를 알려주거나 반환된 HTML을 파일로 저장해 전달한다.

### 9. 워치리스트·데이터·운영
- 워치리스트: `watchlist(action="get" | "update", tickers, themes)` (리소스 `pm://watchlist`)
- 시세 원자료: `market_prices(ticker, market, period, view="rows" | "summary" | "csv")`
- SEC 공시: `filings_recent(ticker, forms, limit)`
- 데이터 품질 의심: `data_quality(tickers, check_type="validate" | "clean" | "outliers" | "missing")`
- 도구 오류가 반복되면 `ops_status()`로 서킷 브레이커·캐시·스케줄러 상태를 보고, 필요시 `ops_action(action="circuit_reset")`

## 알려진 제약 (답변 시 유의)

- `stock_snapshot`의 미국 기술 지표는 Alpha Vantage를 쓰며 무료 한도 때문에 호출 사이 대기가 있어 30초 이상 걸릴 수 있다. 빠른 확인은 `stock_factors(sections=["technical"])`를 쓴다(로컬 계산).
- `finnhub_data`는 FINNHUB_API_KEY가 필요하다.
- 한국 시세는 PyKrx → KIS Developers → Yahoo(.KS/.KQ) 순으로 대체된다. KIS 단계는 `KIS_APP_KEY`/`KIS_APP_SECRET`이 있을 때만 동작한다.
- LLM이 필요한 경로(`news_sentiment(use_llm=True)`, `report_create(kind="theme")`의 뉴스 요약)는 `ANTHROPIC_API_KEY`가 필요하다. 키가 없으면 `use_llm=False`(키워드 기반)로 답한다.
- `filings_recent`는 미국 종목 전용이다(한국 공시 미지원). SEC는 연락처가 담긴 User-Agent를 요구하므로 `SEC_EDGAR_USER_AGENT`가 비어 있으면 임시 값으로 요청하다 차단될 수 있다.
- 빈 결과는 캐시하지 않는다. 일시적인 "가격 데이터가 없습니다" 오류는 같은 요청을 한 번 더 시도하면 풀릴 수 있다.

## 응답 형식

분석 결과는 다음 순서로 정리한다: **핵심 결론 → 근거 지표 (점수·신호) → 리스크/유의점 → (요청 시) 차트·리포트 파일**. 수치는 도구가 반환한 값을 그대로 인용하고 임의로 만들어내지 않는다.
