# pm-mcp 전체 도구 카탈로그 (v2, 도구 25개)

모든 도구는 `mcp__pm-mcp__<이름>` 형태로 호출한다. **굵게**는 필수, `이름(값)`은 기본값, `이름?`은 선택값, `∈`는 허용 값이다.
아래 표는 `scripts/gen_tool_catalog.py`가 서버 스키마에서 생성한다. 도구를 바꾸면 스크립트를 다시 실행한다.

## 테마/아이디어 발굴
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `theme_propose` | lookback_days(7), max_themes(5) | 최근 뉴스 밀도를 기준으로 투자 테마를 추천합니다. |
| `theme_explore` | **theme**, lookback_days(7) | 테마의 대표 종목 후보와 뉴스/가격 개요(마크다운)를 반환합니다. 종목 선정 전 탐색 단계에 사용합니다. |
| `theme_analyze` | **theme**, market('US') ∈ US/KR, top_n(5), include_sentiment(True), include_backtest(False), rerank_by_backtest(False), backtest_start('2024-01-01'), backtest_end('2024-12-31'), factor_weights? | 테마 → 종목 발굴 → 팩터 점수 → (선택) 백테스트까지 한 번에 수행하는 테마 종합 분석. |
| `dip_candidates` | **theme**, tickers?, top_n(5), drawdown_min(0.2), ret10_min(0.0), event_min(0.5), save_csv(True) | 낙폭이 크지만 펀더멘털·이벤트 점수가 양호한 저점 매수 후보를 선별합니다. |

## 종목 분석·랭킹·백테스트
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `stock_snapshot` | **ticker**, include_signal(False) | 종목 스냅샷: 가격·변동성, 밸류에이션·수익성·성장성, 애널리스트 컨센서스, 뉴스 감성, 기술적 신호와 종합 신호(composite_signal). |
| `stock_factors` | **ticker**, market('US') ∈ US/KR, sections? ∈ technical/financial/sentiment, period('1y') ∈ 1mo/3mo/6mo/1y/2y/5y, sentiment_days(7) | 팩터 분석: 기술적(10) · 재무(20) · 감성(10) 지표 값, 해석, 섹션별 점수와 종합 점수/추천 등급. |
| `stock_compare` | **tickers** | 2~5개 종목을 스냅샷 지표(가격·밸류에이션·감성·종합 신호)로 나란히 비교하고 종합 점수 순으로 정렬합니다. |
| `stock_rank` | **tickers**, method('factor') ∈ factor/advanced/fundamental, market('US') ∈ US/KR, include_technical(True), include_financial(True), include_sentiment(True), use_sector_weights(True), use_market_adjustment(True), sector_neutral(False), dip_weight(0.12), use_dip_bonus(True) | 여러 종목을 점수화해 순위를 매깁니다. |
| `market_overview` | sector? | 시장 국면(강세/약세/횡보)과 랭킹 엔진이 쓰는 섹터별 팩터 가중치를 반환합니다. 매수/매도 판단 답변의 맥락으로 사용합니다. |
| `backtest_strategy` | **ticker**, market('US') ∈ US/KR, start_date('2023-01-01'), end_date('2024-12-31'), rebalance_period(30), buy_threshold(60.0), sell_threshold(40.0), initial_capital(10000.0) | 팩터 점수 기반 매매 전략 백테스트. CAGR·MDD·Sharpe·승률·거래 내역과 SPY 대비 성과를 반환합니다. |

## 시장 데이터·뉴스·공시
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `market_prices` | **ticker**, market? ∈ US/KR, start?, end?, period? ∈ 1mo/3mo/6mo/1y/2y/5y, interval('1d') ∈ 1d/1wk/1mo, view('rows') ∈ rows/summary/csv, cursor(0), page_size(100) | OHLCV 시세 조회 (US/KR). 기본 구간은 최근 1년이며 start 또는 period로 지정합니다. |
| `news_search` | **queries**, lookback_days(7), max_results(10) | Google News RSS에서 검색어별 최근 기사(제목·출처·URL·요약)를 가져옵니다. |
| `news_sentiment` | **tickers**, lookback_days(7), view('detail') ∈ detail/compare/timeline, use_llm(False) | 종목 뉴스 감성 분석 (bullish/bearish 점수, 분포, 투자 신호). |
| `text_sentiment` | **text** | 임의의 텍스트(뉴스 헤드라인·본문)의 감성, 점수, 영향도와 매칭된 키워드를 분석합니다. |
| `filings_recent` | **ticker**, forms?, limit(10) | SEC EDGAR 최근 공시 목록 (기본 8-K/10-Q/10-K). 미국 종목 전용입니다. |
| `finnhub_data` | **kind** ∈ summary/news/insider/analyst/earnings/financials, symbol?, from_date?, to_date? | Finnhub 데이터 (FINNHUB_API_KEY 필요, 미국 종목). |

## 포트폴리오·워치리스트
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `portfolio_analyze` | **holdings_text**, aspect('comprehensive') ∈ comprehensive/pnl/rebalance/dividends/alerts/correlation/sectors, cash(0), target_weights_text?, threshold_pct(5.0), targets_text?, days_ahead(90), period('1y') ∈ 1mo/3mo/6mo/1y/2y/5y | 포트폴리오 분석. |
| `portfolio_quick_check` | **holdings_text** | 느슨하게 적힌 보유 종목(매수일·매수가 선택)을 받아 페이즈(상승/유지/불안정/적신호), 모멘텀·변동성·낙폭·SPY 상관, 손익, 펀더멘털 점수를 행 데이터와 마크다운 표로 반환합니다. 매수가가 없으면 매수일 종가를 사용합니다. |
| `portfolio_store` | **action** ∈ save/load/list, name?, holdings_text?, cash(0) | 포트폴리오를 이름으로 저장(save, 같은 이름은 덮어씀)/불러오기(load)/목록(list) 합니다. |
| `watchlist` | action('get') ∈ get/update, tickers?, themes? | 스케줄러가 추적하는 워치리스트 조회(get) 또는 교체(update: 전달한 목록으로 덮어씀). |

## 차트·리포트
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `chart` | **kind** ∈ candlestick/technical/comparison/relative_strength/returns/allocation/correlation/sectors/dashboard, ticker?, tickers?, holdings_text?, period('6mo') ∈ 1mo/3mo/6mo/1y/2y/5y, benchmark('SPY'), indicators? ∈ rsi/macd/bbands/volume, ma_periods?, show_volume(True), normalize(True), save_as? | Plotly 차트를 생성해 HTML 조각(chart_html, CDN 스크립트 사용)으로 반환합니다. |
| `report_create` | **kind** ∈ theme/theme_overview/selection/portfolio_phase/portfolio_overview, **tickers**, theme?, with_images(False), days? | 마크다운 리포트를 생성해 본문을 반환합니다. |

## 데이터 품질·운영
| 도구 | 파라미터 | 설명 |
|---|---|---|
| `data_quality` | **tickers**, check_type('validate') ∈ validate/clean/outliers/missing, period('1y') ∈ 1mo/3mo/6mo/1y/2y/5y, threshold(3.0) | 가격 데이터 품질 점검. 분석 결과가 이상할 때 먼저 확인합니다. |
| `ops_status` | history_limit(10) | 서버 운영 상태: 캐시 통계, 외부 API 서킷 브레이커 상태, 스케줄러 작업과 최근 실행 이력. |
| `ops_action` | **action** ∈ cache_clear/cache_expire/circuit_reset/scheduler_start/scheduler_stop/scheduler_run_job, target? | 운영 작업 실행. cache_clear는 모든 캐시를 지우므로 이후 요청이 느려집니다. |

## 리소스
| URI | 내용 |
|---|---|
| `pm://watchlist` | 워치리스트 (tickers, themes) |
| `pm://portfolios` | 저장된 포트폴리오 이름 목록 |
| `pm://portfolios/{name}` | 저장된 포트폴리오 상세 |
| `pm://reference/news-keywords` | 뉴스 감성·영향도 키워드 사전 |
| `pm://reference/sector-weights` | 고급 랭킹 섹터별 팩터 가중치 |

## 프롬프트
- `analyze_stock(ticker, market)` · `portfolio_checkup(holdings_text, cash)` · `discover_themes(lookback_days)`

## v1 → v2 도구 대응표
| v1 (제거됨) | v2 |
|---|---|
| `market_get_prices`, `market_get_prices_paginated`, `market_get_prices_summary`, `market_write_prices_csv` | `market_prices(view=rows/summary/csv)` |
| `stock_comprehensive_analysis`, `stock_investment_signal` | `stock_snapshot(include_signal)` |
| `comprehensive_analyze`, `technical_analyze`, `financial_analyze`, `sentiment_analyze` | `stock_factors(sections)` |
| `technical_rsi/macd/bbands/sma/ema/adx/summary`, `technical_compare` | `stock_factors(sections=["technical"])` (로컬 계산) |
| `rank_stocks`, `ranking_advanced`, `analytics_rank` | `stock_rank(method=factor/advanced/fundamental)` |
| `market_condition`, `sector_weights_info` | `market_overview(sector)` |
| `propose_themes_tool`, `explore_theme_tool`, `propose_tickers_tool` | `theme_propose`, `theme_explore` |
| `theme_analyze_with_factors` | `theme_analyze` |
| `analyze_dip_candidates_tool` | `dip_candidates` |
| `analyze_selection_tool`, `create_theme_report`, `create_portfolio_phase_report`, `present_theme`, `present_portfolio`, `reports_generate` | `report_create(kind)` |
| `news_sentiment_analyze`, `news_sentiment_compare`, `news_timeline` | `news_sentiment(view)` |
| `news_sentiment_text` | `text_sentiment` |
| `news_impact_keywords` | 리소스 `pm://reference/news-keywords` |
| `news_deduplicate` | 제거 |
| `filings_fetch_recent` | `filings_recent` |
| `finnhub_news/insider/analyst/earnings/financials/summary` | `finnhub_data(kind)` |
| `portfolio_comprehensive`, `portfolio_pnl`, `portfolio_rebalance`, `portfolio_dividends`, `portfolio_alerts`, `portfolio_correlation`, `portfolio_sectors` | `portfolio_analyze(aspect)` |
| `portfolio_analyze_nl_tool`, `portfolio_evaluate`, `portfolio_evaluate_detailed` | `portfolio_quick_check` |
| `portfolio_save`, `portfolio_load`, `portfolio_list` | `portfolio_store(action)` |
| `watchlist_get`, `watchlist_update` | `watchlist(action)` |
| `chart_*` 9종 | `chart(kind)` |
| `data_validate`, `data_validate_and_clean`, `data_quality_summary`, `data_clean`, `data_check_outliers`, `data_check_missing` | `data_quality(check_type)` |
| `cache_stats`, `circuit_status`, `scheduler_status`, `scheduler_history` | `ops_status` |
| `cache_clear`, `cache_expire`, `circuit_reset`, `scheduler_start/stop/run_job` | `ops_action(action)` |
| `obsidian_write`, `present_*_save`, `news_search_log_tool`, `help_commands` | 제거 (리포트는 마크다운 반환, 사용법은 서버 instructions와 프롬프트) |
