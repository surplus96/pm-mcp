# PM-MCP (Portfolio Manager MCP Server)

미국·한국 주식을 분석하는 포트폴리오 매니저 MCP 서버입니다. 시세·재무·공시·뉴스를 수집해 팩터 점수, 랭킹, 백테스트, 차트, 마크다운 리포트를 만들고, Claude(Claude Code / Desktop)가 도구 호출로 이 과정을 대화형으로 진행합니다.

- 대상 시장: 미국 주식, 한국 주식(KOSPI/KOSDAQ)
- 서버: MCP Python SDK v2 `MCPServer` (v2.0.0), stdio 또는 Streamable HTTP
- 구성: 도구 25개, 리소스 5개, 프롬프트 3개

> 모든 결과는 정보 제공 목적이며 투자 권유가 아닙니다.

## 기능

| 영역 | 도구 | 하는 일 |
|---|---|---|
| 테마 발굴 | `theme_propose`, `theme_explore`, `theme_analyze`, `dip_candidates` | 뉴스 기반 테마 추천 → 대표 종목 → 팩터 점수·백테스트 → 저점 매수 후보 |
| 종목 분석 | `stock_snapshot`, `stock_factors`, `stock_compare`, `stock_rank`, `market_overview`, `backtest_strategy` | 종합 신호(Buy/Hold/Sell), 기술 10·재무 20·감성 10 팩터, 비교·순위, 시장 국면, 팩터 전략 백테스트 |
| 원천 데이터 | `market_prices`, `news_search`, `news_sentiment`, `text_sentiment`, `filings_recent`, `finnhub_data` | OHLCV, 뉴스 원문·감성, SEC 공시, Finnhub |
| 포트폴리오 | `portfolio_analyze`, `portfolio_quick_check`, `portfolio_store`, `watchlist` | 건강도·손익·리밸런싱·배당·알림·상관·섹터, 페이즈 진단(상승/유지/불안정/적신호), 저장, 워치리스트 |
| 출력 | `chart`, `report_create` | Plotly 차트 9종, 마크다운 리포트 5종 |
| 운영 | `data_quality`, `ops_status`, `ops_action` | 데이터 품질 점검, 캐시·서킷 브레이커·스케줄러 상태와 조작 |

- 리소스: `pm://watchlist`, `pm://portfolios`, `pm://portfolios/{name}`, `pm://reference/news-keywords`, `pm://reference/sector-weights`
- 프롬프트: `analyze_stock`, `portfolio_checkup`, `discover_themes`
- 파라미터 전체 목록과 v1 → v2 도구 이름 대응표: [tool-catalog.md](skills/pm-mcp-director/references/tool-catalog.md)

## 구조

```
mcp_server/
  mcp_app.py          stdio 진입점 (python -m mcp_server.mcp_app)
  mcp_app_http.py     Streamable HTTP 진입점 (/mcp, Bearer 토큰)
  config.py           환경 변수·경로·점수/표시 설정
  endpoints/          MCP 인터페이스 계층: 도구·리소스·프롬프트 등록
    discovery.py analysis.py data.py portfolio.py output.py ops.py resources.py
  tools/              데이터 수집·분석 로직 (시세, 팩터, 랭킹, 백테스트, 차트, 캐시, 스케줄러 등)
  pipelines/          여러 도구를 묶은 리포트·후보 선별 파이프라인
  data/               정적 참조 데이터 (DART/KRX/SEC 코드표, 이벤트 가중치)
skills/pm-mcp-director/  Claude가 도구를 조합하는 방법을 담은 스킬
scripts/              실행 스크립트, 도구 카탈로그 생성기
tests/                오프라인 테스트
data/                 런타임 산출물 (watchlist.json 외에는 git 추적 제외)
```

데이터 소스와 필요한 키:

| 용도 | 소스 | 환경 변수 |
|---|---|---|
| 미국 시세·펀더멘털 | yfinance, SEC EDGAR | (없음) / `SEC_EDGAR_USER_AGENT` 권장 |
| 미국 기술 신호 (`stock_snapshot`) | Alpha Vantage | `ALPHA_VANTAGE_API_KEY` |
| 미국 뉴스·내부자·애널리스트 | Finnhub | `FINNHUB_API_KEY` |
| 한국 시세 | PyKrx → KIS Developers → Yahoo(.KS/.KQ) 순 대체 | (없음) / `KIS_APP_KEY`, `KIS_APP_SECRET` |
| 한국 재무 | OpenDART, FinanceDataReader | `DART_API_KEY` |
| 뉴스 | Google News RSS | (없음) |
| LLM 요약·감성 | Anthropic Claude (기본 `claude-sonnet-5`) | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` |

키가 없는 기능은 건너뛰거나 오류로 알리고, 나머지는 그대로 동작합니다. 전체 변수와 선택 설정은 [.env.example](.env.example)에 있습니다.

## 설치

Python 3.12 이상이 필요합니다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 필요한 키만 채움
```

서버는 시작할 때 저장소 루트의 `.env`를 읽습니다.

## Claude에 연결

**Claude Code** (모든 프로젝트에서 사용하려면 `-s user`):
```bash
claude mcp add -s user pm-mcp -e PYTHONPATH=$PWD -- $PWD/.venv/bin/python -m mcp_server.mcp_app
claude mcp get pm-mcp        # Status: ✔ Connected 확인
```
새 대화를 열면 `/mcp`에 도구 25개가 보입니다. 해제는 `claude mcp remove pm-mcp -s user`.

**Claude Desktop 등 JSON 설정**: [mcp_config.sample.json](mcp_config.sample.json)을 복사해 `/path/to/PM-MCP`를 실제 경로로 바꿉니다. 프로젝트 루트의 `.mcp.json`은 경로가 PC마다 달라 git에서 제외됩니다.

**Streamable HTTP**:
```bash
scripts/run_mcp_http.sh      # http://127.0.0.1:8010/mcp
```
- `PM_MCP_TOKEN`을 설정하면 `Authorization: Bearer <token>` 헤더가 필요합니다.
- `HOST`를 loopback 외 주소로 지정하려면 `PM_MCP_TOKEN`과 `ALLOWED_HOSTS`가 필수입니다.

**스킬**: [skills/pm-mcp-director/](skills/pm-mcp-director/)는 요청 의도별로 도구를 어떤 순서로 부를지 안내합니다. claude.ai에 올릴 때는 폴더를 zip으로 묶어 Settings → Capabilities → Skills에 업로드합니다. 도구 이름이 바뀌면 이전에 올린 스킬도 교체해야 합니다.

## 사용 예시

| 자연어 요청 | 내부 호출 |
|---|---|
| "삼성전자 분석해줘" | `stock_snapshot(ticker='005930', include_signal=True)`, `stock_factors(ticker='005930', market='KR')` |
| "AAPL, MSFT, NVDA 순위 매겨줘" | `stock_rank(tickers=['AAPL','MSFT','NVDA'], method='factor')` |
| "AAPL 10주 150달러, MSFT 5주 400달러 들고 있는데 점검해줘" | `portfolio_analyze(holdings_text='AAPL:10@150, MSFT:5@400')` |
| "삼성전자 최근 6개월 시세 보여줘" | `market_prices(ticker='005930', market='KR', period='6mo')` |
| "요즘 뜨는 테마 추천해줘" | `theme_propose()` → `theme_explore(theme=...)` → `theme_analyze(theme=...)` |
| "AI 테마 리포트 만들어줘. 티커는 AAPL, MSFT, NVDA" | `report_create(kind='theme', theme='AI', tickers=['AAPL','MSFT','NVDA'])` |
| "최근 일주일 AI 칩, 클라우드 성장 뉴스 5개 요약해줘" | `news_search(queries=['AI chips','cloud growth'], lookback_days=7, max_results=5)` |
| "AAPL 최근 10-Q/8-K 3건 보여줘" | `filings_recent(ticker='AAPL', forms=['10-Q','8-K'], limit=3)` |
| "NVDA 2023~2024 팩터 전략 백테스트" | `backtest_strategy(ticker='NVDA', start_date='2023-01-01', end_date='2024-12-31')` |

입력 규칙:
- 티커: 미국은 심볼(`AAPL`), 한국은 6자리 코드(`005930`)와 `market="KR"`
- 보유 종목: `'TICKER:SHARES@ENTRY_PRICE, ...'`. `portfolio_quick_check`는 `'AAPL@2024-10-01:185, NVO'`처럼 느슨한 형식도 받습니다.

## 런타임 데이터

| 위치 | 내용 |
|---|---|
| `data/diskcache/` | API 응답 캐시 |
| `data/portfolio/` | `portfolio_store`로 저장한 포트폴리오 |
| `data/charts/` | `chart(save_as=...)`로 저장한 HTML |
| `data/processed/` | `market_prices(view="csv")`, `dip_candidates` CSV (`processed/dip/`) |
| `data/reports/` | 스케줄러 주간 리포트 |
| `data/watchlist.json` | 워치리스트 (git 추적) |
| `/tmp/pm-mcp-images` | 리포트용 PNG 차트 (`IMAGE_OUTPUT_DIR`) |

**캐시**: 데이터 유형별 TTL(가격 4시간, 펀더멘털 24시간, 뉴스 1시간, 공시 6시간 등)로 `data/diskcache/`에 저장합니다. 빈 결과(`None`, 빈 DataFrame·컬렉션)는 일시 장애일 수 있어 캐시하지 않습니다. 전체 삭제는 `ops_action(action="cache_clear")`.

**스케줄러**: 서버 시작 시 자동으로 켜지지 않습니다. `ops_action(action="scheduler_start")`로 시작하며 시간대는 `SCHEDULER_TIMEZONE`(기본 Asia/Seoul)입니다.

| 작업 | 주기 |
|---|---|
| `market_refresh` | 평일 18:30 |
| `news_scan` | 4시간마다 |
| `filings_check` | 평일 09:00 |
| `weekly_report` | 금요일 18:00 |
| `cache_cleanup` | 매일 00:00 |
| `metrics_precompute` | 평일 19:00 |

한 번만 실행하려면 `ops_action(action="scheduler_run_job", target="<작업>")`, 상태는 `ops_status()`로 확인합니다.

## 테스트

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

테스트는 네트워크 없이 동작합니다. 도구·리소스·프롬프트 등록, 오류 변환, 경로 조작 방어, 캐시 정책, HTTP 인증, Claude 래퍼(가짜 클라이언트)를 검증합니다.

실제 데이터로 도구를 하나씩 호출해 보려면 MCP Inspector를 씁니다:
```bash
npx @modelcontextprotocol/inspector .venv/bin/python -m mcp_server.mcp_app
```

## 개발 가이드

**도구 추가·변경**
1. `mcp_server/endpoints/<영역>.py`에 함수를 작성합니다. 무거운 import는 함수 안에서 합니다(서버 시작 속도).
   - docstring 첫 문단이 도구 설명이 됩니다. 파라미터 제약은 `Annotated[..., Field(...)]`, 선택지는 `Literal`로 표현합니다.
   - 실패는 `ToolError`를 던지거나, 기존 `{"error": ...}` 반환은 `check()`로 감쌉니다.
   - 오래 걸리는 작업은 `async` + `ctx.report_progress` + `anyio.to_thread.run_sync`를 씁니다 (`theme_analyze` 참고).
2. 같은 파일의 `TOOLS`에 `(함수, 어노테이션)`을 추가합니다. 어노테이션은 `_common.py`의 `READ_EXTERNAL`, `READ_LOCAL`, `WRITE_LOCAL`, `WRITE_EXTERNAL`, `DESTRUCTIVE_LOCAL` 중에서 고릅니다.
3. `tests/test_server.py`의 `EXPECTED_TOOLS`를 갱신하고 테스트를 추가합니다.
4. 도구 카탈로그를 다시 생성합니다:
   ```bash
   .venv/bin/python scripts/gen_tool_catalog.py
   ```
5. 워크플로가 바뀌면 [SKILL.md](skills/pm-mcp-director/SKILL.md)와 `endpoints/__init__.py`의 `INSTRUCTIONS`를 함께 고칩니다.

**주의**
- stdio 모드에서 stdout은 JSON-RPC 전용입니다. 라이브러리 `print()`는 lifespan에서 stderr로 돌리지만, 새 코드는 `logging`을 쓰세요.
- 사용자 입력으로 파일 경로를 만들 때는 `config.safe_filename()`을 거칩니다.
- 외부 API 호출은 `tools/resilience.py`의 서킷 브레이커와 `tools/cache_manager.py`의 `@cached`를 적용합니다.
