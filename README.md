## PM-MCP (Portfolio Manager MCP Agent)

**개요**: MCP 서버 기반 펀드 매니저 에이전트. 뉴스·재무·공시 데이터를 수집/요약하고, 후보군 랭킹과 마크다운 리포트를 생성합니다. Claude 호스트앱에서 도구 호출로 전 과정을 대화형으로 제어합니다.
- 타겟: 미국 주식 + 한국 주식(KOSPI/KOSDAQ)
- 호스트앱: Claude (MCP 연동)

### 주요 프로세스
- 신규 투자 진행 프로세스:
  1. 전반적 시장/섹터/기업 동향 파악
  2. 사용자에게 종목 카테고리 추천
  3. 사용자 지정 종목·테마 정밀 파악
  4. 후보 기업/종목 리스트업 및 데이터 수집
  5. 분석·평가·랭킹 및 리포트 작성(예상 이익률·근거)
- 보유 종목 진단/알림 프로세스:
  1. 보유 종목 진단 및 페이즈(상승/유지/불안정/적신호) 알림
  2. 적신호 단계 시 정밀 분석 및 대응 제안

### 아키텍처 요약
- Claude 호스트앱 + MCP 서버 (MCP Python SDK v2 `MCPServer`, stdio 또는 Streamable HTTP)
- 코드 구조:
  - `mcp_server/endpoints/` — MCP 인터페이스 계층 (도구 25개, 리소스 5개, 프롬프트 3개)
  - `mcp_server/tools/`, `mcp_server/pipelines/` — 데이터 수집·분석 로직
  - 도구 목록과 v1→v2 이름 대응: [skills/pm-mcp-director/references/tool-catalog.md](skills/pm-mcp-director/references/tool-catalog.md)
- 데이터 소스:
  - 뉴스: Google News RSS, Finnhub
  - 시세/재무(US): yfinance, SEC EDGAR, Alpha Vantage, Finnhub
  - 시세/재무(KR): PyKrx, KIS Developers, OpenDART, FinanceDataReader
  - LLM 요약·감성 분석: Anthropic Claude (기본 `claude-sonnet-5`)
- 분석/랭킹: 팩터(성장/수익성/밸류/퀄리티/모멘텀/변동성) + 이벤트·감성 스코어, 백테스트
- 스토리지: diskcache(`data/diskcache`), 산출물은 `data/` 하위(CSV, 차트, 주간 리포트)
- 스케줄링: APScheduler (데일리/주간 잡)

### 설치 및 실행
Python 3.12 이상이 필요합니다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 없으면 아래 환경 변수를 직접 작성
```

- stdio(Claude Code / Desktop): [mcp_config.sample.json](mcp_config.sample.json)을 복사해 경로를 환경에 맞게 수정하거나(Claude Code는 프로젝트 루트의 `.mcp.json`, git 추적 제외), 다음 명령으로 등록
  ```bash
  claude mcp add pm-mcp -e PYTHONPATH=$PWD -- $PWD/.venv/bin/python -m mcp_server.mcp_app
  ```
- Streamable HTTP: `scripts/run_mcp_http.sh` → `http://127.0.0.1:8010/mcp`
  - `PM_MCP_TOKEN`을 설정하면 `Authorization: Bearer <token>` 헤더가 필요합니다
  - `HOST`를 loopback 외 주소로 지정하려면 `PM_MCP_TOKEN`과 `ALLOWED_HOSTS`가 필수입니다

주요 환경 변수: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `SEC_EDGAR_USER_AGENT`, `DART_API_KEY`, `KIS_APP_KEY`, `KIS_APP_SECRET`

### 테스트
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
테스트는 네트워크 없이 동작합니다.

### Claude 자연어 예시 프롬프트
- 종목 분석: "삼성전자 분석해줘."
  - 내부 호출: `stock_snapshot(ticker='005930', include_signal=True)`, `stock_factors(ticker='005930', market='KR')`
- 테마 리포트: "AI 테마 리포트 만들어줘. 티커는 AAPL, MSFT, NVDA 사용해."
  - 내부 호출: `report_create(kind='theme', theme='AI', tickers=['AAPL','MSFT','NVDA'])`
- 포트폴리오: "AAPL 10주 150달러, MSFT 5주 400달러 들고 있는데 점검해줘."
  - 내부 호출: `portfolio_analyze(holdings_text='AAPL:10@150, MSFT:5@400')`
- 가격: "삼성전자 최근 6개월 시세 보여줘."
  - 내부 호출: `market_prices(ticker='005930', market='KR', period='6mo')`
- 뉴스: "최근 일주일 AI 칩과 클라우드 성장 관련 뉴스 5개만 요약해줘."
  - 내부 호출: `news_search(queries=['AI chips','cloud growth'], lookback_days=7, max_results=5)`
- 공시: "AAPL의 최근 10-Q/8-K 3건 보여줘."
  - 내부 호출: `filings_recent(ticker='AAPL', forms=['10-Q','8-K'], limit=3)`
