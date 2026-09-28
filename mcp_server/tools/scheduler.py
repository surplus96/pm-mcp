"""
스케줄링 모듈 - 자동화된 작업 실행

Features:
- APScheduler 기반 작업 스케줄링
- Cron 및 Interval 트리거 지원
- 작업 상태 모니터링
- 수동 트리거 지원

스케줄:
- 시장 데이터 갱신: 평일 18:30 (장 마감 후)
- 뉴스 스캔: 4시간마다
- 공시 체크: 평일 09:00
- 포트폴리오 리포트: 금요일 18:00
- 캐시 정리: 매일 00:00
- 메트릭 사전 계산: 평일 19:00
"""
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, JobExecutionEvent

logger = logging.getLogger(__name__)


# ===== 작업 실행 결과 저장 =====
class JobHistory:
    """작업 실행 이력 관리"""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.history: List[Dict] = []

    def add(self, job_id: str, status: str, result: Any = None, error: str = None):
        entry = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        # 최대 개수 유지
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]

    def get_recent(self, limit: int = 10) -> List[Dict]:
        return self.history[-limit:]

    def get_by_job(self, job_id: str, limit: int = 5) -> List[Dict]:
        return [h for h in self.history if h["job_id"] == job_id][-limit:]


_job_history = JobHistory()


# ===== 스케줄러 클래스 =====
class PMScheduler:
    """PM-MCP 스케줄러

    사용 예시:
        scheduler = PMScheduler()
        scheduler.start()

        # 수동 실행
        await scheduler.run_job("market_refresh")

        # 상태 조회
        status = scheduler.get_status()
    """

    _instance: Optional["PMScheduler"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 스케줄러 설정
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': ThreadPoolExecutor(10),
            'processpool': ProcessPoolExecutor(3)
        }
        job_defaults = {
            'coalesce': True,  # 밀린 작업 합치기
            'max_instances': 1,  # 동시 실행 방지
            'misfire_grace_time': 60 * 5  # 5분 유예
        }

        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='Asia/Seoul'
        )

        # 이벤트 리스너 등록
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

        # 작업 등록
        self._setup_jobs()
        self._initialized = True

    def _job_listener(self, event: JobExecutionEvent):
        """작업 실행 이벤트 처리"""
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
            _job_history.add(event.job_id, "error", error=str(event.exception))
        else:
            logger.info(f"Job {event.job_id} completed")
            _job_history.add(event.job_id, "success", result=event.retval)

    def _setup_jobs(self):
        """스케줄 작업 등록"""

        # 1. 시장 데이터 갱신 (평일 18:30)
        self.scheduler.add_job(
            job_market_refresh,
            CronTrigger(day_of_week='mon-fri', hour=18, minute=30),
            id='market_refresh',
            name='시장 데이터 갱신',
            replace_existing=True
        )

        # 2. 뉴스 스캔 (4시간마다)
        self.scheduler.add_job(
            job_news_scan,
            IntervalTrigger(hours=4),
            id='news_scan',
            name='뉴스 스캔',
            replace_existing=True
        )

        # 3. SEC 공시 체크 (평일 09:00)
        self.scheduler.add_job(
            job_filings_check,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=0),
            id='filings_check',
            name='SEC 공시 체크',
            replace_existing=True
        )

        # 4. 주간 포트폴리오 리포트 (금요일 18:00)
        self.scheduler.add_job(
            job_weekly_report,
            CronTrigger(day_of_week='fri', hour=18, minute=0),
            id='weekly_report',
            name='주간 포트폴리오 리포트',
            replace_existing=True
        )

        # 5. 캐시 정리 (매일 00:00)
        self.scheduler.add_job(
            job_cache_cleanup,
            CronTrigger(hour=0, minute=0),
            id='cache_cleanup',
            name='캐시 정리',
            replace_existing=True
        )

        # 6. 메트릭 사전 계산 (평일 19:00)
        self.scheduler.add_job(
            job_metrics_precompute,
            CronTrigger(day_of_week='mon-fri', hour=19, minute=0),
            id='metrics_precompute',
            name='메트릭 사전 계산',
            replace_existing=True
        )

    def start(self):
        """스케줄러 시작"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("PMScheduler started")

    def stop(self):
        """스케줄러 중지"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("PMScheduler stopped")

    def pause(self):
        """스케줄러 일시 중지"""
        self.scheduler.pause()
        logger.info("PMScheduler paused")

    def resume(self):
        """스케줄러 재개"""
        self.scheduler.resume()
        logger.info("PMScheduler resumed")

    def get_jobs(self) -> List[Dict]:
        """등록된 작업 목록 조회"""
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = getattr(job, 'next_run_time', None)
            jobs.append({
                "id": job.id,
                "name": getattr(job, 'name', job.id),
                "next_run": next_run.isoformat() if next_run else None,
                "trigger": str(job.trigger)
            })
        return jobs

    def get_status(self) -> Dict:
        """스케줄러 상태 조회"""
        return {
            "running": self.scheduler.running,
            "jobs_count": len(self.scheduler.get_jobs()),
            "jobs": self.get_jobs(),
            "recent_history": _job_history.get_recent(10)
        }

    def run_job_now(self, job_id: str) -> Dict:
        """작업 즉시 실행"""
        job = self.scheduler.get_job(job_id)
        if not job:
            return {"error": f"Job '{job_id}' not found"}

        try:
            # 즉시 실행
            result = job.func()
            _job_history.add(job_id, "manual_success", result=result)
            return {"status": "success", "job_id": job_id, "result": result}
        except Exception as e:
            _job_history.add(job_id, "manual_error", error=str(e))
            return {"status": "error", "job_id": job_id, "error": str(e)}

    def add_custom_job(
        self,
        func: Callable,
        job_id: str,
        name: str,
        trigger_type: str = "interval",
        **trigger_kwargs
    ) -> Dict:
        """사용자 정의 작업 추가

        Args:
            func: 실행할 함수
            job_id: 작업 ID
            name: 작업 이름
            trigger_type: "interval" 또는 "cron"
            **trigger_kwargs: 트리거 설정 (hours, minutes, day_of_week, hour, minute 등)
        """
        if trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_kwargs)
        elif trigger_type == "cron":
            trigger = CronTrigger(**trigger_kwargs)
        else:
            return {"error": f"Unknown trigger type: {trigger_type}"}

        self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            name=name,
            replace_existing=True
        )
        return {"status": "added", "job_id": job_id, "name": name}

    def remove_job(self, job_id: str) -> Dict:
        """작업 제거"""
        try:
            self.scheduler.remove_job(job_id)
            return {"status": "removed", "job_id": job_id}
        except Exception as e:
            return {"error": str(e)}

    def get_job_history(self, job_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """작업 실행 이력 조회"""
        if job_id:
            return _job_history.get_by_job(job_id, limit)
        return _job_history.get_recent(limit)


# ===== 스케줄 작업 함수들 =====

def job_market_refresh() -> Dict:
    """시장 데이터 갱신 작업"""
    logger.info("Running job: market_refresh")
    try:
        from mcp_server.tools.cache_manager import cache_manager
        from mcp_server.tools.market_data import get_fundamentals_snapshot, get_momentum_metrics

        # 관심 종목 목록 (설정 파일에서 로드하거나 기본값 사용)
        watchlist = _get_watchlist()

        results = {"updated": [], "failed": []}
        for ticker in watchlist:
            try:
                # 캐시 무효화 후 새 데이터 로드
                cache_manager.delete(f"fundamentals:{ticker}")
                cache_manager.delete(f"momentum:{ticker}")

                get_fundamentals_snapshot(ticker)
                get_momentum_metrics(ticker)
                results["updated"].append(ticker)
            except Exception as e:
                logger.warning(f"Failed to refresh {ticker}: {e}")
                results["failed"].append(ticker)

        logger.info(f"Market refresh completed: {len(results['updated'])} updated, {len(results['failed'])} failed")
        return results
    except Exception as e:
        logger.error(f"Market refresh failed: {e}")
        raise


def job_news_scan() -> Dict:
    """뉴스 스캔 작업"""
    logger.info("Running job: news_scan")
    try:
        from mcp_server.tools.news_search import search_news

        # 관심 테마 목록
        themes = _get_watch_themes()

        results = {"themes": {}}
        for theme in themes:
            try:
                news = search_news([f"{theme} stocks"], lookback_days=1, max_results=5)
                hit_count = sum(len(n.get("hits", [])) for n in news)
                results["themes"][theme] = hit_count
            except Exception as e:
                logger.warning(f"News scan failed for {theme}: {e}")
                results["themes"][theme] = -1

        logger.info(f"News scan completed: {len(themes)} themes scanned")
        return results
    except Exception as e:
        logger.error(f"News scan failed: {e}")
        raise


def job_filings_check() -> Dict:
    """SEC 공시 체크 작업"""
    logger.info("Running job: filings_check")
    try:
        from mcp_server.tools.filings import fetch_recent_filings

        watchlist = _get_watchlist()

        results = {"filings": {}}
        for ticker in watchlist:
            try:
                filings = fetch_recent_filings(ticker, limit=3, use_cache=False)
                # 최근 1일 이내 공시만 필터링
                recent = [f for f in filings if _is_recent_filing(f, days=1)]
                if recent:
                    results["filings"][ticker] = len(recent)
            except Exception as e:
                logger.warning(f"Filings check failed for {ticker}: {e}")

        logger.info(f"Filings check completed: {len(results['filings'])} tickers with new filings")
        return results
    except Exception as e:
        logger.error(f"Filings check failed: {e}")
        raise


def job_weekly_report() -> Dict:
    """주간 포트폴리오 리포트 생성"""
    logger.info("Running job: weekly_report")
    try:
        from mcp_server.tools.portfolio import evaluate_holdings
        from mcp_server.config import REPORTS_PATH

        watchlist = _get_watchlist()
        if not watchlist:
            return {"status": "skipped", "reason": "no watchlist"}

        # 포트폴리오 평가
        evaluation = evaluate_holdings(watchlist)

        # 마크다운 리포트 생성
        date_str = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"# Weekly Portfolio Report",
            f"",
            f"Date: {date_str}",
            f"",
            f"## Holdings Summary",
            f"",
            f"| Ticker | Phase | Score |",
            f"|--------|-------|-------|"
        ]

        for e in evaluation:
            lines.append(f"| {e.get('ticker')} | {e.get('phase', 'N/A')} | {e.get('score', 'N/A')} |")

        body = "\n".join(lines)

        report_path = os.path.join(REPORTS_PATH, f"weekly_{date_str}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(body)

        logger.info(f"Weekly report saved: {report_path}")
        return {"status": "success", "report_path": report_path}
    except Exception as e:
        logger.error(f"Weekly report failed: {e}")
        raise


def job_cache_cleanup() -> Dict:
    """만료된 캐시 정리"""
    logger.info("Running job: cache_cleanup")
    try:
        from mcp_server.tools.cache_manager import cache_manager

        expired = cache_manager.expire()
        stats = cache_manager.stats()

        logger.info(f"Cache cleanup completed: {expired} items expired")
        return {"expired": expired, "stats": stats}
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise


def job_metrics_precompute() -> Dict:
    """메트릭 사전 계산"""
    logger.info("Running job: metrics_precompute")
    try:
        from mcp_server.tools.collect import compute_basic_metrics

        watchlist = _get_watchlist()

        results = {"computed": [], "failed": []}
        for ticker in watchlist:
            try:
                compute_basic_metrics(ticker)
                results["computed"].append(ticker)
            except Exception as e:
                logger.warning(f"Metrics precompute failed for {ticker}: {e}")
                results["failed"].append(ticker)

        logger.info(f"Metrics precompute completed: {len(results['computed'])} computed")
        return results
    except Exception as e:
        logger.error(f"Metrics precompute failed: {e}")
        raise


# ===== 헬퍼 함수 =====

def _get_watchlist() -> List[str]:
    """관심 종목 목록 로드"""
    try:
        from mcp_server.config import WATCHLIST_PATH
        import json
        with open(WATCHLIST_PATH, 'r') as f:
            data = json.load(f)
            return data.get("tickers", [])
    except Exception:
        # 기본 워치리스트
        return ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"]


def _get_watch_themes() -> List[str]:
    """관심 테마 목록 로드"""
    try:
        from mcp_server.config import WATCHLIST_PATH
        import json
        with open(WATCHLIST_PATH, 'r') as f:
            data = json.load(f)
            return data.get("themes", [])
    except Exception:
        # 기본 테마
        return ["AI", "semiconductor", "renewable energy", "biotech"]


def _is_recent_filing(filing: Dict, days: int = 1) -> bool:
    """최근 공시 여부 확인"""
    try:
        from datetime import datetime, timedelta
        filing_date = filing.get("filingDate")
        if not filing_date:
            return False
        fd = datetime.strptime(filing_date, "%Y-%m-%d")
        return fd >= datetime.now() - timedelta(days=days)
    except Exception:
        return False


# ===== 글로벌 스케줄러 인스턴스 =====
def get_scheduler() -> PMScheduler:
    """스케줄러 싱글톤 인스턴스 반환"""
    return PMScheduler()
