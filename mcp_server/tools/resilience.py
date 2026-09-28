"""
복원력 모듈 - 에러 핸들링, 재시도, 서킷 브레이커

Features:
- tenacity 기반 재시도 로직 (지수 백오프)
- 서킷 브레이커 패턴
- API별 타임아웃 설정
- 구조화된 로깅
- 폴백 체인 지원
"""
from __future__ import annotations
from typing import Callable, TypeVar, Optional, Any, List
from functools import wraps
import time
import logging
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ===== 타임아웃 설정 =====
class Timeout:
    """API별 타임아웃 상수 (초)

    LLM: Claude API 요청 타임아웃. 환경변수 LLM_TIMEOUT_SEC 로 오버라이드 가능.
    """
    import os as _os
    YFINANCE = 30
    LLM = float(_os.getenv("LLM_TIMEOUT_SEC", "300"))
    SEC_EDGAR = 20
    RSS = 10
    DEFAULT = 15


# ===== 재시도 설정 =====
class RetryConfig:
    """API별 재시도 설정"""
    YFINANCE = {"attempts": 2, "min_wait": 2, "max_wait": 10}
    SEC_EDGAR = {"attempts": 2, "min_wait": 2, "max_wait": 10}
    RSS = {"attempts": 1, "min_wait": 1, "max_wait": 3}
    DEFAULT = {"attempts": 2, "min_wait": 1, "max_wait": 5}


# ===== 서킷 브레이커 =====
class CircuitOpenError(Exception):
    """서킷이 열려있을 때 발생하는 예외"""
    pass


class CircuitBreaker:
    """서킷 브레이커 패턴 구현

    연속 실패가 threshold에 도달하면 서킷을 열고,
    reset_timeout 후 half-open 상태로 전환하여 재시도합니다.

    사용 예시:
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        @breaker
        def risky_api_call():
            return requests.get(url)
    """

    _instances: dict[str, "CircuitBreaker"] = {}

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_max_calls: int = 1
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> "CircuitBreaker":
        """이름으로 서킷 브레이커 인스턴스 가져오기 (싱글톤)"""
        if name not in cls._instances:
            cls._instances[name] = cls(name=name, **kwargs)
        return cls._instances[name]

    def _check_state(self) -> None:
        """상태 전환 체크"""
        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time > self.reset_timeout):
                self.state = "half-open"
                self.half_open_calls = 0
                logger.info(f"CircuitBreaker[{self.name}]: open -> half-open")

    def _on_success(self) -> None:
        """성공 시 처리"""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"CircuitBreaker[{self.name}]: half-open -> closed")
        else:
            self.failure_count = 0

    def _on_failure(self) -> None:
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half-open":
            self.state = "open"
            logger.warning(f"CircuitBreaker[{self.name}]: half-open -> open (failure in half-open)")
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"CircuitBreaker[{self.name}]: closed -> open (threshold reached: {self.failure_count})")

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """서킷 브레이커를 통해 함수 호출"""
        self._check_state()

        if self.state == "open":
            raise CircuitOpenError(f"Circuit[{self.name}] is open. Try again after {self.reset_timeout}s")

        if self.state == "half-open":
            self.half_open_calls += 1
            if self.half_open_calls > self.half_open_max_calls:
                raise CircuitOpenError(f"Circuit[{self.name}] is half-open. Max calls exceeded.")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """데코레이터로 사용"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper

    def reset(self) -> None:
        """서킷 브레이커 리셋"""
        self.failure_count = 0
        self.success_count = 0
        self.state = "closed"
        self.last_failure_time = None
        logger.info(f"CircuitBreaker[{self.name}]: manually reset to closed")

    def get_status(self) -> dict:
        """현재 상태 조회"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "reset_timeout": self.reset_timeout
        }


# ===== 재시도 데코레이터 =====
def retry_with_backoff(
    attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
    exceptions: tuple = (requests.Timeout, requests.ConnectionError, requests.HTTPError),
    on_retry: Optional[Callable] = None
):
    """지수 백오프를 사용한 재시도 데코레이터

    Args:
        attempts: 최대 시도 횟수
        min_wait: 최소 대기 시간 (초)
        max_wait: 최대 대기 시간 (초)
        exceptions: 재시도할 예외 타입들
        on_retry: 재시도 전 호출할 콜백

    사용 예시:
        @retry_with_backoff(attempts=3, min_wait=1, max_wait=10)
        def fetch_data(url):
            return requests.get(url, timeout=10)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_api(api_name: str = "default"):
    """API별 사전 정의된 재시도 설정 적용

    Args:
        api_name: API 이름 (YFINANCE, SEC_EDGAR, RSS, DEFAULT)

    사용 예시:
        @retry_api("SEC_EDGAR")
        def fetch_filings(ticker):
            ...
    """
    config = getattr(RetryConfig, api_name.upper(), RetryConfig.DEFAULT)
    return retry_with_backoff(
        attempts=config["attempts"],
        min_wait=config["min_wait"],
        max_wait=config["max_wait"]
    )


# ===== 폴백 체인 =====
class FallbackChain:
    """다단계 폴백 체인

    여러 데이터 소스를 순차적으로 시도하고,
    모두 실패하면 캐시나 기본값을 반환합니다.

    사용 예시:
        chain = FallbackChain("news")
        chain.add(fetch_from_google_news, name="google_news")
        chain.add(fetch_from_finnhub, name="finnhub")
        chain.add(fetch_from_rss, name="rss")
        chain.set_cache_fallback(get_cached_news)
        chain.set_default([])

        result = chain.execute(query="AI stocks")
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.providers: List[tuple[str, Callable]] = []
        self.cache_fallback: Optional[Callable] = None
        self.default_value: Any = None

    def add(self, func: Callable, name: Optional[str] = None) -> "FallbackChain":
        """폴백 체인에 제공자 추가"""
        provider_name = name or func.__name__
        self.providers.append((provider_name, func))
        return self

    def set_cache_fallback(self, func: Callable) -> "FallbackChain":
        """캐시 폴백 설정"""
        self.cache_fallback = func
        return self

    def set_default(self, value: Any) -> "FallbackChain":
        """기본값 설정"""
        self.default_value = value
        return self

    def execute(self, *args, **kwargs) -> Any:
        """폴백 체인 실행"""
        errors = []

        # 1. 등록된 제공자 순차 시도
        for provider_name, func in self.providers:
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    logger.debug(f"FallbackChain[{self.name}]: success from {provider_name}")
                    return result
            except Exception as e:
                logger.warning(f"FallbackChain[{self.name}]: {provider_name} failed - {type(e).__name__}: {e}")
                errors.append((provider_name, e))

        # 2. 캐시 폴백 시도
        if self.cache_fallback:
            try:
                cached = self.cache_fallback(*args, **kwargs)
                if cached is not None:
                    logger.info(f"FallbackChain[{self.name}]: using cached data")
                    return cached
            except Exception as e:
                logger.warning(f"FallbackChain[{self.name}]: cache fallback failed - {e}")

        # 3. 기본값 반환
        logger.warning(f"FallbackChain[{self.name}]: all providers failed, returning default")
        return self.default_value


# ===== 서킷 브레이커 인스턴스 =====
# 주요 API별 서킷 브레이커 사전 생성
circuit_yfinance = CircuitBreaker.get_instance("yfinance", failure_threshold=5, reset_timeout=60)
circuit_llm = CircuitBreaker.get_instance("llm", failure_threshold=5, reset_timeout=60)
circuit_sec = CircuitBreaker.get_instance("sec_edgar", failure_threshold=5, reset_timeout=120)
circuit_rss = CircuitBreaker.get_instance("rss", failure_threshold=10, reset_timeout=30)


def get_all_circuit_status() -> dict:
    """모든 서킷 브레이커 상태 조회"""
    return {
        name: cb.get_status()
        for name, cb in CircuitBreaker._instances.items()
    }


def reset_all_circuits() -> None:
    """모든 서킷 브레이커 리셋"""
    for cb in CircuitBreaker._instances.values():
        cb.reset()
