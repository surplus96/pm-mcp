"""
캐시 관리자 모듈 - diskcache 기반 TTL 캐싱 시스템

Features:
- 데이터 유형별 TTL 정책
- 데코레이터 패턴으로 쉬운 적용
- 동시성 안전 (diskcache 내장 락)
- 캐시 통계 및 관리 기능
"""
from __future__ import annotations
from typing import Optional, Any, Callable, TypeVar, ParamSpec
from functools import wraps
import hashlib
import json
import os
import logging

from diskcache import FanoutCache

logger = logging.getLogger(__name__)

# 기본 캐시 디렉토리
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "diskcache"
)

P = ParamSpec("P")
T = TypeVar("T")


def _is_cacheable(value: Any) -> bool:
    """None·빈 DataFrame/Series·빈 컬렉션은 캐싱하지 않는다."""
    if value is None:
        return False
    if getattr(value, "empty", False) is True:  # pandas DataFrame / Series
        return False
    if isinstance(value, (dict, list, tuple, set, str)) and not value:
        return False
    return True


class TTL:
    """TTL 상수 (초 단위)"""
    REALTIME = 15 * 60          # 15분 - 실시간 가격
    INTRADAY = 1 * 60 * 60      # 1시간 - 장중 데이터
    DAILY = 4 * 60 * 60         # 4시간 - 일봉 데이터
    FUNDAMENTAL = 24 * 60 * 60  # 24시간 - 펀더멘털
    NEWS = 1 * 60 * 60          # 1시간 - 뉴스
    FILING = 6 * 60 * 60        # 6시간 - SEC 공시
    METRICS = 4 * 60 * 60       # 4시간 - 계산된 메트릭
    LONG = 7 * 24 * 60 * 60     # 7일 - 장기 캐시


class CacheManager:
    """통합 캐시 관리자

    사용 예시:
        cache = CacheManager()

        # 직접 사용
        cache.set("key", value, ttl=TTL.DAILY)
        value = cache.get("key")

        # 데코레이터 사용
        @cache.cached(ttl=TTL.FUNDAMENTAL, prefix="fundamentals")
        def get_fundamentals(ticker: str):
            ...
    """

    _instance: Optional["CacheManager"] = None

    def __new__(cls, cache_dir: Optional[str] = None, size_limit: int = int(1e9)):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[str] = None, size_limit: int = int(1e9)):
        if self._initialized:
            return

        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # FanoutCache: 여러 샤드로 동시성 향상
        self.cache = FanoutCache(
            self.cache_dir,
            shards=4,
            size_limit=size_limit,
            timeout=1  # 락 타임아웃 1초
        )

        self._initialized = True
        logger.info(f"CacheManager initialized: {self.cache_dir}")

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"{prefix}:{key_hash}"

    def get(self, key: str, default: Any = None) -> Any:
        """캐시에서 값 조회"""
        try:
            value = self.cache.get(key, default=default)
            if value is not default:
                logger.debug(f"Cache HIT: {key}")
            return value
        except Exception as e:
            logger.warning(f"Cache get error: {key} - {e}")
            return default

    def set(self, key: str, value: Any, ttl: int) -> bool:
        """캐시에 값 저장"""
        try:
            self.cache.set(key, value, expire=ttl)
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {key} - {e}")
            return False

    def delete(self, key: str) -> bool:
        """캐시에서 키 삭제"""
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error: {key} - {e}")
            return False

    def clear(self) -> int:
        """전체 캐시 삭제"""
        try:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cache cleared: {count} items")
            return count
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0

    def expire(self) -> int:
        """만료된 캐시 정리"""
        try:
            count = self.cache.expire()
            if count > 0:
                logger.info(f"Expired {count} cache items")
            return count
        except Exception as e:
            logger.warning(f"Cache expire error: {e}")
            return 0

    def stats(self) -> dict:
        """캐시 통계"""
        try:
            return {
                "directory": self.cache_dir,
                "size_bytes": self.cache.volume(),
                "size_mb": round(self.cache.volume() / (1024 * 1024), 2),
                "item_count": len(self.cache),
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {}

    def cached(
        self,
        ttl: int,
        prefix: str = "",
        key_func: Optional[Callable[..., str]] = None
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """캐싱 데코레이터

        Args:
            ttl: 캐시 만료 시간 (초)
            prefix: 캐시 키 접두사
            key_func: 커스텀 키 생성 함수 (선택)

        사용 예시:
            @cache_manager.cached(ttl=TTL.DAILY, prefix="prices")
            def get_prices(ticker: str, period: str = "1y"):
                ...
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # 캐시 키 생성
                func_prefix = prefix or func.__name__
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._make_key(func_prefix, *args, **kwargs)

                # 캐시 조회
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # 함수 실행 및 캐싱 (빈 결과는 일시 장애일 수 있으므로 캐싱하지 않음)
                result = func(*args, **kwargs)
                if _is_cacheable(result):
                    self.set(cache_key, result, ttl)

                return result

            # 캐시 무효화 메서드 추가
            def invalidate(*args: P.args, **kwargs: P.kwargs) -> bool:
                func_prefix = prefix or func.__name__
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._make_key(func_prefix, *args, **kwargs)
                return self.delete(cache_key)

            wrapper.invalidate = invalidate  # type: ignore
            wrapper.cache_manager = self  # type: ignore

            return wrapper
        return decorator


# 글로벌 캐시 매니저 인스턴스
cache_manager = CacheManager()


def cached(
    ttl: int,
    prefix: str = "",
    key_func: Optional[Callable[..., str]] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """편의용 글로벌 캐싱 데코레이터

    사용 예시:
        from mcp_server.tools.cache_manager import cached, TTL

        @cached(ttl=TTL.FUNDAMENTAL, prefix="fundamentals")
        def get_fundamentals(ticker: str):
            ...
    """
    return cache_manager.cached(ttl=ttl, prefix=prefix, key_func=key_func)


# 하위 호환성을 위한 간단한 함수들


