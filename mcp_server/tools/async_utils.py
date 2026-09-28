"""
비동기 유틸리티 모듈 - 병렬 데이터 수집 및 처리

Features:
- asyncio.gather() 기반 병렬 처리
- Semaphore로 동시 요청 수 제한 (rate limit 대응)
- 동기 함수를 비동기로 래핑
- 배치 처리 지원
"""
from __future__ import annotations
from typing import List, Dict, Any, Callable, TypeVar, Optional, Sequence
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 기본 설정
DEFAULT_MAX_CONCURRENT = 5  # 동시 요청 수
DEFAULT_BATCH_SIZE = 10     # 배치 크기
_executor = ThreadPoolExecutor(max_workers=10)


class AsyncBatcher:
    """배치 비동기 처리기

    사용 예시:
        batcher = AsyncBatcher(max_concurrent=5)

        # 여러 티커의 펀더멘털을 병렬로 조회
        results = await batcher.gather(
            get_fundamentals_snapshot,
            ["AAPL", "MSFT", "GOOGL", "NVDA"]
        )
    """

    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def run_sync_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """동기 함수를 스레드에서 비동기로 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))

    async def _run_with_semaphore(self, func: Callable[..., T], *args, **kwargs) -> T:
        """세마포어로 동시 실행 수 제한"""
        async with self.semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await self.run_sync_in_thread(func, *args, **kwargs)

    async def gather(
        self,
        func: Callable[..., T],
        items: Sequence[Any],
        return_exceptions: bool = True
    ) -> List[T]:
        """여러 항목에 대해 함수를 병렬 실행

        Args:
            func: 실행할 함수 (동기/비동기 모두 지원)
            items: 함수에 전달할 항목 리스트
            return_exceptions: True면 예외도 결과에 포함

        Returns:
            각 항목에 대한 결과 리스트
        """
        tasks = [self._run_with_semaphore(func, item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)

    async def gather_with_args(
        self,
        func: Callable[..., T],
        args_list: List[tuple],
        return_exceptions: bool = True
    ) -> List[T]:
        """여러 인자 세트에 대해 함수를 병렬 실행

        Args:
            func: 실행할 함수
            args_list: [(arg1, arg2, ...), ...] 형태의 인자 리스트
            return_exceptions: True면 예외도 결과에 포함
        """
        tasks = [self._run_with_semaphore(func, *args) for args in args_list]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)

    async def gather_dict(
        self,
        func: Callable[..., T],
        items: Sequence[Any],
        key_func: Optional[Callable[[Any], str]] = None,
        return_exceptions: bool = True
    ) -> Dict[str, T]:
        """결과를 딕셔너리로 반환

        Args:
            func: 실행할 함수
            items: 항목 리스트
            key_func: 키 생성 함수 (기본: str(item))
            return_exceptions: True면 예외도 결과에 포함
        """
        results = await self.gather(func, items, return_exceptions)
        key_fn = key_func or str
        return {key_fn(item): result for item, result in zip(items, results)}


async def parallel_map(
    func: Callable[..., T],
    items: Sequence[Any],
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> List[T]:
    """함수를 여러 항목에 병렬 적용

    Args:
        func: 실행할 함수 (동기/비동기 모두 지원)
        items: 항목 리스트
        max_concurrent: 최대 동시 실행 수

    사용 예시:
        results = await parallel_map(get_fundamentals_snapshot, ["AAPL", "MSFT", "GOOGL"])
    """
    batcher = AsyncBatcher(max_concurrent=max_concurrent)
    return await batcher.gather(func, items)


async def parallel_map_dict(
    func: Callable[..., T],
    items: Sequence[Any],
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> Dict[str, T]:
    """함수를 여러 항목에 병렬 적용하고 딕셔너리로 반환

    사용 예시:
        results = await parallel_map_dict(get_fundamentals_snapshot, ["AAPL", "MSFT"])
        # {"AAPL": {...}, "MSFT": {...}}
    """
    batcher = AsyncBatcher(max_concurrent=max_concurrent)
    return await batcher.gather_dict(func, items)


def run_async(coro):
    """비동기 코루틴을 동기 컨텍스트에서 실행

    사용 예시:
        result = run_async(parallel_map(func, items))
    """
    try:
        loop = asyncio.get_running_loop()
        # 이미 이벤트 루프가 실행 중이면 새 스레드에서 실행
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # 이벤트 루프가 없으면 새로 생성
        return asyncio.run(coro)


def make_async(func: Callable[..., T]) -> Callable[..., T]:
    """동기 함수를 비동기 함수로 래핑

    사용 예시:
        @make_async
        def slow_function(x):
            time.sleep(1)
            return x * 2

        result = await slow_function(5)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))
    return wrapper


class BatchProcessor:
    """배치 단위로 항목 처리

    사용 예시:
        processor = BatchProcessor(batch_size=10, max_concurrent=5)
        results = await processor.process(func, items)
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self.batch_size = batch_size
        self.batcher = AsyncBatcher(max_concurrent=max_concurrent)

    async def process(
        self,
        func: Callable[..., T],
        items: Sequence[Any],
        on_batch_complete: Optional[Callable[[int, int], None]] = None
    ) -> List[T]:
        """배치 단위로 처리

        Args:
            func: 실행할 함수
            items: 항목 리스트
            on_batch_complete: 배치 완료 시 콜백 (current_batch, total_batches)
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self.batcher.gather(func, batch)
            results.extend(batch_results)

            if on_batch_complete:
                current_batch = i // self.batch_size + 1
                on_batch_complete(current_batch, total_batches)

        return results


# 편의 함수들
