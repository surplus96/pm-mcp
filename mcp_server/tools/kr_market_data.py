"""한국 주식 시장 데이터 어댑터 (PyKrx + FinanceDataReader)

PyKrx: KRX 공식 스크래핑, 정확한 시장 분류
FinanceDataReader: 통합 금융 데이터 제공
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
from typing import Optional, Dict
import logging
from mcp_server.tools.cache_manager import cached, TTL

logger = logging.getLogger(__name__)


class KoreanMarketAdapter:
    """한국 주식 시장 데이터 어댑터"""

    def __init__(self):
        self._pykrx_available = False
        self._fdr_available = False
        self._init_libraries()

    def _init_libraries(self):
        """라이브러리 초기화 및 가용성 확인"""
        try:
            import pykrx
            self._pykrx_available = True
            logger.info("PyKrx initialized successfully")
        except ImportError:
            logger.warning("PyKrx not available, Korean stock data may be limited")

        try:
            import FinanceDataReader as fdr
            self._fdr_available = True
            logger.info("FinanceDataReader initialized successfully")
        except ImportError:
            logger.warning("FinanceDataReader not available, Korean stock data may be limited")

    @cached(ttl=TTL.DAILY, prefix="kr_stock_list")
    def get_stock_listing(self, market: str = "KOSPI") -> pd.DataFrame:
        """상장 종목 리스트 조회 (일일 캐시)

        Args:
            market: 시장 구분 ("KOSPI", "KOSDAQ", "KONEX", "ALL")

        Returns:
            상장 종목 DataFrame (columns: Code, Name, Sector, ...)
        """
        if not self._fdr_available:
            logger.error("FinanceDataReader not available")
            return pd.DataFrame()

        try:
            import FinanceDataReader as fdr
            # FDR doesn't accept ``"ALL"`` — it expects a real market name.
            # Map ``ALL`` to the umbrella ``"KRX"`` so callers using the
            # legacy keyword keep working without a noisy log line.
            fdr_market = "KRX" if market.upper() == "ALL" else market
            df = fdr.StockListing(fdr_market)
            logger.info(f"Retrieved {len(df)} stocks from {fdr_market}")
            return df
        except Exception as e:
            logger.error(f"Failed to get stock listing for {market}: {e}")
            return pd.DataFrame()

    def get_market_by_ticker(self, ticker: str) -> Optional[str]:
        """종목 코드로 시장 구분 조회 (KOSPI/KOSDAQ)

        Args:
            ticker: 6자리 종목 코드 (예: "005930")

        Returns:
            시장 구분 ("KOSPI", "KOSDAQ", None)
        """
        # 전체 상장 종목 리스트에서 조회 (캐시됨)
        all_stocks = self.get_stock_listing("ALL")

        if all_stocks.empty:
            logger.warning(f"Cannot determine market for {ticker}, stock list is empty")
            return None

        # Code 컬럼에서 종목 찾기
        if "Code" in all_stocks.columns:
            matched = all_stocks[all_stocks["Code"] == ticker]
            if not matched.empty and "Market" in matched.columns:
                market = matched.iloc[0]["Market"]
                logger.debug(f"Ticker {ticker} found in market: {market}")
                return market

        logger.warning(f"Ticker {ticker} not found in stock listings")
        return None

    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        market: Optional[str] = None
    ) -> pd.DataFrame:
        """일별 OHLCV 데이터 조회

        Args:
            ticker: 종목 코드 (6자리 또는 .KS/.KQ 포함)
            start: 시작일 (YYYYMMDD 또는 YYYY-MM-DD)
            end: 종료일 (YYYYMMDD 또는 YYYY-MM-DD)
            market: 시장 구분 (None이면 자동 감지)

        Returns:
            OHLCV DataFrame
        """
        if not self._pykrx_available:
            logger.error("PyKrx not available")
            return pd.DataFrame()

        try:
            from pykrx import stock

            # 티커 정규화 (6자리 코드만)
            clean_ticker = ticker.replace(".KS", "").replace(".KQ", "")

            # 날짜 형식 정규화 (YYYYMMDD)
            start_date = self._normalize_date(start) if start else self._default_start_date()
            end_date = self._normalize_date(end) if end else self._default_end_date()

            # PyKrx로 데이터 조회
            # ⚠️ pykrx 시그니처는 ``get_market_ohlcv(from_date, to_date, ticker)``
            # 이다. 과거 버전의 ``(ticker, from_date, to_date)`` 순서로 호출하면
            # pykrx 가 첫 인자를 date 로 해석하다 ``ValueError`` 를 삼키고
            # 빈 DataFrame 을 돌려줘 정적 ``KRX 로그인 실패`` warning 만 남긴다.
            df = stock.get_market_ohlcv(start_date, end_date, clean_ticker)

            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()

            # 인덱스를 컬럼으로 변환 (Date)
            df = df.reset_index()
            df.rename(columns={'날짜': 'Date', '시가': 'Open', '고가': 'High',
                              '저가': 'Low', '종가': 'Close', '거래량': 'Volume'}, inplace=True)

            logger.info(f"Retrieved {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Failed to get OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    @cached(ttl=TTL.FUNDAMENTAL, prefix="kr_fundamental")
    def get_fundamental(self, ticker: str, date: Optional[str] = None) -> Dict:
        """펀더멘털 데이터 조회 (24시간 캐시)

        Args:
            ticker: 종목 코드
            date: 기준일 (YYYYMMDD, None이면 최근 영업일)

        Returns:
            펀더멘털 데이터 딕셔너리
        """
        if not self._pykrx_available:
            logger.error("PyKrx not available")
            return {}

        try:
            from pykrx import stock

            clean_ticker = ticker.replace(".KS", "").replace(".KQ", "")
            date_str = self._normalize_date(date) if date else self._default_end_date()

            # 펀더멘털 데이터 조회
            # pykrx signature: get_market_fundamental_by_date(from, to, ticker)
            fundamental = stock.get_market_fundamental_by_date(date_str, date_str, clean_ticker)

            if fundamental.empty:
                logger.warning(f"No fundamental data for {ticker}")
                return {}

            # Series를 dict로 변환
            result = {
                "ticker": ticker,
                "date": date_str,
                "PER": float(fundamental.iloc[0]["PER"]) if "PER" in fundamental.columns else None,
                "PBR": float(fundamental.iloc[0]["PBR"]) if "PBR" in fundamental.columns else None,
                "DIV": float(fundamental.iloc[0]["DIV"]) if "DIV" in fundamental.columns else None,
            }

            logger.info(f"Retrieved fundamental data for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Failed to get fundamental for {ticker}: {e}")
            return {}

    def get_ticker_name(self, ticker: str) -> Optional[str]:
        """종목명 조회

        Args:
            ticker: 종목 코드

        Returns:
            종목명 (없으면 None)
        """
        all_stocks = self.get_stock_listing("ALL")

        if all_stocks.empty:
            return None

        clean_ticker = ticker.replace(".KS", "").replace(".KQ", "")

        if "Code" in all_stocks.columns and "Name" in all_stocks.columns:
            matched = all_stocks[all_stocks["Code"] == clean_ticker]
            if not matched.empty:
                return matched.iloc[0]["Name"]

        return None

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """날짜 문자열을 YYYYMMDD 형식으로 정규화"""
        if not date_str:
            return ""

        # 이미 YYYYMMDD 형식이면 그대로 반환
        if len(date_str) == 8 and date_str.isdigit():
            return date_str

        # YYYY-MM-DD 형식이면 변환
        if "-" in date_str:
            return date_str.replace("-", "")

        # 그 외는 그대로 반환
        return date_str

    @staticmethod
    def _default_start_date() -> str:
        """기본 시작일 (1년 전)"""
        one_year_ago = datetime.now().replace(year=datetime.now().year - 1)
        return one_year_ago.strftime('%Y%m%d')

    @staticmethod
    def _default_end_date() -> str:
        """기본 종료일 (오늘)"""
        return datetime.now().strftime('%Y%m%d')


# 싱글톤 인스턴스
_kr_adapter = None


def get_kr_adapter() -> KoreanMarketAdapter:
    """한국 마켓 어댑터 싱글톤 인스턴스 반환"""
    global _kr_adapter
    if _kr_adapter is None:
        _kr_adapter = KoreanMarketAdapter()
    return _kr_adapter
