"""기술적 지표 계산 모듈 (10개 팩터)

TA-Lib 기반 기술적 분석 지표:
- 모멘텀 지표 (4개): RSI, MACD, Stochastic, Williams %R
- 트렌드 지표 (3개): ADX, CCI, MA Cross
- 변동성 지표 (2개): Bollinger Band Width, ATR
- 거래량 지표 (1개): OBV
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA library not available, technical indicators will be disabled")


class TechnicalFactors:
    """기술적 팩터 계산 클래스"""

    @staticmethod
    def calculate_all(df: pd.DataFrame, periods: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """10개 기술적 팩터 계산

        Args:
            df: OHLCV DataFrame (컬럼: Open, High, Low, Close, Volume)
            periods: 지표별 기간 설정 (기본값 사용 가능)

        Returns:
            기술적 팩터 딕셔너리

        Example:
            >>> df = get_prices("AAPL", period="6mo")
            >>> factors = TechnicalFactors.calculate_all(df)
            >>> print(factors['RSI'], factors['MACD'])
        """
        if not TA_AVAILABLE:
            logger.warning("TA library not available")
            return {}

        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for technical indicators (rows: {len(df)})")
            return {}

        # 필수 컬럼 확인
        required_cols = ['Close', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            return {}

        # 기본 기간 설정
        if periods is None:
            periods = {
                'RSI': 14,
                'MACD_fast': 12,
                'MACD_slow': 26,
                'MACD_signal': 9,
                'Stochastic': 14,
                'Williams': 14,
                'ADX': 14,
                'CCI': 20,
                'BB': 20,
                'ATR': 14,
                'MA_short': 20,
                'MA_long': 50,
            }

        try:
            factors = {}

            # === 모멘텀 지표 (4개) ===
            factors['RSI'] = TechnicalFactors._calculate_rsi(df, periods['RSI'])
            factors['MACD'] = TechnicalFactors._calculate_macd(
                df, periods['MACD_fast'], periods['MACD_slow'], periods['MACD_signal']
            )
            factors['Stochastic'] = TechnicalFactors._calculate_stochastic(df, periods['Stochastic'])
            factors['Williams_R'] = TechnicalFactors._calculate_williams_r(df, periods['Williams'])

            # === 트렌드 지표 (3개) ===
            factors['ADX'] = TechnicalFactors._calculate_adx(df, periods['ADX'])
            factors['CCI'] = TechnicalFactors._calculate_cci(df, periods['CCI'])
            factors['MA_Cross'] = TechnicalFactors._calculate_ma_cross(
                df, periods['MA_short'], periods['MA_long']
            )

            # === 변동성 지표 (2개) ===
            factors['BB_Width'] = TechnicalFactors._calculate_bb_width(df, periods['BB'])
            factors['ATR'] = TechnicalFactors._calculate_atr(df, periods['ATR'])

            # === 거래량 지표 (1개) ===
            factors['OBV'] = TechnicalFactors._calculate_obv(df)

            # NaN 값 제거
            factors = {k: v for k, v in factors.items() if not pd.isna(v)}

            logger.info(f"Calculated {len(factors)} technical factors")
            return factors

        except Exception as e:
            logger.error(f"Failed to calculate technical factors: {e}")
            return {}

    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """RSI (Relative Strength Index) 계산"""
        try:
            rsi = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
            return float(rsi.iloc[-1]) if not rsi.empty else np.nan
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """MACD (Moving Average Convergence Divergence) 계산"""
        try:
            macd_indicator = ta.trend.MACD(df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
            macd = macd_indicator.macd()
            return float(macd.iloc[-1]) if not macd.empty else np.nan
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, period: int = 14) -> float:
        """Stochastic Oscillator 계산"""
        try:
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close'], window=period, smooth_window=3
            ).stoch()
            return float(stoch.iloc[-1]) if not stoch.empty else np.nan
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> float:
        """Williams %R 계산"""
        try:
            williams = ta.momentum.WilliamsRIndicator(
                df['High'], df['Low'], df['Close'], lbp=period
            ).williams_r()
            return float(williams.iloc[-1]) if not williams.empty else np.nan
        except Exception as e:
            logger.warning(f"Williams %R calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
        """ADX (Average Directional Index) 계산"""
        try:
            adx = ta.trend.ADXIndicator(
                df['High'], df['Low'], df['Close'], window=period
            ).adx()
            return float(adx.iloc[-1]) if not adx.empty else np.nan
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> float:
        """CCI (Commodity Channel Index) 계산"""
        try:
            cci = ta.trend.CCIIndicator(
                df['High'], df['Low'], df['Close'], window=period
            ).cci()
            return float(cci.iloc[-1]) if not cci.empty else np.nan
        except Exception as e:
            logger.warning(f"CCI calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_ma_cross(df: pd.DataFrame, short: int = 20, long: int = 50) -> float:
        """이동평균 교차 지표 (MA Cross)

        Returns:
            양수: 골든 크로스 (상승 신호)
            음수: 데드 크로스 (하락 신호)
            값의 절대값: 교차 강도
        """
        try:
            ma_short = df['Close'].rolling(window=short).mean()
            ma_long = df['Close'].rolling(window=long).mean()

            if ma_short.empty or ma_long.empty:
                return np.nan

            # 단기 MA - 장기 MA (정규화)
            current_diff = ma_short.iloc[-1] - ma_long.iloc[-1]
            current_price = df['Close'].iloc[-1]

            # 가격 대비 %로 정규화
            normalized_diff = (current_diff / current_price) * 100
            return float(normalized_diff)
        except Exception as e:
            logger.warning(f"MA Cross calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_bb_width(df: pd.DataFrame, period: int = 20) -> float:
        """Bollinger Band Width 계산 (변동성 지표)"""
        try:
            bb_indicator = ta.volatility.BollingerBands(df['Close'], window=period, window_dev=2)
            bb_width = bb_indicator.bollinger_wband()
            return float(bb_width.iloc[-1]) if not bb_width.empty else np.nan
        except Exception as e:
            logger.warning(f"BB Width calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """ATR (Average True Range) 계산"""
        try:
            atr = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=period
            ).average_true_range()
            return float(atr.iloc[-1]) if not atr.empty else np.nan
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return np.nan

    @staticmethod
    def _calculate_obv(df: pd.DataFrame) -> float:
        """OBV (On Balance Volume) 계산"""
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(
                df['Close'], df['Volume']
            ).on_balance_volume()

            # OBV는 절대값이 크므로 최근 20일 변화율로 정규화
            if len(obv) < 20:
                return np.nan

            obv_change = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) * 100
            return float(obv_change)
        except Exception as e:
            logger.warning(f"OBV calculation failed: {e}")
            return np.nan

    @staticmethod
    def get_factor_interpretation(factors: Dict[str, float]) -> Dict[str, str]:
        """기술적 팩터 해석

        Args:
            factors: 계산된 기술적 팩터

        Returns:
            팩터별 해석 딕셔너리
        """
        interpretation = {}

        # RSI
        if 'RSI' in factors:
            rsi = factors['RSI']
            if rsi > 70:
                interpretation['RSI'] = "과매수 (Overbought)"
            elif rsi < 30:
                interpretation['RSI'] = "과매도 (Oversold)"
            else:
                interpretation['RSI'] = "중립 (Neutral)"

        # MACD
        if 'MACD' in factors:
            macd = factors['MACD']
            if macd > 0:
                interpretation['MACD'] = "상승 추세 (Bullish)"
            else:
                interpretation['MACD'] = "하락 추세 (Bearish)"

        # MA Cross
        if 'MA_Cross' in factors:
            ma_cross = factors['MA_Cross']
            if ma_cross > 1:
                interpretation['MA_Cross'] = "강한 골든 크로스 (Strong Golden Cross)"
            elif ma_cross > 0:
                interpretation['MA_Cross'] = "골든 크로스 (Golden Cross)"
            elif ma_cross < -1:
                interpretation['MA_Cross'] = "강한 데드 크로스 (Strong Death Cross)"
            else:
                interpretation['MA_Cross'] = "데드 크로스 (Death Cross)"

        # ADX
        if 'ADX' in factors:
            adx = factors['ADX']
            if adx > 25:
                interpretation['ADX'] = "강한 추세 (Strong Trend)"
            elif adx > 20:
                interpretation['ADX'] = "추세 형성 중 (Trending)"
            else:
                interpretation['ADX'] = "추세 없음 (No Trend)"

        return interpretation


# 편의 함수
def calculate_technical_score(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> float:
    """기술적 팩터 종합 스코어 계산

    Args:
        df: OHLCV DataFrame
        weights: 팩터별 가중치 (기본값: 균등 가중)

    Returns:
        0-100 사이의 종합 스코어
    """
    factors = TechnicalFactors.calculate_all(df)

    if not factors:
        return 50.0  # 중립 스코어

    # 기본 가중치 (균등)
    if weights is None:
        weights = {k: 1.0 / len(factors) for k in factors.keys()}

    # 팩터별 정규화 및 스코어 계산
    # (각 지표를 0-100 스케일로 변환)
    normalized_scores = {}

    # RSI: 0-100 이미 스케일됨
    if 'RSI' in factors:
        normalized_scores['RSI'] = factors['RSI']

    # MACD: -inf ~ +inf → sigmoid로 0-100 변환
    if 'MACD' in factors:
        macd = factors['MACD']
        normalized_scores['MACD'] = 100 / (1 + np.exp(-macd))

    # 나머지 지표들도 유사하게 정규화...
    # (간단히 하기 위해 RSI, MACD만 사용)

    # 가중 평균
    total_score = sum(
        normalized_scores.get(k, 50) * weights.get(k, 0)
        for k in normalized_scores.keys()
    )

    return float(np.clip(total_score, 0, 100))
