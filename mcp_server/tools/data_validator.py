"""
데이터 품질 검증 모듈
- 누락 데이터 검증 및 보간
- 이상치 탐지 및 처리
- 데이터 정합성 검사
- 품질 리포트 생성
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import pandas as pd
import numpy as np
import yfinance as yf


logger = logging.getLogger(__name__)

# 경고 무시
warnings.filterwarnings('ignore', category=FutureWarning)


class QualityLevel(Enum):
    """데이터 품질 등급"""
    EXCELLENT = "excellent"  # 95%+ 품질
    GOOD = "good"            # 85-95% 품질
    FAIR = "fair"            # 70-85% 품질
    POOR = "poor"            # 50-70% 품질
    CRITICAL = "critical"    # 50% 미만


@dataclass
class ValidationResult:
    """검증 결과"""
    passed: bool
    check_name: str
    message: str
    details: Optional[Dict] = None
    severity: str = "info"  # info, warning, error

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QualityReport:
    """품질 리포트"""
    ticker: str
    quality_score: float
    quality_level: QualityLevel
    checks: List[ValidationResult]
    summary: Dict
    recommendations: List[str]
    generated_at: str

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level.value,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at
        }


class DataValidator:
    """데이터 품질 검증기"""

    def __init__(self):
        self.checks = []

    def validate_price_data(self, df: pd.DataFrame, ticker: str = "") -> QualityReport:
        """
        가격 데이터 종합 검증

        Args:
            df: OHLCV DataFrame
            ticker: 종목 심볼

        Returns:
            QualityReport
        """
        self.checks = []
        recommendations = []

        if df.empty:
            return QualityReport(
                ticker=ticker,
                quality_score=0,
                quality_level=QualityLevel.CRITICAL,
                checks=[ValidationResult(False, "empty_data", "데이터가 비어 있습니다.", severity="error")],
                summary={"total_rows": 0},
                recommendations=["데이터 소스를 확인하세요."],
                generated_at=datetime.now().isoformat()
            )

        # 컬럼 정규화
        df = self._normalize_columns(df)

        # 1. 기본 검사
        self._check_required_columns(df)
        self._check_data_types(df)
        self._check_date_range(df)

        # 2. 누락 데이터 검사
        missing_result = self._check_missing_values(df)
        if missing_result.details and missing_result.details.get("missing_pct", 0) > 5:
            recommendations.append("누락 데이터가 5% 이상입니다. 보간 처리를 권장합니다.")

        # 3. 이상치 검사
        outlier_result = self._check_outliers(df)
        if outlier_result.details and outlier_result.details.get("outlier_count", 0) > 0:
            recommendations.append("이상치가 감지되었습니다. 윈저화 처리를 권장합니다.")

        # 4. 0값 검사
        zero_result = self._check_zero_values(df)
        if zero_result.details and zero_result.details.get("zero_count", 0) > 0:
            recommendations.append("0값이 감지되었습니다. 데이터 정합성을 확인하세요.")

        # 5. 가격 정합성 검사
        self._check_price_consistency(df)

        # 6. 거래량 검사
        self._check_volume(df)

        # 7. 날짜 연속성 검사
        gap_result = self._check_date_gaps(df)
        if gap_result.details and gap_result.details.get("gap_count", 0) > 5:
            recommendations.append("날짜 갭이 많습니다. 휴장일 여부를 확인하세요.")

        # 8. 변동성 검사
        self._check_extreme_moves(df)

        # 품질 점수 계산
        passed_count = sum(1 for c in self.checks if c.passed)
        total_count = len(self.checks)
        quality_score = (passed_count / total_count * 100) if total_count > 0 else 0

        # 품질 등급 결정
        if quality_score >= 95:
            quality_level = QualityLevel.EXCELLENT
        elif quality_score >= 85:
            quality_level = QualityLevel.GOOD
        elif quality_score >= 70:
            quality_level = QualityLevel.FAIR
        elif quality_score >= 50:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.CRITICAL

        # 요약 생성
        summary = {
            "total_rows": len(df),
            "date_range": f"{df['Date'].min()} ~ {df['Date'].max()}" if "Date" in df.columns else "N/A",
            "passed_checks": passed_count,
            "total_checks": total_count,
            "warnings": sum(1 for c in self.checks if c.severity == "warning"),
            "errors": sum(1 for c in self.checks if c.severity == "error")
        }

        if not recommendations:
            recommendations.append("데이터 품질이 양호합니다.")

        return QualityReport(
            ticker=ticker,
            quality_score=round(quality_score, 1),
            quality_level=quality_level,
            checks=self.checks,
            summary=summary,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat()
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 정규화"""
        df = df.copy()

        # MultiIndex 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 인덱스를 Date 컬럼으로
        if df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # 컬럼명 표준화
        col_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj close": "Adj Close",
            "volume": "Volume"
        }
        df.columns = [col_map.get(c.lower(), c) for c in df.columns]

        return df

    def _check_required_columns(self, df: pd.DataFrame):
        """필수 컬럼 검사"""
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="required_columns",
                message=f"필수 컬럼 누락: {missing}",
                details={"missing_columns": missing},
                severity="error"
            ))
        else:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="required_columns",
                message="필수 컬럼이 모두 존재합니다."
            ))

    def _check_data_types(self, df: pd.DataFrame):
        """데이터 타입 검사"""
        issues = []
        price_cols = ["Open", "High", "Low", "Close"]

        for col in price_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"{col}: 숫자형이 아님")

        if "Volume" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Volume"]):
                issues.append("Volume: 숫자형이 아님")

        if issues:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="data_types",
                message=f"데이터 타입 문제: {issues}",
                details={"issues": issues},
                severity="warning"
            ))
        else:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="data_types",
                message="데이터 타입이 올바릅니다."
            ))

    def _check_date_range(self, df: pd.DataFrame):
        """날짜 범위 검사"""
        if "Date" not in df.columns:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="date_range",
                message="Date 컬럼이 없습니다.",
                severity="warning"
            ))
            return

        try:
            df["Date"] = pd.to_datetime(df["Date"])
            min_date = df["Date"].min()
            max_date = df["Date"].max()
            date_range = (max_date - min_date).days

            self.checks.append(ValidationResult(
                passed=True,
                check_name="date_range",
                message=f"날짜 범위: {min_date.date()} ~ {max_date.date()} ({date_range}일)",
                details={
                    "min_date": str(min_date.date()),
                    "max_date": str(max_date.date()),
                    "days": date_range
                }
            ))
        except Exception as e:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="date_range",
                message=f"날짜 파싱 오류: {e}",
                severity="error"
            ))

    def _check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """누락 데이터 검사"""
        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        cols_to_check = [c for c in price_cols if c in df.columns]

        missing_counts = df[cols_to_check].isna().sum()
        total_missing = missing_counts.sum()
        total_cells = len(df) * len(cols_to_check)
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        if total_missing == 0:
            result = ValidationResult(
                passed=True,
                check_name="missing_values",
                message="누락 데이터가 없습니다.",
                details={"missing_count": 0, "missing_pct": 0}
            )
        elif missing_pct < 1:
            result = ValidationResult(
                passed=True,
                check_name="missing_values",
                message=f"누락 데이터: {total_missing}개 ({missing_pct:.2f}%)",
                details={
                    "missing_count": int(total_missing),
                    "missing_pct": round(missing_pct, 2),
                    "by_column": missing_counts.to_dict()
                },
                severity="info"
            )
        else:
            result = ValidationResult(
                passed=False,
                check_name="missing_values",
                message=f"누락 데이터 과다: {total_missing}개 ({missing_pct:.2f}%)",
                details={
                    "missing_count": int(total_missing),
                    "missing_pct": round(missing_pct, 2),
                    "by_column": missing_counts.to_dict()
                },
                severity="warning"
            )

        self.checks.append(result)
        return result

    def _check_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """이상치 검사 (3σ 기준)"""
        if "Close" not in df.columns:
            result = ValidationResult(
                passed=True,
                check_name="outliers",
                message="Close 컬럼이 없어 이상치 검사를 건너뜁니다."
            )
            self.checks.append(result)
            return result

        close = df["Close"].dropna()
        if len(close) < 10:
            result = ValidationResult(
                passed=True,
                check_name="outliers",
                message="데이터가 충분하지 않아 이상치 검사를 건너뜁니다."
            )
            self.checks.append(result)
            return result

        # 일일 수익률 기준 이상치 탐지
        returns = close.pct_change().dropna()
        mean = returns.mean()
        std = returns.std()

        if std == 0:
            result = ValidationResult(
                passed=True,
                check_name="outliers",
                message="변동성이 없어 이상치 검사를 건너뜁니다."
            )
            self.checks.append(result)
            return result

        z_scores = (returns - mean) / std
        outliers = returns[abs(z_scores) > 3]
        outlier_count = len(outliers)

        if outlier_count == 0:
            result = ValidationResult(
                passed=True,
                check_name="outliers",
                message="3σ 기준 이상치가 없습니다.",
                details={"outlier_count": 0, "threshold": "3σ"}
            )
        elif outlier_count < 5:
            result = ValidationResult(
                passed=True,
                check_name="outliers",
                message=f"이상치 {outlier_count}개 감지 (허용 범위)",
                details={
                    "outlier_count": outlier_count,
                    "threshold": "3σ",
                    "outlier_returns": outliers.tolist()[:5]
                },
                severity="info"
            )
        else:
            result = ValidationResult(
                passed=False,
                check_name="outliers",
                message=f"이상치 과다: {outlier_count}개",
                details={
                    "outlier_count": outlier_count,
                    "threshold": "3σ",
                    "max_outlier": float(outliers.abs().max())
                },
                severity="warning"
            )

        self.checks.append(result)
        return result

    def _check_zero_values(self, df: pd.DataFrame) -> ValidationResult:
        """0값 검사"""
        price_cols = ["Open", "High", "Low", "Close"]
        cols_to_check = [c for c in price_cols if c in df.columns]

        zero_counts = {}
        total_zeros = 0
        for col in cols_to_check:
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                zero_counts[col] = int(zeros)
                total_zeros += zeros

        if total_zeros == 0:
            result = ValidationResult(
                passed=True,
                check_name="zero_values",
                message="가격 0값이 없습니다.",
                details={"zero_count": 0}
            )
        else:
            result = ValidationResult(
                passed=False,
                check_name="zero_values",
                message=f"가격 0값 감지: {total_zeros}개",
                details={"zero_count": total_zeros, "by_column": zero_counts},
                severity="warning"
            )

        self.checks.append(result)
        return result

    def _check_price_consistency(self, df: pd.DataFrame):
        """가격 정합성 검사 (High >= Low, High >= Open/Close 등)"""
        issues = []

        if all(c in df.columns for c in ["High", "Low"]):
            invalid = df[df["High"] < df["Low"]]
            if len(invalid) > 0:
                issues.append(f"High < Low: {len(invalid)}건")

        if all(c in df.columns for c in ["High", "Open", "Close"]):
            invalid_open = df[df["High"] < df["Open"]]
            invalid_close = df[df["High"] < df["Close"]]
            if len(invalid_open) > 0:
                issues.append(f"High < Open: {len(invalid_open)}건")
            if len(invalid_close) > 0:
                issues.append(f"High < Close: {len(invalid_close)}건")

        if all(c in df.columns for c in ["Low", "Open", "Close"]):
            invalid_open = df[df["Low"] > df["Open"]]
            invalid_close = df[df["Low"] > df["Close"]]
            if len(invalid_open) > 0:
                issues.append(f"Low > Open: {len(invalid_open)}건")
            if len(invalid_close) > 0:
                issues.append(f"Low > Close: {len(invalid_close)}건")

        if issues:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="price_consistency",
                message=f"가격 정합성 오류: {issues}",
                details={"issues": issues},
                severity="error"
            ))
        else:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="price_consistency",
                message="가격 정합성이 올바릅니다."
            ))

    def _check_volume(self, df: pd.DataFrame):
        """거래량 검사"""
        if "Volume" not in df.columns:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="volume",
                message="Volume 컬럼이 없습니다.",
                severity="info"
            ))
            return

        # 음수 거래량 검사
        negative = (df["Volume"] < 0).sum()
        if negative > 0:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="volume",
                message=f"음수 거래량: {negative}건",
                details={"negative_count": int(negative)},
                severity="error"
            ))
            return

        # 0 거래량 비율 검사
        zero_volume = (df["Volume"] == 0).sum()
        zero_pct = zero_volume / len(df) * 100

        if zero_pct > 20:
            self.checks.append(ValidationResult(
                passed=False,
                check_name="volume",
                message=f"거래량 0 과다: {zero_volume}건 ({zero_pct:.1f}%)",
                details={"zero_count": int(zero_volume), "zero_pct": round(zero_pct, 1)},
                severity="warning"
            ))
        else:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="volume",
                message=f"거래량 데이터 양호 (0값 {zero_pct:.1f}%)",
                details={"zero_count": int(zero_volume), "zero_pct": round(zero_pct, 1)}
            ))

    def _check_date_gaps(self, df: pd.DataFrame) -> ValidationResult:
        """날짜 갭 검사 (주말/휴일 제외)"""
        if "Date" not in df.columns:
            result = ValidationResult(
                passed=True,
                check_name="date_gaps",
                message="Date 컬럼이 없어 갭 검사를 건너뜁니다."
            )
            self.checks.append(result)
            return result

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # 날짜 차이 계산
        date_diff = df["Date"].diff().dt.days.dropna()

        # 3일 이상 갭 (주말 + 1일 허용)
        gaps = date_diff[date_diff > 3]
        gap_count = len(gaps)

        if gap_count == 0:
            result = ValidationResult(
                passed=True,
                check_name="date_gaps",
                message="날짜 갭이 정상 범위입니다.",
                details={"gap_count": 0}
            )
        elif gap_count < 5:
            result = ValidationResult(
                passed=True,
                check_name="date_gaps",
                message=f"날짜 갭 {gap_count}개 (휴장일 가능성)",
                details={
                    "gap_count": gap_count,
                    "max_gap_days": int(gaps.max())
                },
                severity="info"
            )
        else:
            result = ValidationResult(
                passed=False,
                check_name="date_gaps",
                message=f"날짜 갭 과다: {gap_count}개",
                details={
                    "gap_count": gap_count,
                    "max_gap_days": int(gaps.max())
                },
                severity="warning"
            )

        self.checks.append(result)
        return result

    def _check_extreme_moves(self, df: pd.DataFrame):
        """극단적 가격 변동 검사 (하루 ±20% 이상)"""
        if "Close" not in df.columns:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="extreme_moves",
                message="Close 컬럼이 없어 변동 검사를 건너뜁니다."
            ))
            return

        returns = df["Close"].pct_change().dropna()
        extreme = returns[abs(returns) > 0.2]  # 20% 이상

        if len(extreme) == 0:
            self.checks.append(ValidationResult(
                passed=True,
                check_name="extreme_moves",
                message="극단적 가격 변동(±20%)이 없습니다."
            ))
        else:
            self.checks.append(ValidationResult(
                passed=True,  # 경고만, 실패는 아님
                check_name="extreme_moves",
                message=f"극단적 변동 {len(extreme)}건 (주식분할/합병 가능성)",
                details={
                    "extreme_count": len(extreme),
                    "max_move": round(float(extreme.abs().max()) * 100, 1)
                },
                severity="warning"
            ))


# ============================================================
# 데이터 정제 함수
# ============================================================

def clean_price_data(
    df: pd.DataFrame,
    fill_missing: bool = True,
    remove_zeros: bool = True,
    winsorize_outliers: bool = True,
    outlier_threshold: float = 3.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    가격 데이터 정제

    Args:
        df: 원본 DataFrame
        fill_missing: 누락값 보간 여부
        remove_zeros: 0값 제거 여부
        winsorize_outliers: 이상치 윈저화 여부
        outlier_threshold: 이상치 임계값 (σ)

    Returns:
        정제된 DataFrame, 변경 내역
    """
    df = df.copy()
    changes = {
        "original_rows": len(df),
        "filled_missing": 0,
        "removed_zeros": 0,
        "winsorized": 0
    }

    # 컬럼 정규화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price_cols = ["Open", "High", "Low", "Close"]
    cols_to_clean = [c for c in price_cols if c in df.columns]

    # 1. 누락값 보간
    if fill_missing:
        for col in cols_to_clean:
            missing_before = df[col].isna().sum()
            df[col] = df[col].ffill().bfill()
            missing_after = df[col].isna().sum()
            changes["filled_missing"] += missing_before - missing_after

    # 2. 0값 처리
    if remove_zeros:
        for col in cols_to_clean:
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].ffill().bfill()
                changes["removed_zeros"] += zeros

    # 3. 이상치 윈저화
    if winsorize_outliers and "Close" in df.columns:
        close = df["Close"]
        returns = close.pct_change()

        mean = returns.mean()
        std = returns.std()

        if std > 0:
            z_scores = (returns - mean) / std
            outlier_mask = abs(z_scores) > outlier_threshold

            if outlier_mask.any():
                # 윈저화: 이상치를 임계값으로 클리핑
                upper = mean + outlier_threshold * std
                lower = mean - outlier_threshold * std
                returns_clipped = returns.clip(lower=lower, upper=upper)

                # 가격 복원
                for col in cols_to_clean:
                    if col != "Close":
                        ratio = df[col] / df["Close"]
                        df[col] = df["Close"].iloc[0] * (1 + returns_clipped).cumprod() * ratio
                df["Close"] = df["Close"].iloc[0] * (1 + returns_clipped).cumprod()

                changes["winsorized"] = int(outlier_mask.sum())

    # Volume 정제
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
        df["Volume"] = df["Volume"].clip(lower=0)

    changes["final_rows"] = len(df)
    changes["rows_removed"] = changes["original_rows"] - changes["final_rows"]

    return df, changes


def validate_and_clean(
    ticker: str,
    period: str = "1y",
    auto_clean: bool = True
) -> Dict[str, Any]:
    """
    데이터 검증 및 정제 통합

    Args:
        ticker: 종목 심볼
        period: 기간
        auto_clean: 자동 정제 여부

    Returns:
        검증 결과 및 정제된 데이터
    """
    # 데이터 조회
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return {"error": f"No data for {ticker}"}
        data = data.reset_index()
    except Exception as e:
        return {"error": str(e)}

    # 검증
    validator = DataValidator()
    report = validator.validate_price_data(data, ticker)

    result = {
        "ticker": ticker,
        "period": period,
        "validation": report.to_dict()
    }

    # 자동 정제
    if auto_clean and report.quality_level in [QualityLevel.FAIR, QualityLevel.POOR, QualityLevel.CRITICAL]:
        cleaned_data, changes = clean_price_data(data)
        result["cleaning"] = changes

        # 정제 후 재검증
        cleaned_report = validator.validate_price_data(cleaned_data, ticker)
        result["validation_after_clean"] = cleaned_report.to_dict()
        result["quality_improved"] = cleaned_report.quality_score > report.quality_score

    return result


def get_data_quality_summary(tickers: List[str], period: str = "1y") -> Dict[str, Any]:
    """
    여러 종목 데이터 품질 요약

    Args:
        tickers: 종목 리스트
        period: 기간

    Returns:
        종목별 품질 요약
    """
    results = []
    validator = DataValidator()

    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data.empty:
                results.append({
                    "ticker": ticker,
                    "quality_score": 0,
                    "quality_level": "critical",
                    "error": "No data"
                })
                continue

            data = data.reset_index()
            report = validator.validate_price_data(data, ticker)

            results.append({
                "ticker": ticker,
                "quality_score": report.quality_score,
                "quality_level": report.quality_level.value,
                "warnings": report.summary.get("warnings", 0),
                "errors": report.summary.get("errors", 0),
                "rows": report.summary.get("total_rows", 0)
            })
        except Exception as e:
            results.append({
                "ticker": ticker,
                "quality_score": 0,
                "quality_level": "critical",
                "error": str(e)
            })

    # 요약 통계
    scores = [r["quality_score"] for r in results if "error" not in r]
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "tickers": results,
        "summary": {
            "total_tickers": len(tickers),
            "average_score": round(avg_score, 1),
            "excellent": sum(1 for r in results if r.get("quality_level") == "excellent"),
            "good": sum(1 for r in results if r.get("quality_level") == "good"),
            "fair": sum(1 for r in results if r.get("quality_level") == "fair"),
            "poor": sum(1 for r in results if r.get("quality_level") == "poor"),
            "critical": sum(1 for r in results if r.get("quality_level") == "critical")
        },
        "checked_at": datetime.now().isoformat()
    }


# 싱글톤 검증기
_validator = DataValidator()


