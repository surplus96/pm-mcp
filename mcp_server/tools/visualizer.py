"""
시각화 모듈 (Plotly 기반)
- 캔들스틱 차트
- 기술적 지표 차트 (RSI, MACD, Bollinger Bands)
- 포트폴리오 시각화 (파이차트, 트리맵, 히트맵)
- 상관관계 히트맵
- 상대강도 차트
- 수익률 분포
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from mcp_server.config import safe_filename
from mcp_server.tools.cache_manager import cache_manager, TTL
from mcp_server.tools.yf_utils import normalize_yf_columns

logger = logging.getLogger(__name__)

# 차트 저장 디렉토리
CHARTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "charts"
)
os.makedirs(CHARTS_DIR, exist_ok=True)

# 기본 색상 팔레트
COLORS = {
    "primary": "#2962FF",
    "secondary": "#FF6D00",
    "positive": "#26A69A",
    "negative": "#EF5350",
    "neutral": "#78909C",
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "text": "#212121"
}

# Plotly 테마 설정
LAYOUT_TEMPLATE = {
    "paper_bgcolor": COLORS["background"],
    "plot_bgcolor": COLORS["background"],
    "font": {"color": COLORS["text"], "family": "Arial"},
    "xaxis": {"gridcolor": COLORS["grid"], "showgrid": True},
    "yaxis": {"gridcolor": COLORS["grid"], "showgrid": True},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50}
}


# ============================================================
# 데이터 조회 헬퍼
# ============================================================

def _get_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """OHLCV 데이터 조회"""
    cache_key = f"ohlcv_{ticker}_{period}_{interval}"
    cached = cache_manager.get(cache_key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        data = normalize_yf_columns(
            yf.download(ticker, period=period, interval=interval, progress=False)
        )
        if data.empty:
            return pd.DataFrame()

        data = data.reset_index()
        cache_manager.set(cache_key, data.to_dict('records'), TTL.DAILY)
        return data
    except Exception as e:
        logger.warning(f"Failed to get OHLCV for {ticker}: {e}")
        return pd.DataFrame()


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def _calculate_bollinger(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드 계산"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


# ============================================================
# 캔들스틱 차트
# ============================================================

def create_candlestick_chart(
    ticker: str,
    period: str = "6mo",
    show_volume: bool = True,
    show_ma: List[int] = None,
    title: str = None
) -> go.Figure:
    """
    캔들스틱 차트 생성

    Args:
        ticker: 종목 심볼
        period: 기간
        show_volume: 거래량 표시 여부
        show_ma: 이동평균선 기간 리스트 (예: [20, 50, 200])
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    df = _get_ohlcv(ticker, period)
    if df.empty:
        return _create_error_chart(f"No data for {ticker}")

    # 서브플롯 설정
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"]
        ),
        row=1, col=1
    )

    # 이동평균선
    if show_ma:
        ma_colors = ["#FF9800", "#2196F3", "#9C27B0", "#4CAF50"]
        for i, ma_period in enumerate(show_ma):
            ma = df["Close"].rolling(window=ma_period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["Date"], y=ma,
                    mode="lines",
                    name=f"MA{ma_period}",
                    line={"color": ma_colors[i % len(ma_colors)], "width": 1}
                ),
                row=1, col=1
            )

    # 거래량
    if show_volume:
        colors = [COLORS["positive"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else COLORS["negative"] for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df["Date"], y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

    # 레이아웃
    fig.update_layout(
        title=title or f"{ticker} Price Chart",
        xaxis_rangeslider_visible=False,
        **LAYOUT_TEMPLATE
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ============================================================
# 기술적 지표 차트
# ============================================================

def create_technical_chart(
    ticker: str,
    period: str = "6mo",
    indicators: List[str] = None
) -> go.Figure:
    """
    기술적 지표 차트 생성

    Args:
        ticker: 종목 심볼
        period: 기간
        indicators: 지표 리스트 ["rsi", "macd", "bbands", "volume"]

    Returns:
        Plotly Figure 객체
    """
    df = _get_ohlcv(ticker, period)
    if df.empty:
        return _create_error_chart(f"No data for {ticker}")

    if indicators is None:
        indicators = ["rsi", "macd"]

    # 서브플롯 수 계산
    num_rows = 1 + len(indicators)
    row_heights = [0.4] + [0.6 / len(indicators)] * len(indicators)

    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=[ticker] + [ind.upper() for ind in indicators]
    )

    # 가격 차트 (볼린저 밴드 포함 가능)
    if "bbands" in indicators:
        upper, middle, lower = _calculate_bollinger(df["Close"])
        fig.add_trace(
            go.Scatter(x=df["Date"], y=upper, mode="lines",
                      name="BB Upper", line={"color": "#90CAF9", "width": 1}),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=lower, mode="lines",
                      name="BB Lower", line={"color": "#90CAF9", "width": 1},
                      fill="tonexty", fillcolor="rgba(144, 202, 249, 0.2)"),
            row=1, col=1
        )
        indicators = [i for i in indicators if i != "bbands"]

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Close"],
            mode="lines",
            name="Close",
            line={"color": COLORS["primary"], "width": 2}
        ),
        row=1, col=1
    )

    current_row = 2

    # RSI
    if "rsi" in indicators:
        rsi = _calculate_rsi(df["Close"])
        fig.add_trace(
            go.Scatter(x=df["Date"], y=rsi, mode="lines",
                      name="RSI", line={"color": COLORS["secondary"]}),
            row=current_row, col=1
        )
        # 과매수/과매도 라인
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                     annotation_text="Overbought", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     annotation_text="Oversold", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # MACD
    if "macd" in indicators:
        macd, signal, histogram = _calculate_macd(df["Close"])

        colors = [COLORS["positive"] if h >= 0 else COLORS["negative"] for h in histogram]
        fig.add_trace(
            go.Bar(x=df["Date"], y=histogram, name="Histogram",
                  marker_color=colors, opacity=0.5),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=macd, mode="lines",
                      name="MACD", line={"color": COLORS["primary"]}),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["Date"], y=signal, mode="lines",
                      name="Signal", line={"color": COLORS["secondary"]}),
            row=current_row, col=1
        )
        current_row += 1

    # Volume
    if "volume" in indicators:
        colors = [COLORS["positive"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else COLORS["negative"] for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                  marker_color=colors, opacity=0.7),
            row=current_row, col=1
        )
        current_row += 1

    fig.update_layout(
        title=f"{ticker} Technical Analysis",
        height=150 * num_rows,
        showlegend=True,
        **LAYOUT_TEMPLATE
    )

    return fig


# ============================================================
# 포트폴리오 시각화
# ============================================================

def create_portfolio_pie_chart(
    holdings: Dict[str, float],
    title: str = "Portfolio Allocation"
) -> go.Figure:
    """
    포트폴리오 비중 파이 차트

    Args:
        holdings: {ticker: value} 딕셔너리
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    labels = list(holdings.keys())
    values = list(holdings.values())

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            textposition="outside",
            marker={"colors": px.colors.qualitative.Set2}
        )
    ])

    total = sum(values)
    fig.update_layout(
        title=title,
        annotations=[{
            "text": f"${total:,.0f}",
            "x": 0.5, "y": 0.5,
            "font_size": 20,
            "showarrow": False
        }],
        **LAYOUT_TEMPLATE
    )

    return fig


def create_portfolio_treemap(
    holdings: Dict[str, Dict],
    title: str = "Portfolio Treemap"
) -> go.Figure:
    """
    포트폴리오 트리맵 (섹터별 그룹핑)

    Args:
        holdings: {ticker: {value, sector, pnl_percent}} 딕셔너리
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    labels = ["Portfolio"]
    parents = [""]
    values = [0]
    colors = [0]

    # 섹터별 그룹핑
    sectors = {}
    for ticker, data in holdings.items():
        sector = data.get("sector", "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((ticker, data))

    for sector, items in sectors.items():
        # 섹터 노드
        sector_value = sum(item[1].get("value", 0) for item in items)
        labels.append(sector)
        parents.append("Portfolio")
        values.append(sector_value)
        colors.append(0)

        # 종목 노드
        for ticker, data in items:
            labels.append(ticker)
            parents.append(sector)
            values.append(data.get("value", 0))
            colors.append(data.get("pnl_percent", 0))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale="RdYlGn",
            cmid=0
        ),
        textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>P&L: %{color:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        **LAYOUT_TEMPLATE
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: Dict[str, Dict[str, float]],
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    상관관계 히트맵

    Args:
        correlation_matrix: 상관관계 매트릭스 딕셔너리
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    df = pd.DataFrame(correlation_matrix)
    tickers = list(df.columns)

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=tickers,
        y=tickers,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis={"side": "bottom"},
        **LAYOUT_TEMPLATE
    )

    return fig


def create_sector_bar_chart(
    sectors: List[Dict],
    title: str = "Sector Allocation"
) -> go.Figure:
    """
    섹터별 비중 막대 차트

    Args:
        sectors: [{sector, weight, value}] 리스트
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    df = pd.DataFrame(sectors)
    df = df.sort_values("weight", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["weight"],
        y=df["sector"],
        orientation="h",
        marker_color=px.colors.qualitative.Set2[:len(df)],
        text=[f"{w:.1f}%" for w in df["weight"]],
        textposition="outside"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Weight (%)",
        yaxis_title="",
        **LAYOUT_TEMPLATE
    )

    return fig


# ============================================================
# 비교 차트
# ============================================================

def create_comparison_chart(
    tickers: List[str],
    period: str = "1y",
    normalize: bool = True,
    title: str = None
) -> go.Figure:
    """
    종목 비교 차트 (정규화)

    Args:
        tickers: 종목 리스트
        period: 기간
        normalize: 100 기준 정규화 여부
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, ticker in enumerate(tickers):
        df = _get_ohlcv(ticker, period)
        if df.empty:
            continue

        prices = df["Close"]
        if normalize:
            prices = (prices / prices.iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=prices,
            mode="lines",
            name=ticker,
            line={"color": colors[i % len(colors)], "width": 2}
        ))

    fig.update_layout(
        title=title or f"Comparison: {', '.join(tickers)}",
        xaxis_title="Date",
        yaxis_title="Normalized Price (100 = Start)" if normalize else "Price",
        hovermode="x unified",
        **LAYOUT_TEMPLATE
    )

    return fig


def create_relative_strength_chart(
    ticker: str,
    benchmark: str = "SPY",
    period: str = "1y",
    title: str = None
) -> go.Figure:
    """
    상대강도 차트 (vs 벤치마크)

    Args:
        ticker: 종목 심볼
        benchmark: 벤치마크 심볼
        period: 기간
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    df_ticker = _get_ohlcv(ticker, period)
    df_bench = _get_ohlcv(benchmark, period)

    if df_ticker.empty or df_bench.empty:
        return _create_error_chart(f"No data for {ticker} or {benchmark}")

    # 날짜 맞추기
    merged = pd.merge(
        df_ticker[["Date", "Close"]].rename(columns={"Close": "ticker"}),
        df_bench[["Date", "Close"]].rename(columns={"Close": "benchmark"}),
        on="Date"
    )

    # 상대강도 계산
    merged["relative_strength"] = (merged["ticker"] / merged["benchmark"]) * 100
    merged["rs_normalized"] = (merged["relative_strength"] / merged["relative_strength"].iloc[0]) * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=[f"{ticker} vs {benchmark}", "Relative Strength"]
    )

    # 가격 비교 (정규화)
    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=(merged["ticker"] / merged["ticker"].iloc[0]) * 100,
            mode="lines", name=ticker,
            line={"color": COLORS["primary"]}
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=(merged["benchmark"] / merged["benchmark"].iloc[0]) * 100,
            mode="lines", name=benchmark,
            line={"color": COLORS["neutral"]}
        ),
        row=1, col=1
    )

    # 상대강도
    colors = [COLORS["positive"] if v >= 100 else COLORS["negative"]
              for v in merged["rs_normalized"]]
    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["rs_normalized"],
            mode="lines", name="RS",
            line={"color": COLORS["secondary"]}
        ),
        row=2, col=1
    )
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS["neutral"], row=2, col=1)

    fig.update_layout(
        title=title or f"{ticker} Relative Strength vs {benchmark}",
        **LAYOUT_TEMPLATE
    )

    return fig


# ============================================================
# 수익률 분포
# ============================================================

def create_returns_distribution(
    ticker: str,
    period: str = "1y",
    title: str = None
) -> go.Figure:
    """
    수익률 분포 히스토그램

    Args:
        ticker: 종목 심볼
        period: 기간
        title: 차트 제목

    Returns:
        Plotly Figure 객체
    """
    df = _get_ohlcv(ticker, period)
    if df.empty:
        return _create_error_chart(f"No data for {ticker}")

    returns = df["Close"].pct_change().dropna() * 100

    # 통계
    mean_ret = returns.mean()
    std_ret = returns.std()
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Daily Returns",
        marker_color=COLORS["primary"],
        opacity=0.7
    ))

    # 평균선
    fig.add_vline(x=mean_ret, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text=f"Mean: {mean_ret:.2f}%")

    # VaR (5%)
    var_5 = returns.quantile(0.05)
    fig.add_vline(x=var_5, line_dash="dot", line_color=COLORS["negative"],
                  annotation_text=f"VaR 5%: {var_5:.2f}%")

    fig.update_layout(
        title=title or f"{ticker} Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        annotations=[
            dict(
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                text=f"Mean: {mean_ret:.2f}%<br>Std: {std_ret:.2f}%<br>Skew: {skew:.2f}<br>Kurt: {kurtosis:.2f}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS["grid"]
            )
        ],
        **LAYOUT_TEMPLATE
    )

    return fig


# ============================================================
# 차트 저장 및 유틸리티
# ============================================================

def save_chart(
    fig: go.Figure,
    filename: str,
    format: str = "html",
    width: int = 1200,
    height: int = 800
) -> str:
    """
    차트 저장

    Args:
        fig: Plotly Figure
        filename: 파일명 (확장자 없이)
        format: 'html', 'png', 'svg', 'pdf'
        width: 이미지 너비
        height: 이미지 높이

    Returns:
        저장된 파일 경로
    """
    filepath = os.path.join(CHARTS_DIR, f"{safe_filename(filename)}.{format}")

    if format == "html":
        fig.write_html(filepath, include_plotlyjs=True)
    else:
        fig.write_image(filepath, width=width, height=height)

    return filepath


def _create_error_chart(message: str) -> go.Figure:
    """에러 메시지 차트"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color=COLORS["negative"])
    )
    fig.update_layout(**LAYOUT_TEMPLATE)
    return fig


def chart_to_html(fig: go.Figure, full_html: bool = False) -> str:
    """차트를 HTML로 변환"""
    return fig.to_html(full_html=full_html, include_plotlyjs="cdn")


# ============================================================
# 종합 대시보드
# ============================================================

def create_stock_dashboard(
    ticker: str,
    period: str = "6mo"
) -> Dict[str, go.Figure]:
    """
    종목 종합 대시보드

    Args:
        ticker: 종목 심볼
        period: 기간

    Returns:
        차트 딕셔너리
    """
    return {
        "candlestick": create_candlestick_chart(ticker, period, show_ma=[20, 50]),
        "technical": create_technical_chart(ticker, period, ["rsi", "macd"]),
        "returns": create_returns_distribution(ticker, period),
        "relative_strength": create_relative_strength_chart(ticker, "SPY", period)
    }


