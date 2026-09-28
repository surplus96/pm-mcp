from __future__ import annotations
from typing import List, Optional, Tuple
import os
import matplotlib
from mcp_server.config import PRESENT_MPL_STYLE, IMAGE_OUTPUT_DIR, safe_filename
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from mcp_server.tools.yf_utils import normalize_yf_columns

try:
    plt.style.use(PRESENT_MPL_STYLE)
except Exception:
    pass


def _ensure_colors(n: int, colors: Optional[List[str]]) -> List[str]:
    if colors and len(colors) >= n:
        return colors[:n]
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def _plot_ma(ax, series, windows: Tuple[int, ...], color: str):
    for w in windows:
        if w <= 1 or w >= len(series):
            continue
        ma = series.rolling(w).mean()
        ax.plot(series.index, ma, color=color, alpha=0.35, linewidth=1, label=f"MA{w}")


def render_price_chart(
    ticker: str,
    days: int = 90,
    out_dir: str = IMAGE_OUTPUT_DIR,
    color: Optional[str] = None,
    yscale: str = "linear",
    ma_windows: Tuple[int, ...] = (20, 50),
    figsize: Tuple[float, float] = (6, 3),
    grid_alpha: float = 0.3,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    hist = normalize_yf_columns(
        yf.download(ticker, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    )
    fig, ax = plt.subplots(figsize=figsize)
    if not hist.empty:
        ax.plot(hist.index, hist["Close"], label=ticker, color=color or "#1f77b4")
        if ma_windows:
            _plot_ma(ax, hist["Close"], ma_windows, color or "#1f77b4")
    ax.set_title(f"{ticker} - {days}D Close")
    ax.set_yscale(yscale if yscale in ("linear", "log") else "linear")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc='upper left')
    fname = safe_filename(f"{ticker}_{days}d.png")
    fpath = os.path.join(out_dir, fname)
    plt.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


def render_multi_price_chart(
    tickers: List[str],
    days: int = 90,
    out_dir: str = IMAGE_OUTPUT_DIR,
    colors: Optional[List[str]] = None,
    yscale: str = "linear",
    ma_windows: Tuple[int, ...] = (),
    figsize: Tuple[float, float] = (7, 3.5),
    grid_alpha: float = 0.3,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    cols = _ensure_colors(len(tickers), colors)
    for i, t in enumerate(tickers):
        hist = normalize_yf_columns(
            yf.download(t, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
        )
        c = cols[i]
        if not hist.empty:
            ax.plot(hist.index, hist["Close"], label=t, color=c)
            if ma_windows:
                _plot_ma(ax, hist["Close"], ma_windows, c)
    ax.set_title(f"{','.join(tickers)} - {days}D Close")
    ax.set_yscale(yscale if yscale in ("linear", "log") else "linear")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    fname = safe_filename(f"multi_{'_'.join(tickers)}_{days}d.png")
    fpath = os.path.join(out_dir, fname)
    plt.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath
