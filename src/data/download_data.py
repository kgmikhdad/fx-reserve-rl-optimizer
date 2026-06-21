"""Download public market proxy data.

The project uses public ETF/currency proxies for demonstration. This module is
optional for the Streamlit demo, which can run entirely on deterministic sample
data from src.data.demo_data.
"""
from __future__ import annotations

import pandas as pd


def download_adjusted_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install yfinance to download public market data.") from exc

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=-1)
    else:
        prices = data.to_frame(name=tickers[0]) if isinstance(data, pd.Series) else data
    prices = prices.dropna(how="all").ffill().dropna()
    return prices[tickers]


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    return np.log(prices / prices.shift(1)).dropna()
