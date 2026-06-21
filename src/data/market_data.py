"""Public market-data helpers for the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class MarketAsset:
    ticker: str
    role: str
    asset_class: str


ASSET_CATALOG: dict[str, MarketAsset] = {
    "BIL": MarketAsset("BIL", "US Treasury bills / cash proxy", "cash"),
    "SHY": MarketAsset("SHY", "1-3 year US Treasuries proxy", "short bonds"),
    "IEI": MarketAsset("IEI", "3-7 year US Treasuries proxy", "intermediate bonds"),
    "IEF": MarketAsset("IEF", "7-10 year US Treasuries proxy", "intermediate bonds"),
    "TLT": MarketAsset("TLT", "20+ year US Treasuries proxy", "long bonds"),
    "GLD": MarketAsset("GLD", "Gold proxy", "gold"),
    "FXE": MarketAsset("FXE", "Euro currency proxy", "currency"),
    "FXY": MarketAsset("FXY", "Japanese yen currency proxy", "currency"),
    "FXB": MarketAsset("FXB", "British pound currency proxy", "currency"),
    "FXC": MarketAsset("FXC", "Canadian dollar currency proxy", "currency"),
    "FXA": MarketAsset("FXA", "Australian dollar currency proxy", "currency"),
    "UUP": MarketAsset("UUP", "US dollar index ETF proxy", "currency basket"),
}


def asset_roles() -> dict[str, str]:
    return {asset: details.role for asset, details in ASSET_CATALOG.items()}


def ticker_map(assets: list[str]) -> dict[str, str]:
    missing = [asset for asset in assets if asset not in ASSET_CATALOG]
    if missing:
        raise ValueError(f"Unknown assets: {missing}")
    return {asset: ASSET_CATALOG[asset].ticker for asset in assets}


def _extract_close_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No market data returned.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in set(raw.columns.get_level_values(0)):
            close = raw["Close"].copy()
        elif "Adj Close" in set(raw.columns.get_level_values(0)):
            close = raw["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close or Adj Close columns.")
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
    elif "Adj Close" in raw.columns:
        close = raw[["Adj Close"]].copy()
    else:
        close = raw.copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if len(tickers) == 1 and list(close.columns) == ["Close"]:
        close.columns = tickers
    return close


def fetch_yfinance_prices(asset_tickers: Mapping[str, str], start: date | str, end: date | str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to use public market data.") from exc

    tickers = list(asset_tickers.values())
    raw = yf.download(tickers=tickers, start=str(start), end=str(end), auto_adjust=True, progress=False)
    close = _extract_close_panel(raw, tickers)
    prices = close.rename(columns={ticker: asset for asset, ticker in asset_tickers.items()})
    selected_columns = [asset for asset in asset_tickers if asset in prices.columns]
    prices = prices.loc[:, selected_columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().dropna(how="all").ffill().dropna(how="any")
    if prices.shape[1] < 2:
        raise ValueError("Fewer than two assets have usable market data.")
    if len(prices) < 60:
        raise ValueError("Not enough observations returned. Use a longer date range.")
    return prices


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Daily":
        return prices
    if frequency == "Weekly":
        return prices.resample("W-FRI").last().dropna(how="any")
    if frequency == "Monthly":
        return prices.resample("ME").last().dropna(how="any")
    raise ValueError(f"Unsupported frequency: {frequency}")


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def fetch_market_dataset(
    assets: list[str], start: date | str, end: date | str, frequency: str = "Daily"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = fetch_yfinance_prices(ticker_map(assets), start=start, end=end)
    prices = resample_prices(prices, frequency=frequency)
    returns = prices_to_returns(prices)
    prices = prices.loc[returns.index]
    return prices, returns


def data_quality_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    annualization = 252.0 if len(returns) >= 100 else 52.0 if len(returns) > 24 else 12.0
    rows = []
    for asset in prices.columns:
        rows.append(
            {
                "asset": asset,
                "first_date": prices[asset].first_valid_index().date(),
                "last_date": prices[asset].last_valid_index().date(),
                "observations": int(returns[asset].count()),
                "missing_prices": int(prices[asset].isna().sum()),
                "annualized_return": float(returns[asset].mean() * annualization),
                "annualized_volatility": float(returns[asset].std(ddof=0) * annualization**0.5),
                "latest_price": float(prices[asset].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).set_index("asset")
