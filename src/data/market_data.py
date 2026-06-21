"""Public market-data helpers for the Streamlit app.

The live-data layer intentionally uses more than one free public source. Yahoo
Finance access through yfinance can intermittently fail on hosted platforms, so
ETF-style symbols use Stooq first when possible, with yfinance used as a fallback
or for Yahoo-only symbols.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Mapping

import pandas as pd
import requests

MODULE_VERSION = "2026-06-21-stooq-first"


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
    "GOVT": MarketAsset("GOVT", "Broad US Treasury bond ETF proxy", "bonds"),
    "AGG": MarketAsset("AGG", "US aggregate bond ETF proxy", "bonds"),
    "BND": MarketAsset("BND", "Total US bond market ETF proxy", "bonds"),
    "GLD": MarketAsset("GLD", "Gold ETF proxy", "gold"),
    "IAU": MarketAsset("IAU", "Gold ETF proxy", "gold"),
    "FXE": MarketAsset("FXE", "Euro currency ETF proxy", "currency"),
    "FXY": MarketAsset("FXY", "Japanese yen currency ETF proxy", "currency"),
    "FXB": MarketAsset("FXB", "British pound currency ETF proxy", "currency"),
    "FXC": MarketAsset("FXC", "Canadian dollar currency ETF proxy", "currency"),
    "FXA": MarketAsset("FXA", "Australian dollar currency ETF proxy", "currency"),
    "UUP": MarketAsset("UUP", "US dollar index ETF proxy", "currency basket"),
    "SPY": MarketAsset("SPY", "S&P 500 ETF proxy; included mainly for stress comparison", "equity"),
    "ACWX": MarketAsset("ACWX", "Global ex-US equity ETF proxy; not a reserve-safe asset", "equity"),
}


class MarketDataError(RuntimeError):
    """Raised when no live market-data provider returns a usable panel."""


def asset_roles() -> dict[str, str]:
    """Return a user-facing description for each built-in market proxy."""
    return {asset: details.role for asset, details in ASSET_CATALOG.items()}


def sanitize_ticker(symbol: str) -> str:
    """Normalize one symbol while preserving suffixes like '=X' and '=F'."""
    return symbol.strip().upper()


def parse_ticker_text(text: str) -> list[str]:
    """Parse comma/space/newline-separated ticker text into unique symbols."""
    if not text:
        return []
    raw_parts = text.replace("\n", ",").replace(";", ",").replace(" ", ",").split(",")
    tickers: list[str] = []
    for part in raw_parts:
        ticker = sanitize_ticker(part)
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def ticker_map(assets: list[str]) -> dict[str, str]:
    """Map selected app asset labels to provider symbols."""
    mapping: dict[str, str] = {}
    for asset in assets:
        clean_asset = sanitize_ticker(asset)
        mapping[clean_asset] = ASSET_CATALOG[clean_asset].ticker if clean_asset in ASSET_CATALOG else clean_asset
    return mapping


def _standardize_price_panel(prices: pd.DataFrame, min_assets: int = 2) -> pd.DataFrame:
    """Sort, align, and validate a close-price panel."""
    if prices.empty:
        raise MarketDataError("No price data returned by the selected provider.")
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    if prices.shape[1] < min_assets:
        raise MarketDataError(
            "Fewer than two selected assets have usable market data after cleaning. "
            "Try liquid ETF-style symbols such as BIL, SHY, IEF, TLT, GLD, SPY, or AGG."
        )
    if len(prices) < 60:
        raise MarketDataError("Not enough observations returned. Use a longer date range.")
    return prices


def _extract_close_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract a close-price panel from yfinance output."""
    if raw.empty:
        raise MarketDataError("No market data returned from Yahoo Finance.")
    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        if "Close" in level_zero:
            close = raw["Close"].copy()
        elif "Adj Close" in level_zero:
            close = raw["Adj Close"].copy()
        else:
            raise MarketDataError("Could not find Close or Adj Close columns in Yahoo Finance data.")
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
    """Download and clean close-price data from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MarketDataError("Install yfinance to use Yahoo Finance market data.") from exc

    tickers = list(asset_tickers.values())
    try:
        raw = yf.download(
            tickers=tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            timeout=15,
        )
        close = _extract_close_panel(raw, tickers)
    except Exception as exc:  # pragma: no cover - upstream/network dependent
        raise MarketDataError(f"Yahoo Finance download failed: {exc}") from exc

    reverse_map = {ticker: asset for asset, ticker in asset_tickers.items()}
    prices = close.rename(columns=reverse_map)
    selected_columns = [asset for asset in asset_tickers if asset in prices.columns]
    prices = prices.loc[:, selected_columns]
    return _standardize_price_panel(prices)


def yahoo_to_stooq_symbol(yahoo_symbol: str) -> str | None:
    """Convert simple US ETF/equity Yahoo symbols to Stooq symbols."""
    symbol = sanitize_ticker(yahoo_symbol)
    if not symbol or any(marker in symbol for marker in ("=", "^", "/")):
        return None
    if not symbol.replace("-", "").replace(".", "").isalnum():
        return None
    return f"{symbol.lower()}.us"


def _stooq_url(stooq_symbol: str, start: date | str, end: date | str) -> str:
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")
    return f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={start_str}&d2={end_str}&i=d"


def fetch_stooq_prices(asset_tickers: Mapping[str, str], start: date | str, end: date | str) -> pd.DataFrame:
    """Download close prices from Stooq for simple ETF/equity symbols."""
    series_by_asset: dict[str, pd.Series] = {}
    errors: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 fx-reserve-rl-optimizer"}

    for asset, yahoo_symbol in asset_tickers.items():
        stooq_symbol = yahoo_to_stooq_symbol(yahoo_symbol)
        if stooq_symbol is None:
            errors.append(f"{asset}: no Stooq mapping for {yahoo_symbol}")
            continue
        url = _stooq_url(stooq_symbol, start, end)
        try:
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
                errors.append(f"{asset}: Stooq returned no close-price data")
                continue
            frame["Date"] = pd.to_datetime(frame["Date"])
            close = pd.Series(frame["Close"].astype(float).to_numpy(), index=frame["Date"], name=asset)
            if close.dropna().empty:
                errors.append(f"{asset}: Stooq close prices are empty")
                continue
            series_by_asset[asset] = close
        except Exception as exc:  # pragma: no cover - upstream/network dependent
            errors.append(f"{asset}: {exc}")
            continue

    if len(series_by_asset) < 2:
        detail = "; ".join(errors[:8])
        raise MarketDataError(
            "Stooq could not build a usable price panel. "
            "Use ETF-style tickers such as BIL, SHY, IEF, TLT, GLD, SPY, AGG. "
            f"Details: {detail}"
        )
    prices = pd.concat(series_by_asset.values(), axis=1)
    return _standardize_price_panel(prices)


def _all_symbols_have_stooq_mapping(asset_tickers: Mapping[str, str]) -> bool:
    return all(yahoo_to_stooq_symbol(symbol) is not None for symbol in asset_tickers.values())


def fetch_prices_with_fallback(asset_tickers: Mapping[str, str], start: date | str, end: date | str) -> pd.DataFrame:
    """Try Stooq first for ETF-style symbols, then yfinance as fallback."""
    errors: list[str] = []

    # For the built-in reserve ETF universe, Stooq is usually more reliable on
    # hosted Streamlit deployments than Yahoo/yfinance, so use it first.
    if _all_symbols_have_stooq_mapping(asset_tickers):
        try:
            return fetch_stooq_prices(asset_tickers, start=start, end=end)
        except Exception as exc:
            errors.append(f"Stooq failed: {exc}")

    try:
        return fetch_yfinance_prices(asset_tickers, start=start, end=end)
    except Exception as exc:
        errors.append(f"Yahoo/yfinance failed: {exc}")

    # If only some symbols map to Stooq, retry Stooq as a partial fallback. This
    # can still produce a usable panel when custom Yahoo-only symbols fail.
    try:
        return fetch_stooq_prices(asset_tickers, start=start, end=end)
    except Exception as exc:
        errors.append(f"Stooq fallback failed: {exc}")

    raise MarketDataError("Live data could not be loaded. " + " | ".join(errors))


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
    """Fetch a cleaned live price/return panel using free public sources."""
    prices = fetch_prices_with_fallback(ticker_map(assets), start=start, end=end)
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


__all__ = [
    "ASSET_CATALOG",
    "MarketAsset",
    "MarketDataError",
    "MODULE_VERSION",
    "asset_roles",
    "data_quality_summary",
    "fetch_market_dataset",
    "fetch_prices_with_fallback",
    "fetch_stooq_prices",
    "fetch_yfinance_prices",
    "parse_ticker_text",
    "prices_to_returns",
    "resample_prices",
    "sanitize_ticker",
    "ticker_map",
    "yahoo_to_stooq_symbol",
]
