"""Interactive Streamlit dashboard for the FX reserve RL optimizer.

This app is intentionally self-contained. Live data is fetched only after the app
has already loaded, and every provider returns diagnostics so Streamlit Cloud
failures are visible to the user instead of appearing as a generic crash.
"""
from __future__ import annotations

import sys
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.demo_data import ASSET_ROLES as DEMO_ASSET_ROLES  # noqa: E402
from src.data.demo_data import demo_dataset  # noqa: E402
from src.portfolio.baselines import (  # noqa: E402
    equal_weight,
    portfolio_returns,
    rolling_mean_variance,
    rolling_minimum_variance,
    static_reserve_weights,
)
from src.portfolio.metrics import drawdown_series, metrics_table, portfolio_turnover  # noqa: E402

DISCLAIMER = (
    "This is a public-data and synthetic-data research prototype. It is not investment advice, "
    "not a live reserve-management system, and does not represent the portfolio, policy, or internal "
    "systems of any central bank, the BIS, or any financial institution."
)

DEFAULT_TICKERS = "BIL, SHY, IEF, TLT, GLD"

PROXY_ROLES: dict[str, str] = {
    "BIL": "US Treasury bills / cash proxy",
    "SHY": "1-3 year US Treasuries proxy",
    "IEI": "3-7 year US Treasuries proxy",
    "IEF": "7-10 year US Treasuries proxy",
    "TLT": "20+ year US Treasuries proxy",
    "GOVT": "Broad US Treasury bond ETF proxy",
    "AGG": "US aggregate bond ETF proxy",
    "BND": "Total US bond market ETF proxy",
    "GLD": "Gold ETF proxy",
    "IAU": "Gold ETF proxy",
    "FXE": "Euro currency ETF proxy",
    "FXY": "Japanese yen currency ETF proxy",
    "FXB": "British pound currency ETF proxy",
    "FXC": "Canadian dollar currency ETF proxy",
    "FXA": "Australian dollar currency ETF proxy",
    "UUP": "US dollar index ETF proxy",
    "SPY": "S&P 500 ETF proxy; included mainly for stress comparison",
    "ACWX": "Global ex-US equity ETF proxy; not a reserve-safe asset",
}

STRESS_LIBRARY: dict[str, dict[str, float]] = {
    "Global bond selloff": {
        "BIL": 0.0005,
        "SHY": -0.002,
        "IEI": -0.006,
        "IEF": -0.010,
        "TLT": -0.030,
        "GOVT": -0.010,
        "AGG": -0.012,
        "BND": -0.012,
        "GLD": 0.006,
        "IAU": 0.006,
        "FXE": -0.004,
        "FXY": 0.003,
        "FXB": -0.006,
        "FXC": -0.006,
        "FXA": -0.008,
        "UUP": 0.006,
        "SPY": -0.025,
        "ACWX": -0.030,
    },
    "USD liquidity squeeze": {
        "BIL": 0.0005,
        "SHY": 0.001,
        "IEI": 0.0015,
        "IEF": 0.002,
        "TLT": 0.004,
        "GOVT": 0.002,
        "AGG": 0.000,
        "BND": 0.000,
        "GLD": -0.006,
        "IAU": -0.006,
        "FXE": -0.018,
        "FXY": -0.012,
        "FXB": -0.016,
        "FXC": -0.014,
        "FXA": -0.020,
        "UUP": 0.020,
        "SPY": -0.030,
        "ACWX": -0.035,
    },
    "Gold selloff": {
        "BIL": 0.0000,
        "SHY": 0.000,
        "IEI": 0.001,
        "IEF": 0.001,
        "TLT": 0.002,
        "GOVT": 0.001,
        "AGG": 0.001,
        "BND": 0.001,
        "GLD": -0.050,
        "IAU": -0.050,
        "FXE": -0.002,
        "FXY": -0.002,
        "FXB": -0.002,
        "FXC": -0.002,
        "FXA": -0.002,
        "UUP": 0.002,
        "SPY": 0.000,
        "ACWX": 0.000,
    },
    "Broad risk-off shock": {
        "BIL": 0.0005,
        "SHY": 0.002,
        "IEI": 0.003,
        "IEF": 0.004,
        "TLT": 0.008,
        "GOVT": 0.003,
        "AGG": 0.000,
        "BND": 0.000,
        "GLD": 0.010,
        "IAU": 0.010,
        "FXE": -0.012,
        "FXY": 0.006,
        "FXB": -0.010,
        "FXC": -0.012,
        "FXA": -0.016,
        "UUP": 0.010,
        "SPY": -0.040,
        "ACWX": -0.045,
    },
    "Foreign-currency depreciation": {
        "BIL": 0.0000,
        "SHY": 0.0000,
        "IEI": 0.0000,
        "IEF": 0.0000,
        "TLT": 0.0000,
        "GOVT": 0.0000,
        "AGG": 0.0000,
        "BND": 0.0000,
        "GLD": 0.0000,
        "IAU": 0.0000,
        "FXE": -0.025,
        "FXY": -0.025,
        "FXB": -0.025,
        "FXC": -0.025,
        "FXA": -0.025,
        "UUP": 0.015,
        "SPY": 0.0000,
        "ACWX": 0.0000,
    },
}

st.set_page_config(page_title="FX Reserve Portfolio Optimizer", layout="wide")


def parse_tickers(text: str) -> list[str]:
    """Parse comma/space/newline-separated ticker text into unique uppercase symbols."""
    if not text:
        return []
    raw_parts = text.replace("\n", ",").replace(";", ",").replace(" ", ",").split(",")
    tickers: list[str] = []
    for part in raw_parts:
        ticker = part.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return demo_dataset()


def standardize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean, align, and validate a price panel."""
    if prices.empty:
        raise ValueError("No price data was returned.")
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if prices.shape[1] < 2:
        raise ValueError("At least two usable assets are required after cleaning.")
    if len(prices) < 60:
        raise ValueError("At least 60 price observations are required. Use a longer date range.")
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


def stooq_symbol(ticker: str) -> str | None:
    """Map simple US ETF/equity symbols to Stooq symbols."""
    symbol = ticker.strip().upper()
    if not symbol or any(marker in symbol for marker in ("=", "^", "/")):
        return None
    if not symbol.replace("-", "").replace(".", "").isalnum():
        return None
    return f"{symbol.lower()}.us"


@st.cache_data(ttl=3600, show_spinner="Fetching Stooq data...")
def fetch_stooq_panel(tickers: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch ETF/equity prices from Stooq and return a diagnostics table."""
    headers = {"User-Agent": "Mozilla/5.0 fx-reserve-rl-optimizer"}
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")
    series: dict[str, pd.Series] = {}
    diagnostics: list[dict[str, str | int]] = []

    for ticker in tickers:
        mapped = stooq_symbol(ticker)
        if mapped is None:
            diagnostics.append({"ticker": ticker, "provider_symbol": "", "status": "skipped", "detail": "Not a simple Stooq ETF/equity symbol", "rows": 0})
            continue
        url = f"https://stooq.com/q/d/l/?s={mapped}&d1={start_str}&d2={end_str}&i=d"
        try:
            response = requests.get(url, timeout=20, headers=headers)
            status = response.status_code
            if status != 200:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "detail": f"HTTP {status}", "rows": 0})
                continue
            frame = pd.read_csv(StringIO(response.text))
            if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "detail": "No Date/Close columns returned", "rows": 0})
                continue
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(), index=frame["Date"], name=ticker).dropna()
            if close.empty:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "detail": "Empty close-price series", "rows": 0})
                continue
            series[ticker] = close
            diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "ok", "detail": "Fetched close prices", "rows": len(close)})
        except Exception as exc:  # pragma: no cover - network dependent
            diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "detail": str(exc), "rows": 0})

    diag = pd.DataFrame(diagnostics)
    if len(series) < 2:
        raise ValueError("Stooq returned fewer than two usable series. See diagnostics below."), diag
    prices = standardize_prices(pd.concat(series.values(), axis=1))
    return prices, diag


@st.cache_data(ttl=3600, show_spinner="Fetching Yahoo Finance data through yfinance...")
def fetch_yfinance_panel(tickers: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices from Yahoo Finance through yfinance and return diagnostics."""
    diagnostics: list[dict[str, str | int]] = []
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency
        diag = pd.DataFrame([{"ticker": "ALL", "provider_symbol": "yfinance", "status": "failed", "detail": f"yfinance import failed: {exc}", "rows": 0}])
        raise RuntimeError("yfinance is not available in this deployment."), diag

    try:
        raw = yf.download(list(tickers), start=str(start), end=str(end), auto_adjust=True, progress=False, threads=False, group_by="column", timeout=20)
        if raw.empty:
            diag = pd.DataFrame([{"ticker": "ALL", "provider_symbol": "Yahoo", "status": "failed", "detail": "Empty dataframe returned", "rows": 0}])
            raise RuntimeError("Yahoo/yfinance returned an empty dataframe."), diag
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"].copy()
            elif "Adj Close" in raw.columns.get_level_values(0):
                close = raw["Adj Close"].copy()
            else:
                close = raw.copy()
        elif "Close" in raw.columns:
            close = raw[["Close"]].copy()
            if len(tickers) == 1:
                close.columns = [tickers[0]]
        elif "Adj Close" in raw.columns:
            close = raw[["Adj Close"]].copy()
            if len(tickers) == 1:
                close.columns = [tickers[0]]
        else:
            close = raw.copy()
        close = close[[col for col in close.columns if str(col).upper() in tickers]]
        close.columns = [str(col).upper() for col in close.columns]
        for ticker in tickers:
            if ticker in close.columns and close[ticker].dropna().shape[0] > 0:
                diagnostics.append({"ticker": ticker, "provider_symbol": ticker, "status": "ok", "detail": "Fetched close prices", "rows": int(close[ticker].dropna().shape[0])})
            else:
                diagnostics.append({"ticker": ticker, "provider_symbol": ticker, "status": "failed", "detail": "No usable close prices", "rows": 0})
        prices = standardize_prices(close)
        return prices, pd.DataFrame(diagnostics)
    except Exception as exc:  # pragma: no cover - network/upstream dependent
        diag = pd.DataFrame(diagnostics or [{"ticker": "ALL", "provider_symbol": "Yahoo", "status": "failed", "detail": str(exc), "rows": 0}])
        raise RuntimeError(f"Yahoo/yfinance failed: {exc}"), diag


def parse_uploaded_price_csv(uploaded_file) -> pd.DataFrame:
    """Read a user-uploaded price CSV.

    Accepted formats:
    - Date column plus one column per asset
    - First column as date index plus one column per asset
    """
    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        raise ValueError("Uploaded CSV is empty.")
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col)
    frame.columns = [str(col).strip().upper() for col in frame.columns]
    return standardize_prices(frame)


def make_price_return_panel(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = resample_prices(standardize_prices(prices), frequency)
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


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, legend_title=None)
    return fig


def drawdown_chart(strategy_returns: pd.DataFrame) -> go.Figure:
    dd = pd.DataFrame({col: drawdown_series(strategy_returns[col]) for col in strategy_returns.columns})
    return line_chart(dd, "Drawdown comparison", "Drawdown")


def weight_area_chart(weights: pd.DataFrame, title: str) -> go.Figure:
    fig = px.area(weights, x=weights.index, y=weights.columns, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio weight", legend_title=None)
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    return px.imshow(corr, text_auto=".2f", aspect="auto", title="Asset return correlation matrix")


def normalise_weights(raw_weights: dict[str, float], assets: list[str]) -> pd.Series:
    values = np.array([raw_weights.get(asset, 0.0) for asset in assets], dtype="float64")
    values = np.maximum(values, 0.0)
    values = np.ones(len(assets)) / len(assets) if np.isclose(values.sum(), 0.0) else values / values.sum()
    return pd.Series(values, index=assets)


def constant_weight_frame(weights: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.tile(weights.to_numpy(), (len(index), 1)), index=index, columns=weights.index)


def apply_rebalance_schedule(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Every observation":
        return weights
    rule = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE"}[frequency]
    scheduled = weights.resample(rule).last().reindex(weights.index, method="ffill")
    return scheduled.fillna(weights.iloc[0])


def build_strategy_set(
    returns: pd.DataFrame,
    transaction_cost_bps: float,
    rolling_window: int,
    max_weight: float,
    risk_aversion: float,
    custom_weights: pd.Series,
    rebalance_frequency: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_strategies: dict[str, pd.DataFrame] = {
        "Equal Weight": equal_weight(returns),
        "Static Reserve": static_reserve_weights(returns),
        "Minimum Variance": rolling_minimum_variance(returns, window=rolling_window, max_weight=max_weight),
        "Mean Variance": rolling_mean_variance(returns, window=rolling_window, risk_aversion=risk_aversion, max_weight=max_weight),
        "Your Custom Portfolio": constant_weight_frame(custom_weights, returns.index),
    }
    strategies = {name: apply_rebalance_schedule(weights, rebalance_frequency) for name, weights in raw_strategies.items()}
    strategy_returns = pd.DataFrame(
        {name: portfolio_returns(returns, weights, transaction_cost_bps=transaction_cost_bps) for name, weights in strategies.items()},
        index=returns.index,
    )
    return strategy_returns, strategies


def annualized_turnover(weights: pd.DataFrame) -> float:
    turnover = portfolio_turnover(weights)
    return 0.0 if turnover.empty else float(turnover.mean() * 252.0)


def add_turnover_and_score(
    metrics: pd.DataFrame,
    weights_by_strategy: dict[str, pd.DataFrame],
    lambda_vol: float,
    lambda_drawdown: float,
    lambda_turnover: float,
) -> pd.DataFrame:
    enriched = metrics.copy()
    enriched["annualized_turnover"] = [annualized_turnover(weights_by_strategy[name]) for name in enriched.index]
    enriched["reserve_utility_score"] = (
        enriched["annualized_return"]
        - lambda_vol * enriched["annualized_volatility"]
        - lambda_drawdown * enriched["max_drawdown"].abs()
        - lambda_turnover * enriched["annualized_turnover"]
    )
    return enriched.sort_values("reserve_utility_score", ascending=False)


def stress_vector(name: str, assets: list[str], custom_shocks_pct: dict[str, float]) -> pd.Series:
    if name == "Custom one-day shock":
        return pd.Series({asset: custom_shocks_pct.get(asset, 0.0) / 100.0 for asset in assets})
    template = STRESS_LIBRARY[name]
    return pd.Series({asset: template.get(asset, 0.0) for asset in assets})


def stress_test_table(weights_by_strategy: dict[str, pd.DataFrame], shock: pd.Series) -> pd.DataFrame:
    rows = []
    for strategy_name, weights in weights_by_strategy.items():
        last_weights = weights.iloc[-1].reindex(shock.index).fillna(0.0)
        stress_return = float((last_weights * shock).sum())
        rows.append({"strategy": strategy_name, "one_day_stress_return": stress_return, "one_day_stress_loss": -stress_return})
    return pd.DataFrame(rows).set_index("strategy").sort_values("one_day_stress_loss")


def normalize_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.divide(prices.iloc[0]).multiply(100.0)


def as_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


st.title("FX Reserve Portfolio Optimizer")
st.caption("Interactive reserve-allocation simulator, live-data dashboard, and DRL project prototype")
st.warning(DISCLAIMER)

st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Choose market-data mode",
    ["Demo data", "Upload price CSV", "Live Stooq ETF data", "Live Yahoo/yfinance data"],
    help=(
        "Use Demo data to verify the app. Use Upload CSV for guaranteed user-controlled data. "
        "Use Stooq first for ETF-style symbols such as BIL, SHY, IEF, TLT, GLD, SPY, AGG. "
        "Yahoo/yfinance is experimental on Streamlit Cloud."
    ),
)
frequency = st.sidebar.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"], index=0)
provider_diagnostics = pd.DataFrame()

try:
    if data_source == "Demo data":
        prices_all, returns_all = load_demo_data()
        asset_options = list(returns_all.columns)
        selected_assets = st.sidebar.multiselect("Assets included in the reserve universe", asset_options, default=asset_options)
        if len(selected_assets) < 2:
            st.sidebar.error("Select at least two assets to run the simulator.")
            st.stop()
        min_date = returns_all.index.min().date()
        max_date = returns_all.index.max().date()
        start_date, end_date = st.sidebar.slider("Backtest date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        prices = prices_all.loc[str(start_date) : str(end_date), selected_assets]
        returns = returns_all.loc[str(start_date) : str(end_date), selected_assets]
        role_lookup = DEMO_ASSET_ROLES | PROXY_ROLES
        source_note = "Synthetic deterministic demo data generated inside the app."

    elif data_source == "Upload price CSV":
        uploaded_file = st.sidebar.file_uploader("Upload price CSV", type=["csv"])
        st.sidebar.caption("CSV format: Date column + one numeric price column per asset.")
        if uploaded_file is None:
            st.info("Upload a price CSV to run the simulator with your own data. Until then, switch to Demo data.")
            st.stop()
        uploaded_prices = parse_uploaded_price_csv(uploaded_file)
        min_date = uploaded_prices.index.min().date()
        max_date = uploaded_prices.index.max().date()
        start_date, end_date = st.sidebar.slider("Backtest date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        selected_assets = st.sidebar.multiselect("Assets included in the reserve universe", list(uploaded_prices.columns), default=list(uploaded_prices.columns))
        if len(selected_assets) < 2:
            st.sidebar.error("Select at least two assets to run the simulator.")
            st.stop()
        prices, returns = make_price_return_panel(uploaded_prices.loc[str(start_date) : str(end_date), selected_assets], frequency)
        role_lookup = {asset: "Uploaded user price series" for asset in selected_assets}
        source_note = "User-uploaded price CSV. This is the most reliable route if hosted live providers fail."

    else:
        ticker_text = st.sidebar.text_area(
            "Ticker symbols",
            value=DEFAULT_TICKERS,
            help="For Stooq, use simple ETF/equity symbols: BIL, SHY, IEF, TLT, GLD, SPY, AGG. For Yahoo, symbols like EURUSD=X may work but are less reliable on Streamlit Cloud.",
        )
        requested_tickers = parse_tickers(ticker_text)
        if len(requested_tickers) < 2:
            st.sidebar.error("Enter at least two ticker symbols.")
            st.stop()
        live_col1, live_col2 = st.sidebar.columns(2)
        with live_col1:
            start_date = st.date_input("Start", value=date(2018, 1, 1), max_value=date.today())
        with live_col2:
            end_date = st.date_input("End", value=date.today(), max_value=date.today())
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            st.stop()
        if data_source == "Live Stooq ETF data":
            prices_raw, provider_diagnostics = fetch_stooq_panel(tuple(requested_tickers), start_date, end_date)
            provider_name = "Stooq"
        else:
            prices_raw, provider_diagnostics = fetch_yfinance_panel(tuple(requested_tickers), start_date, end_date)
            provider_name = "Yahoo/yfinance"
        prices, returns = make_price_return_panel(prices_raw, frequency)
        selected_assets = list(returns.columns)
        role_lookup = {asset: PROXY_ROLES.get(asset, f"{provider_name} market symbol") for asset in selected_assets}
        source_note = f"Live public market data from {provider_name}. See provider diagnostics below for per-symbol status."
except Exception as exc:
    st.error("Data loading failed. The app itself is running; the selected data source did not return a usable price panel.")
    st.exception(exc)
    if not provider_diagnostics.empty:
        st.subheader("Provider diagnostics")
        st.dataframe(provider_diagnostics, width="stretch")
    st.info(
        "Fast workaround: switch to Demo data, or use Upload price CSV. For live Stooq, start with exactly: BIL, SHY, IEF, TLT, GLD and Start = 2018-01-01."
    )
    st.stop()

st.sidebar.header("Simulation controls")
transaction_cost_bps = st.sidebar.slider("Transaction cost per unit of turnover, in basis points", 0.0, 25.0, 1.0, 0.5)
rolling_window = st.sidebar.slider("Rolling estimation window, observations", 12, 252, 63, 1)
max_weight = st.sidebar.slider("Maximum weight per asset for optimized baselines", 0.20, 1.00, 0.60, 0.05)
risk_aversion = st.sidebar.slider("Mean-variance risk aversion", 1.0, 50.0, 10.0, 1.0)
rebalance_frequency = st.sidebar.selectbox("Portfolio rebalancing frequency", ["Every observation", "Weekly", "Monthly", "Quarterly"], index=0)

st.sidebar.header("Reserve utility preferences")
lambda_vol = st.sidebar.slider("Volatility penalty", 0.0, 5.0, 1.0, 0.1)
lambda_drawdown = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lambda_turnover = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

if len(returns) < rolling_window + 5:
    st.error("The selected date range is too short for the chosen rolling window.")
    st.stop()

st.sidebar.header("Custom reserve portfolio")
st.sidebar.caption("Set raw weights. The app normalises them to sum to 100%.")
raw_custom_weights = {
    asset: st.sidebar.slider(f"{asset} raw weight", min_value=0.0, max_value=100.0, value=round(100.0 / len(selected_assets), 1), step=1.0)
    for asset in selected_assets
}
custom_weights = normalise_weights(raw_custom_weights, selected_assets)

strategy_returns, weights_by_strategy = build_strategy_set(
    returns=returns,
    transaction_cost_bps=transaction_cost_bps,
    rolling_window=rolling_window,
    max_weight=max_weight,
    risk_aversion=risk_aversion,
    custom_weights=custom_weights,
    rebalance_frequency=rebalance_frequency,
)
metrics = metrics_table(strategy_returns)
metrics_scored = add_turnover_and_score(metrics, weights_by_strategy, lambda_vol, lambda_drawdown, lambda_turnover)
equity = (1.0 + strategy_returns).cumprod()

selected_strategy = st.sidebar.selectbox(
    "Strategy to inspect",
    list(strategy_returns.columns),
    index=list(strategy_returns.columns).index("Static Reserve") if "Static Reserve" in strategy_returns.columns else 0,
)
stress_name = st.sidebar.selectbox("One-day stress scenario", list(STRESS_LIBRARY.keys()) + ["Custom one-day shock"])
custom_shocks_pct: dict[str, float] = {}
if stress_name == "Custom one-day shock":
    st.sidebar.caption("Custom shock is in percent return for one day.")
    for asset in selected_assets:
        custom_shocks_pct[asset] = st.sidebar.slider(f"{asset} shock (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.25)
shock = stress_vector(stress_name, selected_assets, custom_shocks_pct)
stress_results = stress_test_table(weights_by_strategy, shock)

st.header("What this project actually does")
st.markdown(
    """
This app is a **reserve-portfolio allocation simulator**. It lets you change the portfolio universe,
data source, market-data frequency, backtest window, transaction costs, optimization constraints,
rebalancing frequency, risk preferences, custom weights, and stress assumptions. Then it compares
several allocation rules: equal weight, static reserve, minimum variance, mean variance, and your custom portfolio.

The deep reinforcement learning part is implemented in the repository as a Gymnasium environment and PPO training script.
The dashboard is the inspection layer: it shows how benchmark and future trained DRL strategies should be judged using
return, volatility, drawdown, turnover, stress loss, and a reserve-utility score.
"""
)

input_col1, input_col2, input_col3, input_col4, input_col5 = st.columns(5)
input_col1.metric("Data mode", data_source)
input_col2.metric("Assets selected", len(selected_assets))
input_col3.metric("Observations", len(returns))
input_col4.metric("Transaction cost", f"{transaction_cost_bps:.1f} bps")
input_col5.metric("Best utility strategy", metrics_scored.index[0])
st.info(source_note)

if not provider_diagnostics.empty:
    with st.expander("Live provider diagnostics", expanded=True):
        st.dataframe(provider_diagnostics, width="stretch")

st.header("1. Asset universe and market data")
asset_table = pd.DataFrame(
    {
        "asset": selected_assets,
        "proxy role": [role_lookup.get(asset, "User-selected reserve proxy") for asset in selected_assets],
        "custom weight": [custom_weights.loc[asset] for asset in selected_assets],
        "stress shock": [shock.loc[asset] for asset in selected_assets],
    }
)
st.dataframe(asset_table.style.format({"custom weight": "{:.2%}", "stress shock": "{:.2%}"}), width="stretch")

with st.expander("Data quality and descriptive statistics", expanded=data_source != "Demo data"):
    quality = data_quality_summary(prices, returns)
    st.dataframe(
        quality.style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "latest_price": "{:.2f}"}),
        width="stretch",
    )

normalize_prices = st.checkbox("Normalize price chart to 100 at first observation", value=True)
plot_prices = normalize_price_panel(prices) if normalize_prices else prices
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(line_chart(plot_prices, "Selected asset price indices", "Index"), width="stretch")
with chart_col2:
    st.plotly_chart(correlation_heatmap(returns), width="stretch")

st.header("2. Strategy performance comparison")
st.plotly_chart(line_chart(equity, "Cumulative wealth after transaction costs", "Wealth index"), width="stretch")
metric_cols = st.columns(4)
selected_metrics = metrics_scored.loc[selected_strategy]
metric_cols[0].metric("Annualized return", f"{selected_metrics['annualized_return']:.2%}")
metric_cols[1].metric("Annualized volatility", f"{selected_metrics['annualized_volatility']:.2%}")
metric_cols[2].metric("Max drawdown", f"{selected_metrics['max_drawdown']:.2%}")
metric_cols[3].metric("Reserve utility score", f"{selected_metrics['reserve_utility_score']:.3f}")

st.subheader("Risk, return, turnover, and reserve-utility ranking")
st.dataframe(
    metrics_scored.style.format(
        {
            "annualized_return": "{:.2%}",
            "annualized_volatility": "{:.2%}",
            "sharpe_ratio": "{:.2f}",
            "sortino_ratio": "{:.2f}",
            "max_drawdown": "{:.2%}",
            "calmar_ratio": "{:.2f}",
            "var_95": "{:.2%}",
            "expected_shortfall_95": "{:.2%}",
            "annualized_turnover": "{:.2f}",
            "reserve_utility_score": "{:.3f}",
        }
    ),
    width="stretch",
)

download_col1, download_col2, download_col3, download_col4 = st.columns(4)
download_col1.download_button("Download strategy metrics CSV", as_csv_download(metrics_scored), "strategy_metrics.csv", "text/csv")
download_col2.download_button("Download strategy returns CSV", as_csv_download(strategy_returns), "strategy_returns.csv", "text/csv")
latest_weights_download = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
download_col3.download_button("Download latest weights CSV", as_csv_download(latest_weights_download), "latest_strategy_weights.csv", "text/csv")
download_col4.download_button("Download price data CSV", as_csv_download(prices), "price_data.csv", "text/csv")

st.header("3. Drawdown and allocation diagnostics")
st.plotly_chart(drawdown_chart(strategy_returns), width="stretch")
st.plotly_chart(weight_area_chart(weights_by_strategy[selected_strategy], f"{selected_strategy}: allocation over time"), width="stretch")

hist_col, turnover_col = st.columns(2)
with hist_col:
    st.subheader(f"{selected_strategy}: return distribution")
    fig = px.histogram(strategy_returns, x=selected_strategy, nbins=60)
    fig.update_layout(xaxis_title="Return", yaxis_title="Frequency")
    st.plotly_chart(fig, width="stretch")
with turnover_col:
    st.subheader(f"{selected_strategy}: turnover over time")
    selected_turnover = portfolio_turnover(weights_by_strategy[selected_strategy])
    turnover_fig = go.Figure()
    turnover_fig.add_trace(go.Scatter(x=selected_turnover.index, y=selected_turnover, mode="lines", name="Turnover"))
    turnover_fig.update_layout(xaxis_title="Date", yaxis_title="Turnover")
    st.plotly_chart(turnover_fig, width="stretch")

st.header("4. Stress test")
st.write("The stress test applies a one-day shock to the latest portfolio weights of each strategy.")
stress_col1, stress_col2 = st.columns(2)
with stress_col1:
    st.subheader("Shock vector")
    st.dataframe(shock.rename("one_day_asset_shock").to_frame().style.format("{:.2%}"), width="stretch")
with stress_col2:
    st.subheader("Strategy-level stress result")
    st.dataframe(stress_results.style.format({"one_day_stress_return": "{:.2%}", "one_day_stress_loss": "{:.2%}"}), width="stretch")
stress_fig = px.bar(stress_results.reset_index(), x="strategy", y="one_day_stress_loss", title=f"One-day stress loss under: {stress_name}")
st.plotly_chart(stress_fig, width="stretch")

st.header("5. DRL implementation status")
st.markdown(
    """
The repository already includes the technical machinery for the reinforcement-learning extension:

```text
src/envs/fx_reserve_env.py     -> Gymnasium portfolio-allocation environment
src/agents/train_ppo.py        -> PPO training entrypoint using Stable-Baselines3
src/portfolio/constraints.py   -> weight projection and concentration constraints
src/portfolio/metrics.py       -> risk and performance metrics
```

In the next stage, you train a PPO agent offline, export its test-period weights and returns, and add it
to this same dashboard as a sixth strategy. Training is intentionally not run inside Streamlit.
"""
)

with st.expander("Show raw calculated data"):
    st.subheader("Strategy returns")
    st.dataframe(strategy_returns, width="stretch")
    st.subheader("Latest strategy weights")
    st.dataframe(latest_weights_download.style.format("{:.2%}"), width="stretch")
    st.subheader("Market returns")
    st.dataframe(returns, width="stretch")
