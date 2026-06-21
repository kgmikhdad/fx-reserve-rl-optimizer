"""Streamlit dashboard for the FX reserve portfolio optimizer.

The app is intentionally conservative for Streamlit Cloud deployment:
- demo mode always works;
- CSV upload mode gives a guaranteed real-data path controlled by the user;
- Stooq and Yahoo/yfinance live modes return visible diagnostics instead of crashing.
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
PROXY_ROLES = {
    "BIL": "US Treasury bills / cash proxy",
    "SHY": "1-3 year US Treasuries proxy",
    "IEI": "3-7 year US Treasuries proxy",
    "IEF": "7-10 year US Treasuries proxy",
    "TLT": "20+ year US Treasuries proxy",
    "GLD": "Gold ETF proxy",
    "SPY": "S&P 500 ETF proxy; mainly for stress comparison",
    "AGG": "US aggregate bond ETF proxy",
    "BND": "Total US bond market ETF proxy",
    "FXE": "Euro currency ETF proxy",
    "FXY": "Japanese yen currency ETF proxy",
}

STRESS_LIBRARY = {
    "Global bond selloff": {"BIL": 0.0005, "SHY": -0.002, "IEI": -0.006, "IEF": -0.010, "TLT": -0.030, "GLD": 0.006, "SPY": -0.025, "AGG": -0.012, "BND": -0.012},
    "USD liquidity squeeze": {"BIL": 0.0005, "SHY": 0.001, "IEI": 0.0015, "IEF": 0.002, "TLT": 0.004, "GLD": -0.006, "SPY": -0.030, "AGG": 0.000, "BND": 0.000},
    "Gold selloff": {"BIL": 0.0000, "SHY": 0.0000, "IEI": 0.001, "IEF": 0.001, "TLT": 0.002, "GLD": -0.050, "SPY": 0.000, "AGG": 0.001, "BND": 0.001},
    "Broad risk-off shock": {"BIL": 0.0005, "SHY": 0.002, "IEI": 0.003, "IEF": 0.004, "TLT": 0.008, "GLD": 0.010, "SPY": -0.040, "AGG": 0.000, "BND": 0.000},
}

st.set_page_config(page_title="FX Reserve Portfolio Optimizer", layout="wide")


def parse_tickers(text: str) -> list[str]:
    parts = text.replace("\n", ",").replace(";", ",").replace(" ", ",").split(",")
    tickers: list[str] = []
    for part in parts:
        ticker = part.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return demo_dataset()


def clean_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError("No price data returned.")
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce").tz_localize(None)
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns = [str(c).strip().upper() for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    if prices.shape[1] < 2:
        raise ValueError("Need at least two usable assets after cleaning.")
    if len(prices) < 60:
        raise ValueError("Need at least 60 observations. Use a longer date range.")
    return prices


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Daily":
        return prices
    if frequency == "Weekly":
        return prices.resample("W-FRI").last().dropna(how="any")
    if frequency == "Monthly":
        return prices.resample("ME").last().dropna(how="any")
    return prices


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def make_panel(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = resample_prices(clean_price_panel(prices), frequency)
    returns = prices_to_returns(prices)
    prices = prices.loc[returns.index]
    return prices, returns


def stooq_symbol(ticker: str) -> str | None:
    symbol = ticker.strip().upper()
    if not symbol or any(marker in symbol for marker in ("=", "^", "/")):
        return None
    if not symbol.replace("-", "").replace(".", "").isalnum():
        return None
    return f"{symbol.lower()}.us"


@st.cache_data(ttl=3600, show_spinner="Fetching Stooq data...")
def fetch_stooq_prices(tickers: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    series: dict[str, pd.Series] = {}
    diagnostics: list[dict[str, object]] = []
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")
    headers = {"User-Agent": "Mozilla/5.0 fx-reserve-rl-optimizer"}

    for ticker in tickers:
        mapped = stooq_symbol(ticker)
        if mapped is None:
            diagnostics.append({"ticker": ticker, "provider_symbol": "", "status": "skipped", "rows": 0, "detail": "Not a Stooq ETF/equity symbol"})
            continue
        url = f"https://stooq.com/q/d/l/?s={mapped}&d1={start_str}&d2={end_str}&i=d"
        try:
            response = requests.get(url, timeout=20, headers=headers)
            if response.status_code != 200:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "rows": 0, "detail": f"HTTP {response.status_code}"})
                continue
            frame = pd.read_csv(StringIO(response.text))
            if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "rows": 0, "detail": "No Date/Close columns"})
                continue
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(), index=frame["Date"], name=ticker).dropna()
            if close.empty:
                diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "rows": 0, "detail": "Empty close series"})
                continue
            series[ticker] = close
            diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "ok", "rows": int(len(close)), "detail": "Fetched"})
        except Exception as exc:  # pragma: no cover - network dependent
            diagnostics.append({"ticker": ticker, "provider_symbol": mapped, "status": "failed", "rows": 0, "detail": str(exc)})

    diag = pd.DataFrame(diagnostics)
    if len(series) < 2:
        return pd.DataFrame(), diag, "Stooq returned fewer than two usable series."
    try:
        return clean_price_panel(pd.concat(series.values(), axis=1)), diag, ""
    except Exception as exc:
        return pd.DataFrame(), diag, str(exc)


@st.cache_data(ttl=3600, show_spinner="Fetching Yahoo/yfinance data...")
def fetch_yfinance_prices(tickers: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    diagnostics: list[dict[str, object]] = []
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency
        diag = pd.DataFrame([{"ticker": "ALL", "provider_symbol": "yfinance", "status": "failed", "rows": 0, "detail": f"Import failed: {exc}"}])
        return pd.DataFrame(), diag, "yfinance import failed."

    try:
        raw = yf.download(list(tickers), start=str(start), end=str(end), auto_adjust=True, progress=False, threads=False, group_by="column", timeout=20)
        if raw.empty:
            diag = pd.DataFrame([{"ticker": "ALL", "provider_symbol": "Yahoo", "status": "failed", "rows": 0, "detail": "Empty dataframe"}])
            return pd.DataFrame(), diag, "Yahoo/yfinance returned an empty dataframe."
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
        close.columns = [str(c).upper() for c in close.columns]
        available = [ticker for ticker in tickers if ticker in close.columns]
        close = close[available]
        for ticker in tickers:
            rows = int(close[ticker].dropna().shape[0]) if ticker in close.columns else 0
            diagnostics.append({"ticker": ticker, "provider_symbol": ticker, "status": "ok" if rows else "failed", "rows": rows, "detail": "Fetched" if rows else "No usable close prices"})
        return clean_price_panel(close), pd.DataFrame(diagnostics), ""
    except Exception as exc:  # pragma: no cover - network/upstream dependent
        diag = pd.DataFrame(diagnostics or [{"ticker": "ALL", "provider_symbol": "Yahoo", "status": "failed", "rows": 0, "detail": str(exc)}])
        return pd.DataFrame(), diag, f"Yahoo/yfinance failed: {exc}"


def parse_uploaded_price_csv(uploaded_file) -> pd.DataFrame:
    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        raise ValueError("Uploaded CSV is empty.")
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col)
    frame.columns = [str(col).strip().upper() for col in frame.columns]
    return clean_price_panel(frame)


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, legend_title=None)
    return fig


def normalize_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.divide(prices.iloc[0]).multiply(100.0)


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


def build_strategy_set(returns: pd.DataFrame, tc_bps: float, window: int, max_weight: float, risk_aversion: float, custom_weights: pd.Series, rebalance: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_strategies = {
        "Equal Weight": equal_weight(returns),
        "Static Reserve": static_reserve_weights(returns),
        "Minimum Variance": rolling_minimum_variance(returns, window=window, max_weight=max_weight),
        "Mean Variance": rolling_mean_variance(returns, window=window, risk_aversion=risk_aversion, max_weight=max_weight),
        "Your Custom Portfolio": constant_weight_frame(custom_weights, returns.index),
    }
    strategies = {name: apply_rebalance_schedule(weights, rebalance) for name, weights in raw_strategies.items()}
    strategy_returns = pd.DataFrame({name: portfolio_returns(returns, weights, transaction_cost_bps=tc_bps) for name, weights in strategies.items()}, index=returns.index)
    return strategy_returns, strategies


def annualized_turnover(weights: pd.DataFrame) -> float:
    turnover = portfolio_turnover(weights)
    return 0.0 if turnover.empty else float(turnover.mean() * 252.0)


def add_turnover_and_score(metrics: pd.DataFrame, weights_by_strategy: dict[str, pd.DataFrame], lambda_vol: float, lambda_drawdown: float, lambda_turnover: float) -> pd.DataFrame:
    enriched = metrics.copy()
    enriched["annualized_turnover"] = [annualized_turnover(weights_by_strategy[name]) for name in enriched.index]
    enriched["reserve_utility_score"] = enriched["annualized_return"] - lambda_vol * enriched["annualized_volatility"] - lambda_drawdown * enriched["max_drawdown"].abs() - lambda_turnover * enriched["annualized_turnover"]
    return enriched.sort_values("reserve_utility_score", ascending=False)


def data_quality_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    annualization = 252.0 if len(returns) >= 100 else 52.0 if len(returns) > 24 else 12.0
    rows = []
    for asset in prices.columns:
        rows.append({"asset": asset, "first_date": prices[asset].first_valid_index().date(), "last_date": prices[asset].last_valid_index().date(), "observations": int(returns[asset].count()), "annualized_return": float(returns[asset].mean() * annualization), "annualized_volatility": float(returns[asset].std(ddof=0) * annualization**0.5), "latest_price": float(prices[asset].iloc[-1])})
    return pd.DataFrame(rows).set_index("asset")


def drawdown_chart(strategy_returns: pd.DataFrame) -> go.Figure:
    dd = pd.DataFrame({col: drawdown_series(strategy_returns[col]) for col in strategy_returns.columns})
    return line_chart(dd, "Drawdown comparison", "Drawdown")


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


def as_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


st.title("FX Reserve Portfolio Optimizer")
st.caption("Reserve-allocation simulator with demo data, CSV upload, Stooq, and Yahoo/yfinance diagnostics")
st.warning(DISCLAIMER)

st.sidebar.header("Data source")
data_source = st.sidebar.radio("Choose market-data mode", ["Demo data", "Upload price CSV", "Live Stooq ETF data", "Live Yahoo/yfinance data"])
frequency = st.sidebar.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"], index=0)
provider_diagnostics = pd.DataFrame()
source_note = ""

try:
    if data_source == "Demo data":
        prices_all, returns_all = load_demo_data()
        selected_assets = st.sidebar.multiselect("Assets included", list(returns_all.columns), default=list(returns_all.columns))
        if len(selected_assets) < 2:
            st.stop()
        min_date = returns_all.index.min().date()
        max_date = returns_all.index.max().date()
        start_date, end_date = st.sidebar.slider("Backtest date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        prices = prices_all.loc[str(start_date):str(end_date), selected_assets]
        returns = returns_all.loc[str(start_date):str(end_date), selected_assets]
        role_lookup = DEMO_ASSET_ROLES | PROXY_ROLES
        source_note = "Synthetic deterministic demo data generated inside the app."
    elif data_source == "Upload price CSV":
        uploaded = st.sidebar.file_uploader("Upload price CSV", type=["csv"])
        st.sidebar.caption("Format: Date column + one price column per asset.")
        if uploaded is None:
            st.info("Upload a price CSV, or switch to Demo data.")
            st.stop()
        uploaded_prices = parse_uploaded_price_csv(uploaded)
        selected_assets = st.sidebar.multiselect("Assets included", list(uploaded_prices.columns), default=list(uploaded_prices.columns))
        if len(selected_assets) < 2:
            st.stop()
        prices, returns = make_panel(uploaded_prices[selected_assets], frequency)
        role_lookup = {asset: "Uploaded price series" for asset in selected_assets}
        source_note = "User-uploaded price CSV. This is the most reliable real-data route if hosted providers fail."
    else:
        tickers = parse_tickers(st.sidebar.text_area("Ticker symbols", value=DEFAULT_TICKERS, help="For Stooq, start with: BIL, SHY, IEF, TLT, GLD"))
        if len(tickers) < 2:
            st.error("Enter at least two tickers.")
            st.stop()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start", value=date(2018, 1, 1), max_value=date.today())
        with col2:
            end_date = st.date_input("End", value=date.today(), max_value=date.today())
        if data_source == "Live Stooq ETF data":
            raw_prices, provider_diagnostics, error = fetch_stooq_prices(tuple(tickers), start_date, end_date)
            source_note = "Live ETF/equity data requested from Stooq. See diagnostics below."
        else:
            raw_prices, provider_diagnostics, error = fetch_yfinance_prices(tuple(tickers), start_date, end_date)
            source_note = "Live market data requested from Yahoo/yfinance. See diagnostics below."
        if error:
            st.error(error)
            if not provider_diagnostics.empty:
                st.dataframe(provider_diagnostics, width="stretch")
            st.info("Switch to Demo data, use Upload price CSV, or for Stooq test exactly: BIL, SHY, IEF, TLT, GLD.")
            st.stop()
        prices, returns = make_panel(raw_prices, frequency)
        selected_assets = list(returns.columns)
        role_lookup = {asset: PROXY_ROLES.get(asset, "Live market symbol") for asset in selected_assets}
except Exception as exc:
    st.error("Data loading failed before portfolio simulation could start.")
    st.exception(exc)
    st.stop()

st.sidebar.header("Simulation controls")
tc_bps = st.sidebar.slider("Transaction cost, bps", 0.0, 25.0, 1.0, 0.5)
rolling_window = st.sidebar.slider("Rolling estimation window", 12, 252, 63, 1)
max_weight = st.sidebar.slider("Maximum optimized weight per asset", 0.20, 1.00, 0.60, 0.05)
risk_aversion = st.sidebar.slider("Mean-variance risk aversion", 1.0, 50.0, 10.0, 1.0)
rebalance = st.sidebar.selectbox("Rebalancing frequency", ["Every observation", "Weekly", "Monthly", "Quarterly"])

st.sidebar.header("Reserve utility preferences")
lambda_vol = st.sidebar.slider("Volatility penalty", 0.0, 5.0, 1.0, 0.1)
lambda_drawdown = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lambda_turnover = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

if len(returns) < rolling_window + 5:
    st.error("The selected data has too few observations for the rolling window. Reduce the window or use more data.")
    st.stop()

st.sidebar.header("Custom portfolio")
raw_custom = {asset: st.sidebar.slider(f"{asset} raw weight", 0.0, 100.0, round(100.0 / len(selected_assets), 1), 1.0) for asset in selected_assets}
custom_weights = normalise_weights(raw_custom, selected_assets)
strategy_returns, weights_by_strategy = build_strategy_set(returns, tc_bps, rolling_window, max_weight, risk_aversion, custom_weights, rebalance)
metrics_scored = add_turnover_and_score(metrics_table(strategy_returns), weights_by_strategy, lambda_vol, lambda_drawdown, lambda_turnover)
equity = (1.0 + strategy_returns).cumprod()

selected_strategy = st.sidebar.selectbox("Strategy to inspect", list(strategy_returns.columns), index=1 if "Static Reserve" in strategy_returns.columns else 0)
stress_name = st.sidebar.selectbox("One-day stress scenario", list(STRESS_LIBRARY.keys()) + ["Custom one-day shock"])
custom_shocks = {}
if stress_name == "Custom one-day shock":
    for asset in selected_assets:
        custom_shocks[asset] = st.sidebar.slider(f"{asset} shock (%)", -10.0, 10.0, 0.0, 0.25)
shock = stress_vector(stress_name, selected_assets, custom_shocks)
stress_results = stress_test_table(weights_by_strategy, shock)

st.header("What this project does")
st.write("This app compares reserve-style allocation rules under different data sources, transaction costs, constraints, custom weights, and stress scenarios.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Data mode", data_source)
c2.metric("Assets", len(selected_assets))
c3.metric("Observations", len(returns))
c4.metric("Transaction cost", f"{tc_bps:.1f} bps")
c5.metric("Best utility strategy", metrics_scored.index[0])
st.info(source_note)

if not provider_diagnostics.empty:
    with st.expander("Live provider diagnostics", expanded=True):
        st.dataframe(provider_diagnostics, width="stretch")

st.header("1. Asset universe and market data")
asset_table = pd.DataFrame({"asset": selected_assets, "role": [role_lookup.get(asset, "User-selected proxy") for asset in selected_assets], "custom_weight": [custom_weights.loc[asset] for asset in selected_assets], "stress_shock": [shock.loc[asset] for asset in selected_assets]})
st.dataframe(asset_table.style.format({"custom_weight": "{:.2%}", "stress_shock": "{:.2%}"}), width="stretch")

with st.expander("Data quality", expanded=data_source != "Demo data"):
    st.dataframe(data_quality_summary(prices, returns).style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "latest_price": "{:.2f}"}), width="stretch")

left, right = st.columns(2)
with left:
    st.plotly_chart(line_chart(normalize_price_panel(prices), "Selected asset price indices", "Index, first observation = 100"), width="stretch")
with right:
    st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", aspect="auto", title="Return correlation matrix"), width="stretch")

st.header("2. Strategy performance comparison")
st.plotly_chart(line_chart(equity, "Cumulative wealth after transaction costs", "Wealth index"), width="stretch")
sel = metrics_scored.loc[selected_strategy]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Annualized return", f"{sel['annualized_return']:.2%}")
m2.metric("Annualized volatility", f"{sel['annualized_volatility']:.2%}")
m3.metric("Max drawdown", f"{sel['max_drawdown']:.2%}")
m4.metric("Reserve utility score", f"{sel['reserve_utility_score']:.3f}")
st.dataframe(metrics_scored.style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "sharpe_ratio": "{:.2f}", "sortino_ratio": "{:.2f}", "max_drawdown": "{:.2%}", "calmar_ratio": "{:.2f}", "var_95": "{:.2%}", "expected_shortfall_95": "{:.2%}", "annualized_turnover": "{:.2f}", "reserve_utility_score": "{:.3f}"}), width="stretch")

d1, d2, d3, d4 = st.columns(4)
d1.download_button("Download metrics CSV", as_csv_download(metrics_scored), "strategy_metrics.csv", "text/csv")
d2.download_button("Download returns CSV", as_csv_download(strategy_returns), "strategy_returns.csv", "text/csv")
latest_weights = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
d3.download_button("Download weights CSV", as_csv_download(latest_weights), "latest_weights.csv", "text/csv")
d4.download_button("Download prices CSV", as_csv_download(prices), "prices.csv", "text/csv")

st.header("3. Drawdown and allocation diagnostics")
st.plotly_chart(drawdown_chart(strategy_returns), width="stretch")
st.plotly_chart(px.area(weights_by_strategy[selected_strategy], x=weights_by_strategy[selected_strategy].index, y=weights_by_strategy[selected_strategy].columns, title=f"{selected_strategy}: allocation over time"), width="stretch")

st.header("4. Stress test")
a, b = st.columns(2)
with a:
    st.dataframe(shock.rename("one_day_asset_shock").to_frame().style.format("{:.2%}"), width="stretch")
with b:
    st.dataframe(stress_results.style.format({"one_day_stress_return": "{:.2%}", "one_day_stress_loss": "{:.2%}"}), width="stretch")
st.plotly_chart(px.bar(stress_results.reset_index(), x="strategy", y="one_day_stress_loss", title=f"Stress loss under: {stress_name}"), width="stretch")

st.header("5. DRL implementation status")
st.markdown(
    """
The repository includes the reinforcement-learning backend files:

```text
src/envs/fx_reserve_env.py
src/agents/train_ppo.py
src/portfolio/constraints.py
src/portfolio/metrics.py
```

The deployed Streamlit app is the lightweight inspection layer. PPO training should be run offline and then exported into the dashboard.
"""
)

with st.expander("Show raw calculated data"):
    st.subheader("Strategy returns")
    st.dataframe(strategy_returns, width="stretch")
    st.subheader("Latest weights")
    st.dataframe(latest_weights.style.format("{:.2%}"), width="stretch")
    st.subheader("Market returns")
    st.dataframe(returns, width="stretch")
