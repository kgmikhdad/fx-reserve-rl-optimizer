"""Professional Streamlit dashboard for FX reserve portfolio analysis."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.demo_data import ASSET_ROLES as DEMO_ASSET_ROLES  # noqa: E402
from src.data.demo_data import demo_dataset  # noqa: E402
from src.data.market_data import (  # noqa: E402
    ASSET_CATALOG,
    asset_roles,
    data_quality_summary,
    fetch_market_dataset,
)
from src.portfolio.baselines import (  # noqa: E402
    equal_weight,
    portfolio_returns,
    rolling_mean_variance,
    rolling_minimum_variance,
    static_reserve_weights,
)
from src.portfolio.metrics import drawdown_series, metrics_table, portfolio_turnover  # noqa: E402

STRESS_LIBRARY: dict[str, dict[str, float]] = {
    "Global bond selloff": {
        "BIL": 0.0005,
        "SHY": -0.002,
        "IEI": -0.006,
        "IEF": -0.010,
        "TLT": -0.030,
        "GLD": 0.006,
        "FXE": -0.004,
        "FXY": 0.003,
        "FXB": -0.006,
        "FXC": -0.006,
        "FXA": -0.008,
        "UUP": 0.006,
    },
    "USD liquidity squeeze": {
        "BIL": 0.0005,
        "SHY": 0.001,
        "IEI": 0.0015,
        "IEF": 0.002,
        "TLT": 0.004,
        "GLD": -0.006,
        "FXE": -0.018,
        "FXY": -0.012,
        "FXB": -0.016,
        "FXC": -0.014,
        "FXA": -0.020,
        "UUP": 0.020,
    },
    "Gold selloff": {
        "BIL": 0.0000,
        "SHY": 0.000,
        "IEI": 0.001,
        "IEF": 0.001,
        "TLT": 0.002,
        "GLD": -0.050,
        "FXE": -0.002,
        "FXY": -0.002,
        "FXB": -0.002,
        "FXC": -0.002,
        "FXA": -0.002,
        "UUP": 0.002,
    },
    "Broad risk-off shock": {
        "BIL": 0.0005,
        "SHY": 0.002,
        "IEI": 0.003,
        "IEF": 0.004,
        "TLT": 0.008,
        "GLD": 0.010,
        "FXE": -0.012,
        "FXY": 0.006,
        "FXB": -0.010,
        "FXC": -0.012,
        "FXA": -0.016,
        "UUP": 0.010,
    },
    "Foreign-currency depreciation": {
        "BIL": 0.0000,
        "SHY": 0.0000,
        "IEI": 0.0000,
        "IEF": 0.0000,
        "TLT": 0.0000,
        "GLD": 0.0000,
        "FXE": -0.025,
        "FXY": -0.025,
        "FXB": -0.025,
        "FXC": -0.025,
        "FXA": -0.025,
        "UUP": 0.015,
    },
}

st.set_page_config(page_title="FX Reserve Portfolio Dashboard", layout="wide")


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return demo_dataset()


@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_live_data(
    assets: tuple[str, ...],
    start: date,
    end: date,
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_market_dataset(list(assets), start=start, end=end, frequency=frequency)


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, legend_title=None, height=430)
    return fig


def drawdown_chart(strategy_returns: pd.DataFrame) -> go.Figure:
    dd = pd.DataFrame({col: drawdown_series(strategy_returns[col]) for col in strategy_returns.columns})
    return line_chart(dd, "Drawdown", "Drawdown")


def weight_area_chart(weights: pd.DataFrame, title: str) -> go.Figure:
    fig = px.area(weights, x=weights.index, y=weights.columns, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Weight", legend_title=None, height=430)
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    return px.imshow(corr, text_auto=".2f", aspect="auto", title="Return correlation")


def normalise_weights(raw_weights: dict[str, float], assets: list[str]) -> pd.Series:
    values = np.array([raw_weights.get(asset, 0.0) for asset in assets], dtype="float64")
    values = np.maximum(values, 0.0)
    if np.isclose(values.sum(), 0.0):
        values = np.ones(len(assets)) / len(assets)
    else:
        values = values / values.sum()
    return pd.Series(values, index=assets)


def constant_weight_frame(weights: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.tile(weights.to_numpy(), (len(index), 1)), index=index, columns=weights.index)


def apply_rebalance_schedule(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Every observation":
        return weights
    rule = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE"}[frequency]
    scheduled = weights.resample(rule).last()
    scheduled = scheduled.reindex(weights.index, method="ffill")
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
        "Mean Variance": rolling_mean_variance(
            returns, window=rolling_window, risk_aversion=risk_aversion, max_weight=max_weight
        ),
        "Custom Portfolio": constant_weight_frame(custom_weights, returns.index),
    }
    strategies = {name: apply_rebalance_schedule(weights, rebalance_frequency) for name, weights in raw_strategies.items()}
    strategy_returns = pd.DataFrame(
        {
            name: portfolio_returns(returns, weights, transaction_cost_bps=transaction_cost_bps)
            for name, weights in strategies.items()
        },
        index=returns.index,
    )
    return strategy_returns, strategies


def annualized_turnover(weights: pd.DataFrame) -> float:
    turnover = portfolio_turnover(weights)
    if turnover.empty:
        return 0.0
    return float(turnover.mean() * 252.0)


def add_turnover_and_score(
    metrics: pd.DataFrame,
    weights_by_strategy: dict[str, pd.DataFrame],
    lambda_vol: float,
    lambda_drawdown: float,
    lambda_turnover: float,
) -> pd.DataFrame:
    enriched = metrics.copy()
    enriched["annualized_turnover"] = [annualized_turnover(weights_by_strategy[name]) for name in enriched.index]
    enriched["reserve_score"] = (
        enriched["annualized_return"]
        - lambda_vol * enriched["annualized_volatility"]
        - lambda_drawdown * enriched["max_drawdown"].abs()
        - lambda_turnover * enriched["annualized_turnover"]
    )
    return enriched.sort_values("reserve_score", ascending=False)


def stress_vector(name: str, assets: list[str], custom_shocks_pct: dict[str, float]) -> pd.Series:
    if name == "Custom shock":
        return pd.Series({asset: custom_shocks_pct.get(asset, 0.0) / 100.0 for asset in assets})
    template = STRESS_LIBRARY[name]
    return pd.Series({asset: template.get(asset, 0.0) for asset in assets})


def stress_test_table(weights_by_strategy: dict[str, pd.DataFrame], shock: pd.Series) -> pd.DataFrame:
    rows = []
    for strategy_name, weights in weights_by_strategy.items():
        latest = weights.iloc[-1].reindex(shock.index).fillna(0.0)
        stress_return = float((latest * shock).sum())
        rows.append({"strategy": strategy_name, "stress_return": stress_return, "stress_loss": -stress_return})
    return pd.DataFrame(rows).set_index("strategy").sort_values("stress_loss")


def normalize_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.divide(prices.iloc[0]).multiply(100.0)


def as_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def rolling_volatility(strategy_returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return strategy_returns.rolling(window).std().multiply(np.sqrt(252.0)).dropna(how="all")


def rolling_sharpe(strategy_returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    mean = strategy_returns.rolling(window).mean().multiply(252.0)
    vol = strategy_returns.rolling(window).std().multiply(np.sqrt(252.0))
    return mean.divide(vol.replace(0.0, np.nan)).dropna(how="all")


def risk_contribution(weights: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    cov = returns.cov().reindex(index=weights.index, columns=weights.index).fillna(0.0)
    w = weights.to_numpy(dtype="float64")
    sigma_w = cov.to_numpy() @ w
    port_var = float(w @ sigma_w)
    if np.isclose(port_var, 0.0):
        contribution = np.zeros_like(w)
    else:
        contribution = w * sigma_w / port_var
    return pd.DataFrame({"weight": w, "risk_contribution": contribution}, index=weights.index)


st.title("FX Reserve Portfolio Dashboard")
st.caption("Strategy comparison · risk diagnostics · stress testing")

with st.expander("Scope", expanded=False):
    st.write("Research prototype using public proxies or synthetic demo data. Not investment advice.")

st.sidebar.header("Data")
data_source = st.sidebar.radio("Source", ["Demo data", "Real market data via yfinance"])
frequency = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)

if data_source == "Demo data":
    prices_all, returns_all = load_demo_data()
    role_lookup = DEMO_ASSET_ROLES | asset_roles()
    asset_options = list(returns_all.columns)
    selected_assets = st.sidebar.multiselect("Asset universe", asset_options, default=asset_options)
    if len(selected_assets) < 2:
        st.sidebar.error("Select at least two assets.")
        st.stop()
    min_date = returns_all.index.min().date()
    max_date = returns_all.index.max().date()
    start_date, end_date = st.sidebar.slider("Backtest window", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    prices = prices_all.loc[str(start_date) : str(end_date), selected_assets]
    returns = returns_all.loc[str(start_date) : str(end_date), selected_assets]
    source_note = "Demo data"
else:
    role_lookup = asset_roles()
    selected_assets = st.sidebar.multiselect(
        "Asset universe",
        list(ASSET_CATALOG.keys()),
        default=["BIL", "SHY", "IEF", "TLT", "GLD", "FXE", "FXY"],
    )
    if len(selected_assets) < 2:
        st.sidebar.error("Select at least two assets.")
        st.stop()
    live_col1, live_col2 = st.sidebar.columns(2)
    with live_col1:
        start_date = st.date_input("Start", value=date(2018, 1, 1), max_value=date.today())
    with live_col2:
        end_date = st.date_input("End", value=date.today(), max_value=date.today())
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    try:
        prices, returns = load_live_data(tuple(selected_assets), start_date, end_date, frequency)
    except Exception as exc:
        st.error("Market data unavailable for the current selection.")
        st.exception(exc)
        st.stop()
    source_note = "Live proxy data"

st.sidebar.header("Portfolio settings")
transaction_cost_bps = st.sidebar.slider("Transaction cost, bps", 0.0, 25.0, 1.0, 0.5)
rolling_window = st.sidebar.slider("Estimation window", 12, 252, 63, 1)
max_weight = st.sidebar.slider("Max weight", 0.20, 1.00, 0.60, 0.05)
risk_aversion = st.sidebar.slider("Risk aversion", 1.0, 50.0, 10.0, 1.0)
rebalance_frequency = st.sidebar.selectbox("Rebalancing", ["Every observation", "Weekly", "Monthly", "Quarterly"], index=0)

st.sidebar.header("Reserve score")
lambda_vol = st.sidebar.slider("Volatility penalty", 0.0, 5.0, 1.0, 0.1)
lambda_drawdown = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lambda_turnover = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

if len(returns) < rolling_window + 5:
    st.error("Selected window is too short for the estimation window.")
    st.stop()

st.sidebar.header("Custom portfolio")
raw_custom_weights = {
    asset: st.sidebar.slider(f"{asset}", min_value=0.0, max_value=100.0, value=round(100.0 / len(selected_assets), 1), step=1.0)
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

selected_strategy = st.sidebar.selectbox("Inspect strategy", list(strategy_returns.columns), index=1)
stress_name = st.sidebar.selectbox("Stress scenario", list(STRESS_LIBRARY.keys()) + ["Custom shock"])
custom_shocks_pct: dict[str, float] = {}
if stress_name == "Custom shock":
    for asset in selected_assets:
        custom_shocks_pct[asset] = st.sidebar.slider(f"{asset} shock (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.25)
shock = stress_vector(stress_name, selected_assets, custom_shocks_pct)
stress_results = stress_test_table(weights_by_strategy, shock)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Source", source_note)
kpi2.metric("Assets", len(selected_assets))
kpi3.metric("Observations", len(returns))
kpi4.metric("Best strategy", metrics_scored.index[0])
kpi5.metric("Best score", f"{metrics_scored['reserve_score'].iloc[0]:.3f}")

tab_overview, tab_data, tab_strategy, tab_risk, tab_stress, tab_export = st.tabs(
    ["Overview", "Data", "Strategy", "Risk", "Stress", "Export"]
)

with tab_overview:
    st.plotly_chart(line_chart(equity, "Cumulative wealth", "Wealth index"), use_container_width=True)
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
                "reserve_score": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

with tab_data:
    asset_table = pd.DataFrame(
        {
            "asset": selected_assets,
            "role": [role_lookup.get(asset, "User-selected proxy") for asset in selected_assets],
            "custom_weight": [custom_weights.loc[asset] for asset in selected_assets],
            "stress_shock": [shock.loc[asset] for asset in selected_assets],
        }
    )
    st.dataframe(asset_table.style.format({"custom_weight": "{:.2%}", "stress_shock": "{:.2%}"}), use_container_width=True)
    st.plotly_chart(line_chart(normalize_price_panel(prices), "Price index", "Index"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(correlation_heatmap(returns), use_container_width=True)
    with col_b:
        quality = data_quality_summary(prices, returns)
        st.dataframe(
            quality.style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "latest_price": "{:.2f}"}),
            use_container_width=True,
        )

with tab_strategy:
    selected_metrics = metrics_scored.loc[selected_strategy]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Annualized return", f"{selected_metrics['annualized_return']:.2%}")
    s2.metric("Annualized volatility", f"{selected_metrics['annualized_volatility']:.2%}")
    s3.metric("Max drawdown", f"{selected_metrics['max_drawdown']:.2%}")
    s4.metric("Reserve score", f"{selected_metrics['reserve_score']:.3f}")
    st.plotly_chart(weight_area_chart(weights_by_strategy[selected_strategy], f"{selected_strategy} allocation"), use_container_width=True)
    latest_weights = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
    st.dataframe(latest_weights.style.format("{:.2%}"), use_container_width=True)

with tab_risk:
    st.plotly_chart(drawdown_chart(strategy_returns), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(line_chart(rolling_volatility(strategy_returns), "Rolling volatility", "Volatility"), use_container_width=True)
    with col_b:
        st.plotly_chart(line_chart(rolling_sharpe(strategy_returns), "Rolling Sharpe", "Sharpe"), use_container_width=True)
    st.subheader("Risk contribution")
    rc = risk_contribution(weights_by_strategy[selected_strategy].iloc[-1], returns)
    st.dataframe(rc.style.format({"weight": "{:.2%}", "risk_contribution": "{:.2%}"}), use_container_width=True)

with tab_stress:
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(shock.rename("asset_shock").to_frame().style.format("{:.2%}"), use_container_width=True)
    with col_b:
        st.dataframe(stress_results.style.format({"stress_return": "{:.2%}", "stress_loss": "{:.2%}"}), use_container_width=True)
    stress_fig = px.bar(stress_results.reset_index(), x="strategy", y="stress_loss", title=f"Stress loss: {stress_name}")
    st.plotly_chart(stress_fig, use_container_width=True)

with tab_export:
    latest_weights = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Metrics", as_csv_download(metrics_scored), "strategy_metrics.csv", "text/csv")
    c2.download_button("Returns", as_csv_download(strategy_returns), "strategy_returns.csv", "text/csv")
    c3.download_button("Weights", as_csv_download(latest_weights), "latest_strategy_weights.csv", "text/csv")
    c4.download_button("Prices", as_csv_download(prices), "prices.csv", "text/csv")

with st.expander("Raw tables", expanded=False):
    st.dataframe(strategy_returns, use_container_width=True)
    st.dataframe(returns, use_container_width=True)
