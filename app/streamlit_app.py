"""Interactive Streamlit dashboard for the FX reserve RL optimizer."""
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
    parse_ticker_text,
)
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


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deterministic demo market data for immediate deployment."""
    return demo_dataset()


@st.cache_data(ttl=3600, show_spinner="Downloading live market data...")
def load_live_data(
    assets: tuple[str, ...],
    start: date,
    end: date,
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and cache public market data for the app."""
    return fetch_market_dataset(list(assets), start=start, end=end, frequency=frequency)


def unique_assets(*groups: list[str]) -> list[str]:
    """Return unique non-empty ticker/asset labels while preserving user order."""
    output: list[str] = []
    for group in groups:
        for asset in group:
            clean = asset.strip().upper()
            if clean and clean not in output:
                output.append(clean)
    return output


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
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
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Asset return correlation matrix")
    return fig


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
    """Convert daily rolling weights into lower-frequency rebalanced weights."""
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
        "Your Custom Portfolio": constant_weight_frame(custom_weights, returns.index),
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
    """Normalize each price series to 100 at the first observation."""
    return prices.divide(prices.iloc[0]).multiply(100.0)


def as_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


st.title("FX Reserve Portfolio Optimizer")
st.caption("Interactive reserve-allocation simulator, live-data dashboard, and DRL project prototype")
st.warning(DISCLAIMER)

st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Choose market-data mode",
    ["Demo data", "Real market data via yfinance"],
    help=(
        "Demo data is deterministic and always available. Real market data downloads public "
        "Yahoo Finance proxy data through yfinance and is cached for one hour."
    ),
)
frequency = st.sidebar.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"], index=0)

if data_source == "Demo data":
    prices_all, returns_all = load_demo_data()
    role_lookup = DEMO_ASSET_ROLES | asset_roles()
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
    source_note = "Synthetic deterministic demo data generated inside the app."
else:
    role_lookup = asset_roles()
    catalog_assets = st.sidebar.multiselect(
        "Built-in public proxy assets",
        list(ASSET_CATALOG.keys()),
        default=["BIL", "SHY", "IEF", "TLT", "GLD", "FXE", "FXY"],
        help="This search box filters only the built-in proxy catalogue. Use the custom ticker box below for any Yahoo Finance symbol.",
    )
    custom_ticker_text = st.sidebar.text_input(
        "Add custom Yahoo Finance tickers",
        value="",
        placeholder="Example: SPY, AGG, EURUSD=X, GC=F",
        help="Type comma- or space-separated Yahoo Finance tickers. Examples: SPY, AGG, EURUSD=X, JPY=X, GC=F, ^TNX.",
    )
    custom_assets = parse_ticker_text(custom_ticker_text)
    requested_assets = unique_assets(catalog_assets, custom_assets)

    if custom_assets:
        st.sidebar.caption(f"Custom tickers requested: {', '.join(custom_assets)}")
    if len(requested_assets) < 2:
        st.sidebar.error("Select at least two assets or add at least two valid Yahoo tickers.")
        st.stop()

    live_col1, live_col2 = st.sidebar.columns(2)
    with live_col1:
        start_date = st.date_input("Start", value=date(2018, 1, 1), max_value=date.today())
    with live_col2:
        end_date = st.date_input("End", value=date.today(), max_value=date.today())
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()
    try:
        prices, returns = load_live_data(tuple(requested_assets), start_date, end_date, frequency)
    except Exception as exc:
        st.error("Live data could not be loaded. Switch to Demo data or try a longer date range.")
        st.info(
            "The built-in search box only searches the app's proxy list. For arbitrary data, use "
            "the custom Yahoo ticker box. Good examples: BIL, SHY, IEF, TLT, GLD, SPY, AGG, EURUSD=X, GC=F."
        )
        st.exception(exc)
        st.stop()

    selected_assets = list(returns.columns)
    dropped_assets = [asset for asset in requested_assets if asset not in selected_assets]
    if dropped_assets:
        st.warning(
            "Some requested symbols were not used because no complete price panel was returned: "
            + ", ".join(dropped_assets)
        )
    for asset in selected_assets:
        role_lookup.setdefault(asset, "Custom Yahoo Finance ticker selected by the user")
    source_note = (
        "Live public market proxy data downloaded with yfinance. The built-in selector searches only "
        "the predefined proxy catalogue; the custom ticker box fetches arbitrary Yahoo Finance symbols. "
        "Data availability, licensing, and reliability depend on the upstream public source."
    )

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
several allocation rules:

1. **Equal Weight**: naive diversification benchmark.  
2. **Static Reserve**: stylised conservative reserve benchmark.  
3. **Minimum Variance**: rolling risk-minimising allocation.  
4. **Mean Variance**: rolling return-risk allocation.  
5. **Your Custom Portfolio**: manually chosen reserve weights.  

The deep reinforcement learning part is implemented in the repository as a Gymnasium environment and
PPO training script. The dashboard is the inspection layer: it shows how benchmark and future trained
DRL strategies should be judged using return, volatility, drawdown, turnover, stress loss, and a
reserve-utility score.
"""
)

input_col1, input_col2, input_col3, input_col4, input_col5 = st.columns(5)
input_col1.metric("Data mode", "Live" if data_source.startswith("Real") else "Demo")
input_col2.metric("Assets selected", len(selected_assets))
input_col3.metric("Observations", len(returns))
input_col4.metric("Transaction cost", f"{transaction_cost_bps:.1f} bps")
input_col5.metric("Best utility strategy", metrics_scored.index[0])
st.info(source_note)

st.header("1. Asset universe and market data")
asset_table = pd.DataFrame(
    {
        "asset": selected_assets,
        "proxy role": [role_lookup.get(asset, "User-selected reserve proxy") for asset in selected_assets],
        "custom weight": [custom_weights.loc[asset] for asset in selected_assets],
        "stress shock": [shock.loc[asset] for asset in selected_assets],
    }
)
st.dataframe(asset_table.style.format({"custom weight": "{:.2%}", "stress shock": "{:.2%}"}), use_container_width=True)

with st.expander("Data quality and descriptive statistics", expanded=data_source.startswith("Real")):
    quality = data_quality_summary(prices, returns)
    st.dataframe(
        quality.style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "latest_price": "{:.2f}"}),
        use_container_width=True,
    )

normalize_prices = st.checkbox("Normalize price chart to 100 at first observation", value=True)
plot_prices = normalize_price_panel(prices) if normalize_prices else prices
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(line_chart(plot_prices, "Selected asset price indices", "Index"), use_container_width=True)
with chart_col2:
    st.plotly_chart(correlation_heatmap(returns), use_container_width=True)

st.header("2. Strategy performance comparison")
st.plotly_chart(line_chart(equity, "Cumulative wealth after transaction costs", "Wealth index"), use_container_width=True)
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
    use_container_width=True,
)

download_col1, download_col2, download_col3 = st.columns(3)
download_col1.download_button("Download strategy metrics CSV", as_csv_download(metrics_scored), "strategy_metrics.csv", "text/csv")
download_col2.download_button("Download strategy returns CSV", as_csv_download(strategy_returns), "strategy_returns.csv", "text/csv")
latest_weights_download = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
download_col3.download_button("Download latest weights CSV", as_csv_download(latest_weights_download), "latest_strategy_weights.csv", "text/csv")

st.header("3. Drawdown and allocation diagnostics")
st.plotly_chart(drawdown_chart(strategy_returns), use_container_width=True)
st.plotly_chart(weight_area_chart(weights_by_strategy[selected_strategy], f"{selected_strategy}: allocation over time"), use_container_width=True)

hist_col, turnover_col = st.columns(2)
with hist_col:
    st.subheader(f"{selected_strategy}: return distribution")
    fig = px.histogram(strategy_returns, x=selected_strategy, nbins=60)
    fig.update_layout(xaxis_title="Return", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
with turnover_col:
    st.subheader(f"{selected_strategy}: turnover over time")
    selected_turnover = portfolio_turnover(weights_by_strategy[selected_strategy])
    turnover_fig = go.Figure()
    turnover_fig.add_trace(go.Scatter(x=selected_turnover.index, y=selected_turnover, mode="lines", name="Turnover"))
    turnover_fig.update_layout(xaxis_title="Date", yaxis_title="Turnover")
    st.plotly_chart(turnover_fig, use_container_width=True)

st.header("4. Stress test")
st.write(
    "The stress test applies a one-day shock to the latest portfolio weights of each strategy. "
    "This is a transparent scenario engine, not a calibrated institutional stress model."
)
stress_col1, stress_col2 = st.columns(2)
with stress_col1:
    st.subheader("Shock vector")
    st.dataframe(shock.rename("one_day_asset_shock").to_frame().style.format("{:.2%}"), use_container_width=True)
with stress_col2:
    st.subheader("Strategy-level stress result")
    st.dataframe(stress_results.style.format({"one_day_stress_return": "{:.2%}", "one_day_stress_loss": "{:.2%}"}), use_container_width=True)
stress_fig = px.bar(stress_results.reset_index(), x="strategy", y="one_day_stress_loss", title=f"One-day stress loss under: {stress_name}")
st.plotly_chart(stress_fig, use_container_width=True)

st.header("5. DRL implementation status")
st.markdown(
    """
The repository already includes the **technical machinery for the reinforcement-learning extension**:

```text
src/envs/fx_reserve_env.py     -> Gymnasium portfolio-allocation environment
src/agents/train_ppo.py        -> PPO training entrypoint using Stable-Baselines3
src/portfolio/constraints.py   -> weight projection and concentration constraints
src/portfolio/metrics.py       -> risk and performance metrics
```

In the next stage, you train a PPO agent offline, export its test-period weights and returns, and add it
to this same dashboard as a sixth strategy. Training is intentionally not run inside Streamlit because
web apps should stay lightweight and reproducible.
"""
)

with st.expander("Show raw calculated data"):
    st.subheader("Strategy returns")
    st.dataframe(strategy_returns, use_container_width=True)
    st.subheader("Latest strategy weights")
    latest_weights = pd.DataFrame({name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}).T
    st.dataframe(latest_weights.style.format("{:.2%}"), use_container_width=True)
    st.subheader("Market returns")
    st.dataframe(returns, use_container_width=True)
