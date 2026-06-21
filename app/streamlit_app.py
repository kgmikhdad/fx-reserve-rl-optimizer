"""Interactive Streamlit dashboard for the FX reserve RL optimizer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.demo_data import ASSET_ROLES, demo_dataset
from src.portfolio.baselines import (
    equal_weight,
    portfolio_returns,
    rolling_mean_variance,
    rolling_minimum_variance,
    static_reserve_weights,
)
from src.portfolio.metrics import drawdown_series, metrics_table, portfolio_turnover

DISCLAIMER = (
    "This is a public-data and synthetic-data research prototype. It is not investment advice, "
    "not a live reserve-management system, and does not represent the portfolio, policy, or internal "
    "systems of any central bank, the BIS, or any financial institution."
)

STRESS_LIBRARY: dict[str, dict[str, float]] = {
    "Global bond selloff": {
        "SHY": -0.002,
        "IEF": -0.010,
        "TLT": -0.030,
        "GLD": 0.006,
        "FXE": -0.004,
        "FXY": 0.003,
    },
    "USD liquidity squeeze": {
        "SHY": 0.001,
        "IEF": 0.002,
        "TLT": 0.004,
        "GLD": -0.006,
        "FXE": -0.018,
        "FXY": -0.012,
    },
    "Gold selloff": {
        "SHY": 0.000,
        "IEF": 0.001,
        "TLT": 0.002,
        "GLD": -0.050,
        "FXE": -0.002,
        "FXY": -0.002,
    },
    "Broad risk-off shock": {
        "SHY": 0.002,
        "IEF": 0.004,
        "TLT": 0.008,
        "GLD": 0.010,
        "FXE": -0.012,
        "FXY": 0.006,
    },
}

st.set_page_config(page_title="FX Reserve Portfolio Optimizer", layout="wide")


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deterministic demo market data for immediate deployment."""
    return demo_dataset()


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
    return pd.DataFrame(
        np.tile(weights.to_numpy(), (len(index), 1)),
        index=index,
        columns=weights.index,
    )


def build_strategy_set(
    returns: pd.DataFrame,
    transaction_cost_bps: float,
    rolling_window: int,
    max_weight: float,
    risk_aversion: float,
    custom_weights: pd.Series,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    strategies: dict[str, pd.DataFrame] = {
        "Equal Weight": equal_weight(returns),
        "Static Reserve": static_reserve_weights(returns),
        "Minimum Variance": rolling_minimum_variance(
            returns,
            window=rolling_window,
            max_weight=max_weight,
        ),
        "Mean Variance": rolling_mean_variance(
            returns,
            window=rolling_window,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
        ),
        "Your Custom Portfolio": constant_weight_frame(custom_weights, returns.index),
    }
    strategy_returns = pd.DataFrame(
        {
            name: portfolio_returns(
                returns,
                weights,
                transaction_cost_bps=transaction_cost_bps,
            )
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
    enriched["annualized_turnover"] = [
        annualized_turnover(weights_by_strategy[name]) for name in enriched.index
    ]
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


def stress_test_table(
    weights_by_strategy: dict[str, pd.DataFrame],
    shock: pd.Series,
) -> pd.DataFrame:
    rows = []
    for strategy_name, weights in weights_by_strategy.items():
        last_weights = weights.iloc[-1].reindex(shock.index).fillna(0.0)
        stress_return = float((last_weights * shock).sum())
        rows.append(
            {
                "strategy": strategy_name,
                "one_day_stress_return": stress_return,
                "one_day_stress_loss": -stress_return,
            }
        )
    return pd.DataFrame(rows).set_index("strategy").sort_values("one_day_stress_loss")


def format_pct_table(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler:
    return df.style.format({col: "{:.2%}" for col in columns if col in df.columns})


prices_all, returns_all = load_demo_data()

st.title("FX Reserve Portfolio Optimizer")
st.caption("Interactive reserve-allocation simulator and DRL project prototype")
st.warning(DISCLAIMER)

st.sidebar.header("Simulation controls")

asset_options = list(returns_all.columns)
selected_assets = st.sidebar.multiselect(
    "Assets included in the reserve universe",
    asset_options,
    default=asset_options,
)
if len(selected_assets) < 2:
    st.sidebar.error("Select at least two assets to run the simulator.")
    st.stop()

min_date = returns_all.index.min().date()
max_date = returns_all.index.max().date()
start_date, end_date = st.sidebar.slider(
    "Backtest date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

transaction_cost_bps = st.sidebar.slider(
    "Transaction cost per unit of turnover, in basis points",
    min_value=0.0,
    max_value=25.0,
    value=1.0,
    step=0.5,
)
rolling_window = st.sidebar.slider("Rolling estimation window, trading days", 21, 252, 63, 21)
max_weight = st.sidebar.slider("Maximum weight per asset for optimized baselines", 0.20, 1.00, 0.60, 0.05)
risk_aversion = st.sidebar.slider("Mean-variance risk aversion", 1.0, 50.0, 10.0, 1.0)

st.sidebar.header("Reserve utility preferences")
lambda_vol = st.sidebar.slider("Volatility penalty", 0.0, 5.0, 1.0, 0.1)
lambda_drawdown = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lambda_turnover = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

prices = prices_all.loc[str(start_date) : str(end_date), selected_assets]
returns = returns_all.loc[str(start_date) : str(end_date), selected_assets]
if len(returns) < rolling_window + 5:
    st.error("The selected date range is too short for the chosen rolling window.")
    st.stop()

st.sidebar.header("Custom reserve portfolio")
st.sidebar.caption("Set raw weights. The app normalises them to sum to 100%.")
raw_custom_weights = {
    asset: st.sidebar.slider(
        f"{asset} raw weight",
        min_value=0.0,
        max_value=100.0,
        value=round(100.0 / len(selected_assets), 1),
        step=1.0,
    )
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
)
metrics = metrics_table(strategy_returns)
metrics_scored = add_turnover_and_score(
    metrics,
    weights_by_strategy,
    lambda_vol=lambda_vol,
    lambda_drawdown=lambda_drawdown,
    lambda_turnover=lambda_turnover,
)
equity = (1.0 + strategy_returns).cumprod()

selected_strategy = st.sidebar.selectbox(
    "Strategy to inspect",
    list(strategy_returns.columns),
    index=list(strategy_returns.columns).index("Static Reserve")
    if "Static Reserve" in strategy_returns.columns
    else 0,
)

stress_name = st.sidebar.selectbox(
    "One-day stress scenario",
    list(STRESS_LIBRARY.keys()) + ["Custom one-day shock"],
)
custom_shocks_pct: dict[str, float] = {}
if stress_name == "Custom one-day shock":
    st.sidebar.caption("Custom shock is in percent return for one day.")
    for asset in selected_assets:
        custom_shocks_pct[asset] = st.sidebar.slider(
            f"{asset} shock (%)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
        )
shock = stress_vector(stress_name, selected_assets, custom_shocks_pct)
stress_results = stress_test_table(weights_by_strategy, shock)

st.header("What this project actually does")
st.markdown(
    """
This app is a **reserve-portfolio allocation simulator**. It lets you change the portfolio universe,
backtest window, transaction costs, optimization constraints, risk preferences, custom weights, and
stress assumptions. Then it compares several allocation rules:

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

input_col1, input_col2, input_col3, input_col4 = st.columns(4)
input_col1.metric("Assets selected", len(selected_assets))
input_col2.metric("Backtest days", len(returns))
input_col3.metric("Transaction cost", f"{transaction_cost_bps:.1f} bps")
input_col4.metric("Best utility strategy", metrics_scored.index[0])

st.header("1. Asset universe and market data")
asset_table = pd.DataFrame(
    {
        "asset": selected_assets,
        "proxy role": [ASSET_ROLES.get(asset, "User-selected reserve proxy") for asset in selected_assets],
        "custom weight": [custom_weights.loc[asset] for asset in selected_assets],
        "stress shock": [shock.loc[asset] for asset in selected_assets],
    }
)
st.dataframe(
    asset_table.style.format({"custom weight": "{:.2%}", "stress shock": "{:.2%}"}),
    use_container_width=True,
)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(line_chart(prices, "Selected asset price indices", "Index"), use_container_width=True)
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

st.header("3. Drawdown and allocation diagnostics")
st.plotly_chart(drawdown_chart(strategy_returns), use_container_width=True)
st.plotly_chart(
    weight_area_chart(weights_by_strategy[selected_strategy], f"{selected_strategy}: allocation over time"),
    use_container_width=True,
)

hist_col, turnover_col = st.columns(2)
with hist_col:
    st.subheader(f"{selected_strategy}: daily return distribution")
    fig = px.histogram(strategy_returns, x=selected_strategy, nbins=60)
    fig.update_layout(xaxis_title="Daily return", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
with turnover_col:
    st.subheader(f"{selected_strategy}: turnover over time")
    selected_turnover = portfolio_turnover(weights_by_strategy[selected_strategy])
    turnover_fig = go.Figure()
    turnover_fig.add_trace(
        go.Scatter(x=selected_turnover.index, y=selected_turnover, mode="lines", name="Turnover")
    )
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
    st.dataframe(
        stress_results.style.format(
            {
                "one_day_stress_return": "{:.2%}",
                "one_day_stress_loss": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

stress_fig = px.bar(
    stress_results.reset_index(),
    x="strategy",
    y="one_day_stress_loss",
    title=f"One-day stress loss under: {stress_name}",
)
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
    latest_weights = pd.DataFrame(
        {name: weights.iloc[-1] for name, weights in weights_by_strategy.items()}
    ).T
    st.dataframe(latest_weights.style.format("{:.2%}"), use_container_width=True)
