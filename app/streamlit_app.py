"""Streamlit dashboard for the FX reserve RL optimizer."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.demo_data import ASSET_ROLES, demo_dataset
from src.portfolio.baselines import run_all_baselines, static_reserve_weights
from src.portfolio.metrics import drawdown_series, metrics_table
from src.stress.stress_scenarios import default_stress_scenarios

DISCLAIMER = (
    "This is a public-data and synthetic-data research prototype. It is not investment advice, "
    "not a live reserve-management system, and does not represent the portfolio, policy, or internal "
    "systems of any central bank, the BIS, or any financial institution."
)

st.set_page_config(page_title="FX Reserve Portfolio Optimizer", layout="wide")


@st.cache_data
def load_demo_results() -> dict[str, pd.DataFrame]:
    prices, returns = demo_dataset()
    strategy_returns, weights = run_all_baselines(returns)
    equity = (1.0 + strategy_returns).cumprod()
    metrics = metrics_table(strategy_returns)
    return {
        "prices": prices,
        "returns": returns,
        "strategy_returns": strategy_returns,
        "equity": equity,
        "metrics": metrics,
        "weights": static_reserve_weights(returns),
        "stress": default_stress_scenarios(),
    }


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, legend_title=None)
    return fig


def drawdown_chart(strategy_returns: pd.DataFrame) -> go.Figure:
    dd = pd.DataFrame({col: drawdown_series(strategy_returns[col]) for col in strategy_returns.columns})
    return line_chart(dd, "Drawdown comparison", "Drawdown")


def weight_area_chart(weights: pd.DataFrame) -> go.Figure:
    fig = px.area(weights, x=weights.index, y=weights.columns, title="Static reserve benchmark weights")
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio weight", legend_title=None)
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Asset return correlation matrix")
    return fig


data = load_demo_results()

st.title("FX Reserve Portfolio Optimizer")
st.caption("A deep reinforcement learning decision-support prototype for stylised reserve allocation")
st.warning(DISCLAIMER)

st.sidebar.header("Controls")
selected_strategy = st.sidebar.selectbox(
    "Strategy to inspect",
    list(data["strategy_returns"].columns),
    index=1,
)
show_raw = st.sidebar.checkbox("Show raw data tables", value=False)

st.header("1. Project motivation")
st.write(
    "Foreign-exchange reserve management is not ordinary return maximisation. "
    "A reserve portfolio must balance safety, liquidity, diversification, controlled drawdown, "
    "and return. This project compares transparent benchmark allocation rules with a constrained "
    "reinforcement-learning environment that can later be trained using PPO or similar algorithms."
)

st.header("2. Asset universe")
st.dataframe(
    pd.DataFrame({"asset": list(ASSET_ROLES.keys()), "proxy role": list(ASSET_ROLES.values())}),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(line_chart(data["prices"], "Demo asset price indices", "Index"), use_container_width=True)
with col2:
    st.plotly_chart(correlation_heatmap(data["returns"]), use_container_width=True)

st.header("3. Benchmark strategy comparison")
st.plotly_chart(line_chart(data["equity"], "Cumulative wealth", "Wealth index"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Risk and performance metrics")
    st.dataframe(data["metrics"].style.format("{:.4f}"), use_container_width=True)
with right:
    st.subheader(f"{selected_strategy}: return distribution")
    fig = px.histogram(data["strategy_returns"], x=selected_strategy, nbins=60)
    st.plotly_chart(fig, use_container_width=True)

st.header("4. Drawdown and allocation diagnostics")
st.plotly_chart(drawdown_chart(data["strategy_returns"]), use_container_width=True)
st.plotly_chart(weight_area_chart(data["weights"]), use_container_width=True)

st.header("5. Stress-test matrix")
st.write("Illustrative scenario losses. These are dashboard placeholders, not calibrated institutional stress tests.")
st.dataframe(data["stress"], use_container_width=True)

st.header("6. Reinforcement-learning implementation")
st.markdown(
    """
The repository includes a custom `FXReservePortfolioEnv` compatible with Gymnasium and a PPO training script using Stable-Baselines3.

The RL design is:

```text
state_t  = recent returns, volatility, momentum, wealth, drawdown
action_t = raw vector projected into feasible long-only portfolio weights
reward_t = portfolio return - volatility penalty - drawdown penalty - turnover penalty - concentration penalty
```

Training is intentionally kept outside the Streamlit app. The app is for inspection and presentation; model training should be run separately and the lightweight outputs can then be committed to `reports/tables/`.
"""
)

st.header("7. Model limitations")
st.markdown(
    """
- The current dashboard uses deterministic demo data so it can deploy immediately.
- Public ETF and currency proxies are not actual reserve assets.
- RL models can overfit historical market regimes.
- Reward design materially affects learned allocation behaviour.
- A production-grade reserve-management system would need mandate-specific constraints, liquidity modelling, governance checks, and human approval layers.
"""
)

if show_raw:
    st.header("Raw demo data")
    st.subheader("Returns")
    st.dataframe(data["returns"], use_container_width=True)
    st.subheader("Strategy returns")
    st.dataframe(data["strategy_returns"], use_container_width=True)
