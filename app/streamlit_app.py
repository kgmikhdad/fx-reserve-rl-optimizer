"""Stable Streamlit dashboard for FX reserve portfolio analysis."""
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

from src.data.demo_data import demo_dataset  # noqa: E402
from src.portfolio.baselines import (  # noqa: E402
    equal_weight,
    portfolio_returns,
    rolling_mean_variance,
    rolling_minimum_variance,
    static_reserve_weights,
)
from src.portfolio.metrics import drawdown_series, metrics_table, portfolio_turnover  # noqa: E402

st.set_page_config(page_title="FX Reserve Portfolio Dashboard", page_icon="📊", layout="wide")

STRESS = {
    "Bond selloff": {"SHY": -0.002, "IEF": -0.010, "TLT": -0.030, "GLD": 0.006, "FXE": -0.004, "FXY": 0.003},
    "Dollar squeeze": {"SHY": 0.001, "IEF": 0.002, "TLT": 0.004, "GLD": -0.006, "FXE": -0.018, "FXY": -0.012},
    "Gold correction": {"SHY": 0.000, "IEF": 0.001, "TLT": 0.002, "GLD": -0.050, "FXE": -0.002, "FXY": -0.002},
    "Risk-off": {"SHY": 0.002, "IEF": 0.004, "TLT": 0.008, "GLD": 0.010, "FXE": -0.012, "FXY": 0.006},
}


@st.cache_data
def load_demo() -> pd.DataFrame:
    prices, _ = demo_dataset()
    return prices


def read_price_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    date_col = next((c for c in raw.columns if c.lower() in {"date", "datetime", "time"}), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    data = raw.set_index(date_col).apply(pd.to_numeric, errors="coerce")
    data = data.loc[data.index.notna()].sort_index().dropna(axis=1, how="all").ffill().dropna()
    data = data.loc[:, data.nunique() > 1]
    data.columns = [str(c).strip().upper() for c in data.columns]
    if data.shape[1] < 2 or len(data) < 60:
        raise ValueError("CSV must contain a date column, at least two price columns, and at least 60 complete rows.")
    return data


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Daily":
        return prices
    return prices.resample({"Weekly": "W-FRI", "Monthly": "ME"}[frequency]).last().dropna()


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def norm_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.divide(prices.iloc[0]).mul(100)


def constant_weights(weights: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.tile(weights.to_numpy(), (len(index), 1)), index=index, columns=weights.index)


def normalise(raw: dict[str, float], assets: list[str]) -> pd.Series:
    x = np.array([raw.get(a, 0.0) for a in assets], dtype=float)
    x = np.maximum(x, 0.0)
    x = np.ones(len(assets)) / len(assets) if np.isclose(x.sum(), 0.0) else x / x.sum()
    return pd.Series(x, index=assets)


def rebalance(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Every observation":
        return weights
    rule = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE"}[frequency]
    return weights.resample(rule).last().reindex(weights.index, method="ffill").fillna(weights.iloc[0])


def strategy_set(returns: pd.DataFrame, cost: float, window: int, max_w: float, risk_aversion: float, custom: pd.Series, freq: str):
    weights = {
        "Equal Weight": equal_weight(returns),
        "Static Benchmark": static_reserve_weights(returns),
        "Minimum Variance": rolling_minimum_variance(returns, window=window, max_weight=max_w),
        "Mean Variance": rolling_mean_variance(returns, window=window, risk_aversion=risk_aversion, max_weight=max_w),
        "Custom Portfolio": constant_weights(custom, returns.index),
    }
    weights = {name: rebalance(w, freq) for name, w in weights.items()}
    rets = pd.DataFrame({name: portfolio_returns(returns, w, transaction_cost_bps=cost) for name, w in weights.items()})
    return rets, weights


def score_metrics(rets: pd.DataFrame, weights: dict[str, pd.DataFrame], periods: int, lv: float, ld: float, lt: float) -> pd.DataFrame:
    m = metrics_table(rets)
    m["turnover"] = [portfolio_turnover(weights[n]).mean() * periods for n in m.index]
    m["reserve_score"] = m["annualized_return"] - lv * m["annualized_volatility"] - ld * m["max_drawdown"].abs() - lt * m["turnover"]
    return m.sort_values("reserve_score", ascending=False)


def line(df: pd.DataFrame, title: str, y: str) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y, legend_title=None, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def dd_frame(rets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: drawdown_series(rets[c]) for c in rets.columns})


def risk_contribution(weights: pd.Series, returns: pd.DataFrame, window: int) -> pd.DataFrame:
    sample = returns.tail(window)
    cov = sample.cov().to_numpy()
    w = weights.reindex(returns.columns).fillna(0).to_numpy()
    variance = float(w.T @ cov @ w)
    rc = np.zeros_like(w) if variance <= 0 else w * (cov @ w) / variance
    return pd.DataFrame({"weight": w, "risk_contribution": rc}, index=returns.columns)


def random_frontier(returns: pd.DataFrame, periods: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    mu = returns.mean().to_numpy() * periods
    cov = returns.cov().to_numpy() * periods
    rows = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(returns.shape[1]))
        r = float(w @ mu)
        v = float(np.sqrt(w.T @ cov @ w))
        rows.append({"return": r, "volatility": v, "sharpe": r / v if v > 0 else 0})
    return pd.DataFrame(rows)


def stress_vector(name: str, assets: list[str], custom: dict[str, float]) -> pd.Series:
    if name == "Custom shock":
        return pd.Series({a: custom.get(a, 0.0) / 100 for a in assets})
    return pd.Series({a: STRESS[name].get(a, 0.0) for a in assets})


def stress_table(weights: dict[str, pd.DataFrame], shock: pd.Series) -> pd.DataFrame:
    rows = []
    for name, w in weights.items():
        loss = -float((w.iloc[-1].reindex(shock.index).fillna(0) * shock).sum())
        rows.append({"strategy": name, "stress_loss": loss, "stress_return": -loss})
    return pd.DataFrame(rows).set_index("strategy").sort_values("stress_loss")


def csv(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


st.sidebar.title("Dashboard Controls")
mode = st.sidebar.radio("Data source", ["Demo data", "Upload price CSV"])
freq = st.sidebar.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"])
periods = {"Daily": 252, "Weekly": 52, "Monthly": 12}[freq]

if mode == "Demo data":
    prices_all = load_demo()
else:
    upload = st.sidebar.file_uploader("CSV with Date + price columns", type="csv")
    if upload is None:
        st.title("FX Reserve Portfolio Dashboard")
        st.info("Upload a price CSV to begin. Required format: Date column plus at least two asset price columns.")
        st.stop()
    try:
        prices_all = read_price_csv(upload)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

prices_all = resample_prices(prices_all, freq)
returns_all = price_returns(prices_all)
prices_all = prices_all.loc[returns_all.index]
assets = st.sidebar.multiselect("Assets", list(returns_all.columns), default=list(returns_all.columns))
if len(assets) < 2:
    st.stop()
start, end = st.sidebar.slider("Date window", returns_all.index.min().date(), returns_all.index.max().date(), (returns_all.index.min().date(), returns_all.index.max().date()))
prices = prices_all.loc[str(start):str(end), assets]
returns = returns_all.loc[str(start):str(end), assets]
if len(returns) < 60:
    st.error("Select a longer date window. At least 60 observations are required.")
    st.stop()

st.sidebar.divider()
cost = st.sidebar.slider("Transaction cost, bps", 0.0, 25.0, 1.0, 0.5)
window = st.sidebar.slider("Rolling window", 12, min(252, max(12, len(returns) // 2)), min(63, max(12, len(returns) // 3)))
max_w = st.sidebar.slider("Max asset weight", 0.20, 1.00, 0.60, 0.05)
risk_aversion = st.sidebar.slider("Risk aversion", 1.0, 50.0, 10.0)
rebal = st.sidebar.selectbox("Rebalancing", ["Every observation", "Weekly", "Monthly", "Quarterly"], index=2)

st.sidebar.divider()
st.sidebar.subheader("Custom Portfolio")
raw = {a: st.sidebar.slider(a, 0.0, 100.0, round(100 / len(assets), 1), 1.0) for a in assets}
custom_weights = normalise(raw, assets)

st.sidebar.divider()
lv = st.sidebar.slider("Volatility penalty", 0.0, 5.0, 1.0, 0.1)
ld = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lt = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

rets, weights = strategy_set(returns, cost, window, max_w, risk_aversion, custom_weights, rebal)
metrics = score_metrics(rets, weights, periods, lv, ld, lt)
equity = (1 + rets).cumprod()
best = metrics.index[0]
selected = st.sidebar.selectbox("Inspect", list(rets.columns), index=list(rets.columns).index(best))
scenario = st.sidebar.selectbox("Stress scenario", list(STRESS.keys()) + ["Custom shock"])
custom_shock = {a: st.sidebar.slider(f"{a} shock %", -15.0, 15.0, 0.0, 0.25) for a in assets} if scenario == "Custom shock" else {}
shock = stress_vector(scenario, assets, custom_shock)
stress = stress_table(weights, shock)

st.title("FX Reserve Portfolio Dashboard")
st.caption("Portfolio allocation, benchmark comparison, risk monitoring, and stress testing")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Assets", len(assets))
k2.metric("Observations", len(returns))
k3.metric("Best score", best)
k4.metric("Best Sharpe", f"{metrics['sharpe_ratio'].max():.2f}")
k5.metric("Worst drawdown", f"{metrics['max_drawdown'].min():.2%}")

t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Data", "Strategies", "Risk", "Stress", "Export"])

with t1:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(line(equity, "Cumulative wealth", "Wealth"), width="stretch")
    with c2:
        st.dataframe(metrics[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "turnover", "reserve_score"]].style.format({"annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}", "sharpe_ratio": "{:.2f}", "max_drawdown": "{:.2%}", "turnover": "{:.2f}", "reserve_score": "{:.3f}"}), width="stretch")

with t2:
    c1, c2 = st.columns(2)
    c1.plotly_chart(line(norm_prices(prices), "Price indices", "Index = 100"), width="stretch")
    c2.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", title="Correlation matrix"), width="stretch")
    quality = pd.DataFrame({"first_date": prices.apply(lambda x: x.first_valid_index().date()), "last_date": prices.apply(lambda x: x.last_valid_index().date()), "observations": returns.count(), "latest_price": prices.iloc[-1]})
    st.dataframe(quality, width="stretch")

with t3:
    fig = px.area(weights[selected], x=weights[selected].index, y=weights[selected].columns, title=f"{selected}: allocation")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    latest = pd.DataFrame({n: w.iloc[-1] for n, w in weights.items()}).T
    c1.dataframe(latest.style.format("{:.2%}"), width="stretch")
    frontier = random_frontier(returns, periods)
    c2.plotly_chart(px.scatter(frontier, x="volatility", y="return", color="sharpe", title="Random long-only frontier"), width="stretch")

with t4:
    c1, c2 = st.columns(2)
    c1.plotly_chart(line(dd_frame(rets), "Drawdown", "Drawdown"), width="stretch")
    c2.plotly_chart(line(rets.rolling(window).std().mul(np.sqrt(periods)).dropna(), "Rolling volatility", "Volatility"), width="stretch")
    st.dataframe(risk_contribution(weights[selected].iloc[-1], returns, window).style.format({"weight": "{:.2%}", "risk_contribution": "{:.2%}"}), width="stretch")

with t5:
    c1, c2 = st.columns(2)
    c1.dataframe(shock.rename("shock").to_frame().style.format("{:.2%}"), width="stretch")
    c2.dataframe(stress.style.format({"stress_loss": "{:.2%}", "stress_return": "{:.2%}"}), width="stretch")
    st.plotly_chart(px.bar(stress.reset_index(), x="strategy", y="stress_loss", title=f"Stress loss: {scenario}"), width="stretch")

with t6:
    latest = pd.DataFrame({n: w.iloc[-1] for n, w in weights.items()}).T
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Metrics", csv(metrics), "metrics.csv", "text/csv")
    d2.download_button("Returns", csv(rets), "strategy_returns.csv", "text/csv")
    d3.download_button("Weights", csv(latest), "latest_weights.csv", "text/csv")
    d4.download_button("Clean prices", csv(prices), "clean_prices.csv", "text/csv")
    st.code("Date,BIL,SHY,IEF,TLT,GLD\n2018-01-02,91.10,82.40,116.50,126.20,125.10\n2018-01-03,91.12,82.45,116.70,126.50,125.60", language="csv")
