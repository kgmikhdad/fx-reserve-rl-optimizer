"""Synthetic-only FX reserve portfolio dashboard.

Self-contained Streamlit app: no external data, no yfinance, no Stooq,
no upload mode, no RL-training imports, and no local src imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FX Reserve Portfolio Command Center", page_icon="📊", layout="wide")

ASSET_INFO = {
    "CASH_USD": "USD cash / T-bills",
    "SHORT_UST": "Short US Treasuries",
    "INT_UST": "Intermediate US Treasuries",
    "LONG_UST": "Long US Treasuries",
    "GOLD": "Gold reserve sleeve",
    "EUR_RESERVE": "Euro reserve sleeve",
    "JPY_RESERVE": "Yen reserve sleeve",
}

STATIC_BASE = {
    "CASH_USD": 0.20,
    "SHORT_UST": 0.25,
    "INT_UST": 0.25,
    "LONG_UST": 0.10,
    "GOLD": 0.10,
    "EUR_RESERVE": 0.07,
    "JPY_RESERVE": 0.03,
}

SCENARIOS = {
    "Base reserve market": ([0.025, 0.032, 0.038, 0.045, 0.040, 0.020, 0.015], [0.004, 0.020, 0.045, 0.105, 0.155, 0.085, 0.095]),
    "Inflation shock": ([0.040, 0.018, -0.005, -0.035, 0.065, 0.010, 0.005], [0.006, 0.035, 0.075, 0.155, 0.180, 0.105, 0.115]),
    "Dollar squeeze": ([0.045, 0.040, 0.030, 0.020, 0.015, -0.035, -0.045], [0.006, 0.025, 0.055, 0.120, 0.170, 0.140, 0.155]),
    "Risk-off regime": ([0.035, 0.050, 0.065, 0.075, 0.075, -0.020, 0.020], [0.005, 0.025, 0.050, 0.115, 0.190, 0.130, 0.145]),
}

STRESS = {
    "Parallel yield shock": {"CASH_USD": 0.000, "SHORT_UST": -0.003, "INT_UST": -0.015, "LONG_UST": -0.055, "GOLD": 0.004, "EUR_RESERVE": -0.006, "JPY_RESERVE": -0.004},
    "Dollar squeeze": {"CASH_USD": 0.001, "SHORT_UST": 0.002, "INT_UST": 0.003, "LONG_UST": 0.006, "GOLD": -0.010, "EUR_RESERVE": -0.035, "JPY_RESERVE": -0.030},
    "Gold correction": {"CASH_USD": 0.000, "SHORT_UST": 0.000, "INT_UST": 0.001, "LONG_UST": 0.002, "GOLD": -0.080, "EUR_RESERVE": -0.004, "JPY_RESERVE": -0.004},
    "Risk-off liquidation": {"CASH_USD": 0.001, "SHORT_UST": 0.004, "INT_UST": 0.009, "LONG_UST": 0.018, "GOLD": 0.018, "EUR_RESERVE": -0.020, "JPY_RESERVE": 0.012},
}


def corr_matrix() -> np.ndarray:
    return np.array([
        [1.00, 0.20, 0.15, 0.10, 0.00, -0.05, -0.05],
        [0.20, 1.00, 0.70, 0.45, 0.05, -0.10, -0.05],
        [0.15, 0.70, 1.00, 0.75, 0.10, -0.15, -0.10],
        [0.10, 0.45, 0.75, 1.00, 0.15, -0.20, -0.15],
        [0.00, 0.05, 0.10, 0.15, 1.00, 0.15, 0.10],
        [-0.05, -0.10, -0.15, -0.20, 0.15, 1.00, 0.35],
        [-0.05, -0.05, -0.10, -0.15, 0.10, 0.35, 1.00],
    ])


@st.cache_data(show_spinner=False)
def synthetic_prices(scenario: str, years: int, seed: int) -> pd.DataFrame:
    assets = list(ASSET_INFO)
    n = years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    mu_annual, vol_annual = SCENARIOS[scenario]
    mu = np.array(mu_annual) / 252.0
    vol = np.array(vol_annual) / np.sqrt(252.0)
    cov = np.outer(vol, vol) * corr_matrix()
    rng = np.random.default_rng(seed)
    returns = rng.multivariate_normal(np.zeros(len(assets)), cov, size=n) + mu
    event_days = rng.choice(np.arange(40, n - 1), size=max(2, years), replace=False)
    for d in event_days:
        returns[d] += rng.normal(-0.002, 0.004, len(assets))
        returns[d, 3] += rng.normal(-0.014, 0.010)
        returns[d, 4] += rng.normal(0.008, 0.012)
        returns[d, 5:] += rng.normal(-0.009, 0.010, 2)
    returns = np.clip(returns, -0.20, 0.20)
    return 100.0 * pd.DataFrame(1.0 + returns, index=dates, columns=assets).cumprod()


def resample(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Daily":
        return prices
    return prices.resample({"Weekly": "W-FRI", "Monthly": "ME"}[frequency]).last().dropna()


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def periods_per_year(frequency: str) -> int:
    return {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]


def cap_norm(x, assets: list[str], cap: float) -> pd.Series:
    arr = np.maximum(np.nan_to_num(np.asarray(x, dtype=float)), 0.0)
    arr = np.ones(len(assets)) / len(assets) if arr.sum() <= 0 else arr / arr.sum()
    cap = max(cap, 1.0 / len(assets))
    for _ in range(len(arr) + 1):
        over = arr > cap
        if not over.any():
            break
        arr[over] = cap
        free = ~over
        residual = 1.0 - arr[over].sum()
        if residual <= 0 or not free.any():
            arr = arr / arr.sum()
            break
        arr[free] = residual * arr[free] / arr[free].sum() if arr[free].sum() > 0 else residual / free.sum()
    return pd.Series(arr / arr.sum(), index=assets)


def const_frame(w: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.tile(w.to_numpy(), (len(index), 1)), index=index, columns=w.index)


def static_w(assets: list[str], cap: float) -> pd.Series:
    return cap_norm(pd.Series({a: STATIC_BASE.get(a, 0) for a in assets}), assets, cap)


def rolling_w(returns: pd.DataFrame, method: str, lookback: int, cap: float, risk_aversion: float) -> pd.DataFrame:
    assets = list(returns.columns)
    fallback = pd.Series(1.0 / len(assets), index=assets)
    rows = []
    for i in range(len(returns)):
        if i < lookback:
            rows.append(fallback)
            continue
        sample = returns.iloc[i - lookback : i]
        cov = sample.cov().to_numpy() + np.eye(len(assets)) * 1e-8
        vol = sample.std(ddof=0).replace(0, np.nan)
        if method == "Inverse Volatility":
            raw = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        elif method == "Minimum Variance":
            raw = np.linalg.pinv(cov) @ np.ones(len(assets))
        elif method == "Mean Variance":
            raw = np.linalg.pinv(cov) @ (sample.mean().to_numpy() / risk_aversion)
        else:
            momentum = (1.0 + sample).prod().to_numpy() - 1.0
            raw = np.maximum(momentum, 0) / (vol.fillna(1).to_numpy() + 1e-8)
        rows.append(cap_norm(raw, assets, cap))
    return pd.DataFrame(rows, index=returns.index)


def rebalance(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "Every observation":
        return weights
    rule = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE"}[frequency]
    return weights.resample(rule).last().reindex(weights.index, method="ffill").fillna(weights.iloc[0])


def port_ret(returns: pd.DataFrame, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    w = weights.reindex(returns.index).ffill().fillna(weights.iloc[0])
    gross = (w.shift(1).fillna(w.iloc[0]) * returns).sum(axis=1)
    costs = w.diff().abs().sum(axis=1).fillna(0) * cost_bps / 10000.0
    return gross - costs


def dd(series: pd.Series) -> pd.Series:
    wealth = (1 + series).cumprod()
    return wealth / wealth.cummax() - 1


def metrics_table(rets: pd.DataFrame, weights: dict[str, pd.DataFrame], periods: int, lv: float, ld: float, lt: float) -> pd.DataFrame:
    rows = []
    for name in rets.columns:
        r = rets[name].dropna()
        ann_return = (1 + r).prod() ** (periods / len(r)) - 1
        ann_vol = r.std(ddof=0) * np.sqrt(periods)
        downside = r[r < 0].std(ddof=0) * np.sqrt(periods)
        max_dd = dd(r).min()
        var95 = r.quantile(0.05)
        es95 = r[r <= var95].mean() if (r <= var95).any() else var95
        turnover = weights[name].diff().abs().sum(axis=1).mean() * periods
        score = ann_return - lv * ann_vol - ld * abs(max_dd) - lt * turnover
        rows.append({
            "strategy": name,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
            "sortino": ann_return / downside if downside > 0 else np.nan,
            "max_drawdown": max_dd,
            "var_95": var95,
            "expected_shortfall_95": es95,
            "turnover": turnover,
            "reserve_score": score,
        })
    return pd.DataFrame(rows).set_index("strategy").sort_values("reserve_score", ascending=False)


def build_strategies(returns: pd.DataFrame, custom: pd.Series, lookback: int, cap: float, risk_aversion: float, rebal: str, cost: float):
    assets = list(returns.columns)
    weights = {
        "Equal Weight": const_frame(pd.Series(1 / len(assets), index=assets), returns.index),
        "Reserve Benchmark": const_frame(static_w(assets, cap), returns.index),
        "Inverse Volatility": rolling_w(returns, "Inverse Volatility", lookback, cap, risk_aversion),
        "Minimum Variance": rolling_w(returns, "Minimum Variance", lookback, cap, risk_aversion),
        "Mean Variance": rolling_w(returns, "Mean Variance", lookback, cap, risk_aversion),
        "Momentum Defensive": rolling_w(returns, "Momentum Defensive", lookback, cap, risk_aversion),
        "Custom Portfolio": const_frame(custom, returns.index),
    }
    weights = {k: rebalance(v, rebal) for k, v in weights.items()}
    strategy_returns = pd.DataFrame({k: port_ret(returns, v, cost) for k, v in weights.items()}, index=returns.index)
    return strategy_returns, weights


def line_chart(df: pd.DataFrame, title: str, y: str) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, yaxis_title=y, xaxis_title=None, legend_title=None, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def risk_contrib(weights: pd.Series, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    sample = returns.tail(lookback)
    cov = sample.cov().to_numpy()
    w = weights.reindex(returns.columns).fillna(0).to_numpy()
    variance = float(w.T @ cov @ w)
    rc = np.zeros_like(w) if variance <= 0 else w * (cov @ w) / variance
    return pd.DataFrame({"weight": w, "risk_contribution": rc}, index=returns.columns)


def frontier(returns: pd.DataFrame, periods: int, cap: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 99)
    assets = list(returns.columns)
    mu = returns.mean().to_numpy() * periods
    cov = returns.cov().to_numpy() * periods
    rows = []
    for _ in range(600):
        w = cap_norm(rng.dirichlet(np.ones(len(assets))), assets, cap).to_numpy()
        ar = float(w @ mu)
        av = float(np.sqrt(w.T @ cov @ w))
        rows.append({"annual_return": ar, "annual_volatility": av, "sharpe": ar / av if av > 0 else np.nan})
    return pd.DataFrame(rows)


def monthly_heatmap(series: pd.Series) -> pd.DataFrame:
    m = (1 + series).resample("ME").prod() - 1
    table = pd.DataFrame({"return": m, "year": m.index.year, "month": m.index.strftime("%b")})
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return table.pivot(index="year", columns="month", values="return").reindex(columns=months)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


st.sidebar.header("Simulation")
scenario = st.sidebar.selectbox("Market scenario", list(SCENARIOS))
years = st.sidebar.slider("Synthetic history", 3, 20, 10)
seed = st.sidebar.number_input("Seed", 1, 9999, 42)
frequency = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
periods = periods_per_year(frequency)

prices_all = resample(synthetic_prices(scenario, years, int(seed)), frequency)
returns_all = returns_from_prices(prices_all)
prices_all = prices_all.loc[returns_all.index]
assets = st.sidebar.multiselect("Assets", list(ASSET_INFO), default=list(ASSET_INFO))
if len(assets) < 2:
    st.stop()

start, end = st.sidebar.slider("Window", returns_all.index.min().date(), returns_all.index.max().date(), (returns_all.index.min().date(), returns_all.index.max().date()))
prices = prices_all.loc[str(start):str(end), assets]
returns = returns_all.loc[str(start):str(end), assets]
if len(returns) < 60:
    st.error("Select a longer window.")
    st.stop()

st.sidebar.header("Portfolio Rules")
min_cap = float(np.ceil((1.0 / len(assets)) * 100) / 100)
cap = st.sidebar.slider("Max asset weight", min_cap, 1.0, max(0.60, min_cap), 0.05)
lookback_max = min(252, max(24, len(returns) // 2))
lookback = st.sidebar.slider("Risk window", 12, lookback_max, min(63, lookback_max))
risk_aversion = st.sidebar.slider("Risk aversion", 1.0, 50.0, 10.0)
rebal = st.sidebar.selectbox("Rebalancing", ["Every observation", "Weekly", "Monthly", "Quarterly"], index=2)
cost = st.sidebar.slider("Cost, bps", 0.0, 25.0, 1.0, 0.5)

st.sidebar.header("Custom Portfolio")
raw_custom = {a: st.sidebar.slider(a, 0.0, 100.0, round(100 / len(assets), 1), 1.0) for a in assets}
custom = cap_norm(pd.Series(raw_custom), assets, cap)

st.sidebar.header("Score Weights")
lv = st.sidebar.slider("Vol penalty", 0.0, 5.0, 1.0, 0.1)
ld = st.sidebar.slider("Drawdown penalty", 0.0, 10.0, 2.0, 0.25)
lt = st.sidebar.slider("Turnover penalty", 0.0, 1.0, 0.10, 0.01)

rets, weights = build_strategies(returns, custom, lookback, cap, risk_aversion, rebal, cost)
metrics = metrics_table(rets, weights, periods, lv, ld, lt)
equity = (1 + rets).cumprod()
best = metrics.index[0]
selected = st.sidebar.selectbox("Inspect", list(rets.columns), index=list(rets.columns).index(best))

st.sidebar.header("Stress")
stress_name = st.sidebar.selectbox("Scenario", list(STRESS) + ["Custom"])
if stress_name == "Custom":
    shock = pd.Series({a: st.sidebar.slider(f"{a} shock %", -20.0, 20.0, 0.0, 0.25) / 100 for a in assets})
else:
    shock = pd.Series({a: STRESS[stress_name].get(a, 0.0) for a in assets})
stress_rows = []
for name, w in weights.items():
    sr = float((w.iloc[-1].reindex(assets).fillna(0) * shock).sum())
    stress_rows.append({"strategy": name, "stress_return": sr, "stress_loss": -sr})
stress_df = pd.DataFrame(stress_rows).set_index("strategy").sort_values("stress_loss", ascending=False)

st.title("FX Reserve Portfolio Command Center")
st.caption("Synthetic allocation, risk, stress testing, and governance analytics")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Scenario", scenario)
k2.metric("Assets", len(assets))
k3.metric("Observations", len(returns))
k4.metric("Top strategy", best)
k5.metric("Top Sharpe", f"{metrics['sharpe'].max():.2f}")
k6.metric("Worst DD", f"{metrics['max_drawdown'].min():.2%}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Executive", "Allocation", "Risk", "Stress", "Scenario Lab", "Export"])

fmt = {"annual_return": "{:.2%}", "annual_volatility": "{:.2%}", "sharpe": "{:.2f}", "sortino": "{:.2f}", "max_drawdown": "{:.2%}", "var_95": "{:.2%}", "expected_shortfall_95": "{:.2%}", "turnover": "{:.2f}", "reserve_score": "{:.3f}"}

with tab1:
    c1, c2 = st.columns([1.35, 1.0])
    c1.plotly_chart(line_chart(equity, "Cumulative wealth", "Wealth"), width="stretch")
    c2.dataframe(metrics.style.format(fmt), width="stretch")
    scatter = metrics.reset_index()
    fig = px.scatter(scatter, x="annual_volatility", y="annual_return", color="reserve_score", size=scatter["max_drawdown"].abs() + 0.01, hover_name="strategy", title="Risk-return map")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

with tab2:
    c1, c2 = st.columns([1.35, 1.0])
    alloc = px.area(weights[selected], x=weights[selected].index, y=weights[selected].columns, title=f"{selected}: allocation")
    alloc.update_layout(yaxis_title="Weight", xaxis_title=None, legend_title=None)
    c1.plotly_chart(alloc, width="stretch")
    latest = pd.DataFrame({name: w.iloc[-1] for name, w in weights.items()}).T
    c2.dataframe(latest.style.format("{:.2%}"), width="stretch")
    concentration = pd.DataFrame({"HHI": {n: float((w.iloc[-1] ** 2).sum()) for n, w in weights.items()}, "largest_sleeve": {n: float(w.iloc[-1].max()) for n, w in weights.items()}, "active_sleeves": {n: int((w.iloc[-1] > 0.01).sum()) for n, w in weights.items()}})
    st.dataframe(concentration.style.format({"HHI": "{:.3f}", "largest_sleeve": "{:.2%}"}), width="stretch")

with tab3:
    c1, c2 = st.columns(2)
    c1.plotly_chart(line_chart(pd.DataFrame({n: dd(rets[n]) for n in rets.columns}), "Drawdown", "Drawdown"), width="stretch")
    c2.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", title="Asset correlation"), width="stretch")
    roll = rets[selected].rolling(min(lookback, len(rets) - 1))
    rolling_vol = roll.std(ddof=0) * np.sqrt(periods)
    rolling_ret = roll.mean() * periods
    c3, c4 = st.columns(2)
    c3.plotly_chart(line_chart(pd.DataFrame({"rolling_volatility": rolling_vol}), "Rolling volatility", "Volatility"), width="stretch")
    c4.plotly_chart(line_chart(pd.DataFrame({"rolling_sharpe": rolling_ret / rolling_vol}), "Rolling Sharpe", "Sharpe"), width="stretch")
    c5, c6 = st.columns(2)
    c5.dataframe(risk_contrib(weights[selected].iloc[-1], returns, lookback).style.format({"weight": "{:.2%}", "risk_contribution": "{:.2%}"}), width="stretch")
    c6.plotly_chart(px.imshow(monthly_heatmap(rets[selected]), text_auto=".1%", aspect="auto", title="Monthly returns"), width="stretch")

with tab4:
    c1, c2 = st.columns([1.0, 1.2])
    c1.dataframe(shock.rename("asset_shock").to_frame().style.format("{:.2%}"), width="stretch")
    c2.dataframe(stress_df.style.format({"stress_return": "{:.2%}", "stress_loss": "{:.2%}"}), width="stretch")
    fig = px.bar(stress_df.reset_index(), x="strategy", y="stress_loss", title=f"Stress loss: {stress_name}")
    fig.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

with tab5:
    front = frontier(returns, periods, cap, int(seed))
    fig = px.scatter(front, x="annual_volatility", y="annual_return", color="sharpe", title="Simulated long-only efficient frontier")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    governance = pd.DataFrame({"control": ["data_mode", "max_weight", "risk_window", "rebalancing", "transaction_cost", "score_formula"], "setting": ["synthetic_only", f"{cap:.0%}", lookback, rebal, f"{cost:.1f} bps", "return - vol - drawdown - turnover penalties"]})
    st.dataframe(governance, width="stretch")

with tab6:
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("Metrics", metrics.to_csv().encode("utf-8"), "strategy_metrics.csv", "text/csv")
    e2.download_button("Returns", rets.to_csv().encode("utf-8"), "strategy_returns.csv", "text/csv")
    e3.download_button("Prices", prices.to_csv().encode("utf-8"), "synthetic_prices.csv", "text/csv")
    e4.download_button("Latest weights", latest.to_csv().encode("utf-8"), "latest_weights.csv", "text/csv")
    st.dataframe(pd.DataFrame({"asset": assets, "role": [ASSET_INFO[a] for a in assets]}), width="stretch")
