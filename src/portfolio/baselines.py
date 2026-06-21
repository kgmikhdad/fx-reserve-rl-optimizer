"""Transparent benchmark allocation strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.constraints import project_to_bounds


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    n = returns.shape[1]
    weights = np.ones(n) / n
    return pd.DataFrame(np.tile(weights, (len(returns), 1)), index=returns.index, columns=returns.columns)


def static_reserve_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """A stylised conservative reserve benchmark for the default 6-asset universe."""
    default = {
        "SHY": 0.30,
        "IEF": 0.25,
        "TLT": 0.15,
        "GLD": 0.15,
        "FXE": 0.075,
        "FXY": 0.075,
    }
    raw = np.array([default.get(col, 1.0 / returns.shape[1]) for col in returns.columns], dtype="float64")
    weights = raw / raw.sum()
    return pd.DataFrame(np.tile(weights, (len(returns), 1)), index=returns.index, columns=returns.columns)


def rolling_minimum_variance(
    returns: pd.DataFrame,
    window: int = 63,
    max_weight: float = 0.60,
) -> pd.DataFrame:
    rows = []
    n = returns.shape[1]
    max_weights = np.repeat(max_weight, n)
    for i in range(len(returns)):
        if i < window:
            rows.append(np.ones(n) / n)
            continue
        cov = returns.iloc[i - window : i].cov().to_numpy()
        inv_diag = 1.0 / np.maximum(np.diag(cov), 1e-10)
        w = inv_diag / inv_diag.sum()
        rows.append(project_to_bounds(w, max_weights=max_weights))
    return pd.DataFrame(rows, index=returns.index, columns=returns.columns)


def rolling_mean_variance(
    returns: pd.DataFrame,
    window: int = 63,
    risk_aversion: float = 10.0,
    max_weight: float = 0.60,
) -> pd.DataFrame:
    rows = []
    n = returns.shape[1]
    max_weights = np.repeat(max_weight, n)
    for i in range(len(returns)):
        if i < window:
            rows.append(np.ones(n) / n)
            continue
        sample = returns.iloc[i - window : i]
        mu = sample.mean().to_numpy()
        var = np.maximum(np.diag(sample.cov().to_numpy()), 1e-10)
        score = mu / (risk_aversion * var)
        w = np.maximum(score, 0.0)
        if np.isclose(w.sum(), 0.0):
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        rows.append(project_to_bounds(w, max_weights=max_weights))
    return pd.DataFrame(rows, index=returns.index, columns=returns.columns)


def portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame, transaction_cost_bps: float = 1.0) -> pd.Series:
    weights = weights.reindex(returns.index).ffill().fillna(1.0 / returns.shape[1])
    gross = (weights.shift(1).fillna(weights.iloc[0]) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * transaction_cost_bps / 10000.0
    return gross - cost


def run_all_baselines(returns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    strategies = {
        "Equal Weight": equal_weight(returns),
        "Static Reserve": static_reserve_weights(returns),
        "Minimum Variance": rolling_minimum_variance(returns),
        "Mean Variance": rolling_mean_variance(returns),
    }
    strategy_returns = pd.DataFrame(
        {name: portfolio_returns(returns, weights) for name, weights in strategies.items()},
        index=returns.index,
    )
    return strategy_returns, strategies
