"""Portfolio performance and risk metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_series(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna()
    return pd.Series(returns, dtype="float64").dropna()


def annualized_return(returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    return float(r.mean() * periods)


def annualized_volatility(returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    excess = r - risk_free_rate / periods
    vol = annualized_volatility(excess, periods)
    if np.isclose(vol, 0.0):
        return 0.0
    return float(annualized_return(excess, periods) / vol)


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    excess = r - risk_free_rate / periods
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_vol = downside.std(ddof=1) * np.sqrt(periods)
    if np.isclose(downside_vol, 0.0):
        return 0.0
    return float(annualized_return(excess, periods) / downside_vol)


def cumulative_wealth(returns: pd.Series | np.ndarray) -> pd.Series:
    r = _to_series(returns)
    return (1.0 + r).cumprod()


def drawdown_series(returns: pd.Series | np.ndarray) -> pd.Series:
    wealth = cumulative_wealth(returns)
    if wealth.empty:
        return wealth
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    if np.isclose(mdd, 0.0):
        return 0.0
    return float(annualized_return(returns, periods) / mdd)


def historical_var(returns: pd.Series | np.ndarray, alpha: float = 0.05) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    return float(-np.quantile(r, alpha))


def expected_shortfall(returns: pd.Series | np.ndarray, alpha: float = 0.05) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    threshold = np.quantile(r, alpha)
    tail = r[r <= threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype="float64")
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def metrics_table(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for name in strategy_returns.columns:
        r = strategy_returns[name].dropna()
        rows[name] = {
            "annualized_return": annualized_return(r),
            "annualized_volatility": annualized_volatility(r),
            "sharpe_ratio": sharpe_ratio(r),
            "sortino_ratio": sortino_ratio(r),
            "max_drawdown": max_drawdown(r),
            "calmar_ratio": calmar_ratio(r),
            "var_95": historical_var(r, 0.05),
            "expected_shortfall_95": expected_shortfall(r, 0.05),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
