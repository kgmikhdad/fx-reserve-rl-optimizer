"""Feature engineering for portfolio allocation models."""
from __future__ import annotations

import pandas as pd


def rolling_features(returns: pd.DataFrame, windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Build rolling mean, volatility, and momentum features from returns."""
    parts: list[pd.DataFrame] = []
    for window in windows:
        parts.append(returns.rolling(window).mean().add_suffix(f"_mean_{window}"))
        parts.append(returns.rolling(window).std().add_suffix(f"_vol_{window}"))
        parts.append(returns.rolling(window).sum().add_suffix(f"_mom_{window}"))
    features = pd.concat(parts, axis=1).replace([float("inf"), float("-inf")], pd.NA)
    return features.fillna(0.0)


def drawdown_features(returns: pd.DataFrame) -> pd.DataFrame:
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return drawdown.add_suffix("_drawdown")
