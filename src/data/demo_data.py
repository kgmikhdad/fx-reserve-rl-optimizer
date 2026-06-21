"""Deterministic demo data so the Streamlit app works immediately after deployment."""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ["SHY", "IEF", "TLT", "GLD", "FXE", "FXY"]
ASSET_ROLES = {
    "SHY": "Short-term US Treasuries proxy",
    "IEF": "Intermediate US Treasuries proxy",
    "TLT": "Long-duration US Treasuries proxy",
    "GLD": "Gold proxy",
    "FXE": "Euro currency proxy",
    "FXY": "Japanese yen proxy",
}


def generate_demo_returns(periods: int = 756, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
    drift = np.array([0.00006, 0.00005, 0.00004, 0.00008, 0.00002, 0.00001])
    vol = np.array([0.0008, 0.0025, 0.0060, 0.0080, 0.0040, 0.0045])
    corr = np.array(
        [
            [1.00, 0.55, 0.35, 0.05, -0.10, -0.05],
            [0.55, 1.00, 0.65, 0.05, -0.05, -0.05],
            [0.35, 0.65, 1.00, 0.00, -0.10, -0.10],
            [0.05, 0.05, 0.00, 1.00, 0.20, 0.10],
            [-0.10, -0.05, -0.10, 0.20, 1.00, 0.35],
            [-0.05, -0.05, -0.10, 0.10, 0.35, 1.00],
        ]
    )
    cov = np.outer(vol, vol) * corr
    shocks = rng.multivariate_normal(drift, cov, size=periods)
    returns = pd.DataFrame(shocks, index=dates, columns=ASSETS)
    # Add a mild bond selloff regime to make drawdowns visible.
    stress_slice = slice(int(periods * 0.55), int(periods * 0.65))
    returns.iloc[stress_slice, returns.columns.get_indexer(["IEF", "TLT"])] -= np.array([0.0008, 0.0020])
    return returns


def generate_demo_prices(returns: pd.DataFrame) -> pd.DataFrame:
    return 100.0 * (1.0 + returns).cumprod()


def demo_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = generate_demo_returns()
    prices = generate_demo_prices(returns)
    return prices, returns
