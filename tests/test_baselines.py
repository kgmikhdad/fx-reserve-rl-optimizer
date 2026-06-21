from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.demo_data import generate_demo_returns
from src.portfolio.baselines import equal_weight, run_all_baselines, static_reserve_weights
from src.portfolio.constraints import validate_weights


def test_equal_weight_valid() -> None:
    returns = generate_demo_returns(periods=80)
    weights = equal_weight(returns)
    assert validate_weights(weights)
    assert weights.shape == returns.shape


def test_static_reserve_valid() -> None:
    returns = generate_demo_returns(periods=80)
    weights = static_reserve_weights(returns)
    assert validate_weights(weights)
    assert np.isclose(weights.iloc[0].sum(), 1.0)


def test_run_all_baselines_outputs_returns() -> None:
    returns = generate_demo_returns(periods=100)
    strategy_returns, weights = run_all_baselines(returns)
    assert isinstance(strategy_returns, pd.DataFrame)
    assert "Static Reserve" in strategy_returns.columns
    assert "Equal Weight" in weights
