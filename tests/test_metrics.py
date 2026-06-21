from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.metrics import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio


def test_annualized_return_positive_series() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    assert annualized_return(returns) > 0


def test_annualized_volatility_constant_series_is_zero() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    assert np.isclose(annualized_volatility(returns), 0.0)


def test_sharpe_handles_zero_volatility() -> None:
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(returns) == 0.0


def test_max_drawdown_is_negative_or_zero() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(returns) <= 0.0
