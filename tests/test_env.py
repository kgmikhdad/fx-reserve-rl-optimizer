from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.data.demo_data import generate_demo_returns


gymnasium_available = importlib.util.find_spec("gymnasium") is not None


@pytest.mark.skipif(not gymnasium_available, reason="gymnasium is not installed")
def test_fx_reserve_env_step() -> None:
    from src.envs.fx_reserve_env import FXReservePortfolioEnv

    returns = generate_demo_returns(periods=80)
    env = FXReservePortfolioEnv(returns=returns, max_weights=np.repeat(0.6, returns.shape[1]))
    obs, _info = env.reset()
    assert obs.shape[0] == returns.shape[1] * 3 + 2
    next_obs, reward, _terminated, _truncated, info = env.step(np.ones(returns.shape[1]))
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert np.isclose(info["weights"].sum(), 1.0)
