"""Gymnasium environment for stylised FX reserve portfolio allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from src.portfolio.constraints import action_to_weights, concentration_penalty


class FXReservePortfolioEnv(gym.Env if gym is not None else object):
    """Long-only portfolio allocation environment.

    The action is a raw continuous vector that is projected into feasible weights.
    The reward is institutional: return minus penalties for volatility, drawdown,
    turnover, and concentration.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        window: int = 21,
        max_weights: np.ndarray | None = None,
        transaction_cost_bps: float = 1.0,
        lambda_vol: float = 0.5,
        lambda_drawdown: float = 1.0,
        lambda_turnover: float = 0.1,
        lambda_concentration: float = 0.05,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to use FXReservePortfolioEnv")
        if len(returns) <= window + 1:
            raise ValueError("returns must contain more observations than the rolling window")
        self.returns = returns.astype("float64")
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.window = window
        self.max_weights = np.asarray(max_weights, dtype="float64") if max_weights is not None else np.ones(self.n_assets)
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.lambda_vol = lambda_vol
        self.lambda_drawdown = lambda_drawdown
        self.lambda_turnover = lambda_turnover
        self.lambda_concentration = lambda_concentration

        obs_dim = self.n_assets * 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32)
        self.current_step = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.wealth = 1.0
        self.peak_wealth = 1.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.wealth = 1.0
        self.peak_wealth = 1.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        previous_weights = self.weights.copy()
        self.weights = action_to_weights(action, max_weights=self.max_weights)

        current_return_vector = self.returns.iloc[self.current_step].to_numpy()
        turnover = float(np.abs(self.weights - previous_weights).sum())
        portfolio_return = float(previous_weights @ current_return_vector - turnover * self.transaction_cost)

        self.wealth *= 1.0 + portfolio_return
        self.peak_wealth = max(self.peak_wealth, self.wealth)
        drawdown = min(self.wealth / self.peak_wealth - 1.0, 0.0)

        rolling_cov = self.returns.iloc[self.current_step - self.window : self.current_step].cov().to_numpy()
        portfolio_vol = float(np.sqrt(max(self.weights @ rolling_cov @ self.weights, 0.0)))
        conc = concentration_penalty(self.weights)

        reward = (
            portfolio_return
            - self.lambda_vol * portfolio_vol
            - self.lambda_drawdown * abs(drawdown)
            - self.lambda_turnover * turnover
            - self.lambda_concentration * conc
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False
        info = {
            "portfolio_return": portfolio_return,
            "wealth": self.wealth,
            "drawdown": drawdown,
            "turnover": turnover,
            "portfolio_volatility": portfolio_vol,
            "weights": self.weights.copy(),
        }
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        window_returns = self.returns.iloc[self.current_step - self.window : self.current_step]
        mean_returns = window_returns.mean().to_numpy()
        vol = window_returns.std().fillna(0.0).to_numpy()
        momentum = self.returns.iloc[self.current_step - min(self.window, 21) : self.current_step].sum().to_numpy()
        drawdown = self.wealth / self.peak_wealth - 1.0
        obs = np.concatenate([mean_returns, vol, momentum, [self.wealth, drawdown]])
        return obs.astype(np.float32)
