"""Train a PPO allocation agent on demo or processed return data.

This script is intentionally separate from the Streamlit app. The deployed app
should load precomputed outputs; it should not train RL models on page load.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.demo_data import generate_demo_returns
from src.envs.fx_reserve_env import FXReservePortfolioEnv

ROOT = Path(__file__).resolve().parents[2]


def load_returns() -> pd.DataFrame:
    processed = ROOT / "data" / "processed" / "returns.csv"
    if processed.exists():
        return pd.read_csv(processed, index_col=0, parse_dates=True)
    return generate_demo_returns()


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 is required. Install requirements.txt first.") from exc

    returns = load_returns()
    max_weights = np.array([0.50, 0.40, 0.25, 0.20, 0.20, 0.20])[: returns.shape[1]]
    env = FXReservePortfolioEnv(returns=returns, max_weights=max_weights)
    model = PPO("MlpPolicy", env, verbose=1, seed=42, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=50_000)

    out_dir = ROOT / "models" / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ppo_fx_reserve")
    print(f"Saved model to {out_dir / 'ppo_fx_reserve.zip'}")


if __name__ == "__main__":
    main()
