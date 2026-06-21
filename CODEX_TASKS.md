# Codex task roadmap

Use these as small implementation prompts in Codex rather than asking for the whole project at once.

## 1. Improve data pipeline

Add robust handling for missing prices, ticker failures, calendar alignment, and optional FRED macro-financial series.

## 2. Add stronger benchmark strategies

Implement risk parity, maximum Sharpe, and CVaR-constrained optimisation. Add unit tests for each.

## 3. Expand RL evaluation

Add an evaluation script that loads a trained PPO model, runs an out-of-sample backtest, and exports returns, weights, and metrics to `reports/tables/`.

## 4. Add SAC agent

Implement a Soft Actor-Critic training script for the continuous allocation environment and compare against PPO.

## 5. Add walk-forward validation

Train on expanding windows and test on forward periods. Export fold-wise results.

## 6. Add stress attribution

Break scenario losses into asset contribution and allocation effect.

## 7. Polish Streamlit deployment

Add pages for data explorer, risk dashboard, stress tests, model explanation, and limitations.
