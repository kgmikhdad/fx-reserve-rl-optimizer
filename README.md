# FX Reserve Portfolio Optimizer using Deep Reinforcement Learning

A research prototype for **stylised foreign-exchange reserve portfolio optimisation** using public market proxies, traditional portfolio benchmarks, and a constrained deep reinforcement learning allocation environment.

The project is framed as a **decision-support prototype** rather than a trading bot. It emphasises reserve-management priorities: **safety, liquidity, diversification, drawdown control, turnover discipline, stress resilience, and transparent benchmark comparison**.

## Live app

After deployment, add the Streamlit URL here.

## Project motivation

Central-bank reserve management is not ordinary return maximisation. Reserve portfolios are typically evaluated through institutional objectives such as:

- capital preservation,
- liquidity,
- currency and duration diversification,
- low operational turnover,
- drawdown control,
- and resilience under market stress.

This project asks whether a constrained deep reinforcement learning agent can learn a dynamic allocation policy for a stylised reserve portfolio while respecting these priorities.

## Public proxy assets

The implementation uses public market proxies rather than actual reserve-management data.

| Asset | Proxy role |
|---|---|
| SHY | Short-term US Treasuries |
| IEF | Intermediate US Treasuries |
| TLT | Long-duration US Treasuries |
| GLD | Gold |
| FXE | Euro currency proxy |
| FXY | Japanese yen proxy |

These are only convenient public proxies. They do **not** represent the reserve portfolio of any institution.

## Methodology

The project has five layers:

1. **Data pipeline**: download or generate price data, compute returns, and build rolling features.
2. **Benchmark portfolios**: equal weight, static reserve benchmark, rolling minimum variance, and rolling mean-variance.
3. **RL environment**: a custom Gymnasium-compatible portfolio allocation environment.
4. **DRL agent**: PPO training using Stable-Baselines3.
5. **Dashboard**: Streamlit app for visualising data, allocation weights, risk metrics, and stress scenarios.

## Reinforcement learning design

At each time step:

```text
state_t  -> market and portfolio features
action_t -> portfolio weights
reward_t -> institutional risk-adjusted reward
```

The environment is long-only and projects raw actions into valid portfolio weights. The reward penalises excessive volatility, drawdown, turnover, and concentration.

## Reward function

The default reward is:

```text
reward_t = portfolio_return_t
           - lambda_vol * rolling_volatility_t
           - lambda_drawdown * drawdown_t
           - lambda_turnover * turnover_t
           - lambda_concentration * concentration_penalty_t
```

This is designed to reflect reserve-management discipline rather than pure return chasing.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
streamlit run app/streamlit_app.py
```

## Train PPO agent

```bash
python -m src.agents.train_ppo
python -m src.agents.evaluate_agent
```

The Streamlit app is intentionally designed to load precomputed results instead of training inside the web app.

## Tests

```bash
pytest tests/
ruff check .
```

## Streamlit deployment

Use Streamlit Community Cloud with:

```text
Repository: kgmikhdad/fx-reserve-rl-optimizer
Branch: main
Main file path: app/streamlit_app.py
```

## Repository structure

```text
app/                 Streamlit dashboard
configs/             YAML configuration files
data/sample/         Synthetic sample data for immediate demo
scripts/             Utility scripts
src/data/            Data download, cleaning, feature engineering
src/portfolio/       Metrics, constraints, baselines, backtester
src/envs/            Custom RL environment
src/agents/          PPO training and evaluation scripts
src/stress/          Stress scenario utilities
src/utils/           Config, logging, seeding
tests/               Unit tests
```

## Limitations

- Public ETFs and currency proxies are used instead of actual reserve assets.
- Transaction costs, liquidity haircuts, and mandate constraints are simplified.
- The RL model may overfit if used without strict out-of-sample validation.
- This is not an investment recommendation system.
- The app is a research prototype for interview and portfolio demonstration purposes.

## Future extensions

- Add SAC and TD3 agents.
- Add risk-parity and CVaR-optimisation benchmarks.
- Add macro-financial features such as rates, inflation, and dollar index proxies.
- Add walk-forward retraining.
- Add richer stress-testing and attribution analysis.
- Add model explainability and regime analysis.

## Disclaimer

This project is a research and educational prototype. It uses publicly available market proxies and synthetic sample data. It does not represent the reserve portfolio, investment policy, internal systems, or recommendations of any central bank, the Bank for International Settlements, or any financial institution. The results are not investment advice.
