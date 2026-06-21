# FX Reserve Portfolio Optimizer using Deep Reinforcement Learning

A research prototype for **stylised foreign-exchange reserve portfolio optimisation** using public market proxies, transparent portfolio benchmarks, a constrained reinforcement-learning environment, and a Streamlit dashboard.

The project is framed as a **decision-support prototype**, not as a trading bot. It emphasises reserve-management priorities: **safety, liquidity, diversification, drawdown control, turnover discipline, stress resilience, and transparent benchmark comparison**.

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

The implementation can use public market proxies rather than actual reserve-management data.

| Asset | Proxy role |
|---|---|
| SHY | Short-term US Treasuries proxy |
| IEF | Intermediate US Treasuries proxy |
| TLT | Long-duration US Treasuries proxy |
| GLD | Gold proxy |
| FXE | Euro currency proxy |
| FXY | Japanese yen proxy |

These are only convenient public proxies. They do **not** represent the reserve portfolio of any institution.

## Current implementation

The repository currently includes:

1. **Deterministic demo data** so the Streamlit app runs immediately after deployment.
2. **Public data download utilities** using `yfinance` for later real-market-proxy experiments.
3. **Portfolio benchmarks**: equal weight, static reserve benchmark, rolling minimum variance, and rolling mean-variance.
4. **Risk metrics**: annualised return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, VaR, expected shortfall, and turnover.
5. **RL environment**: a custom Gymnasium-compatible `FXReservePortfolioEnv`.
6. **PPO training script** using Stable-Baselines3.
7. **Stress-test scaffold** with illustrative scenario-loss matrix.
8. **Streamlit dashboard** for data, benchmark, risk, drawdown, allocation, stress-test, and model-explanation views.
9. **Pytest tests** and a GitHub Actions CI workflow.

## Reinforcement learning design

At each time step:

```text
state_t  -> recent returns, volatility, momentum, wealth, drawdown
action_t -> raw action vector projected into feasible long-only portfolio weights
reward_t -> institutional risk-adjusted reward
```

The environment is long-only and projects raw actions into valid portfolio weights. The reward penalises excessive volatility, drawdown, turnover, and concentration.

## Reward function

The default reward is:

```text
reward_t = portfolio_return_t
           - lambda_vol * rolling_volatility_t
           - lambda_drawdown * abs(drawdown_t)
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

The Streamlit app also runs without CSV files because it generates deterministic demo data in memory.

## Train PPO agent

```bash
python -m src.agents.train_ppo
```

Training is intentionally separate from the Streamlit app. The deployed dashboard should load lightweight precomputed outputs rather than train models on page load.

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
scripts/             Utility scripts
src/data/            Demo data, public data download, feature engineering
src/portfolio/       Metrics, constraints, baselines
src/envs/            Custom Gymnasium environment
src/agents/          PPO training script
src/stress/          Stress scenario utilities
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
