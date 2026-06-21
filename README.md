# FX Reserve Portfolio Optimizer using Deep Reinforcement Learning

A research prototype for **stylised foreign-exchange reserve portfolio optimisation** using public market proxies, transparent portfolio benchmarks, a constrained reinforcement-learning environment, and a Streamlit dashboard.

The project is framed as a **decision-support prototype**, not as a trading bot. It emphasises reserve-management priorities: **safety, liquidity, diversification, drawdown control, turnover discipline, stress resilience, and transparent benchmark comparison**.

## Live app

After deployment, add the Streamlit URL here.

## What the dashboard can do now

The Streamlit app supports two market-data modes:

1. **Demo data**: deterministic synthetic data that always works, useful for quick demonstrations.
2. **Real market data via yfinance**: public ETF/currency proxy data downloaded inside the app and cached for one hour.

Interactive controls include:

- asset universe selection,
- live-data start and end dates,
- daily, weekly, or monthly return frequency,
- transaction-cost assumption,
- rolling estimation window,
- maximum asset-weight constraint,
- mean-variance risk aversion,
- portfolio rebalancing frequency,
- custom manual portfolio weights,
- reserve-utility penalty weights,
- stress-scenario selection,
- custom one-day shock design,
- CSV downloads for metrics, returns, and latest weights.

The app compares:

1. **Equal Weight**
2. **Static Reserve Benchmark**
3. **Rolling Minimum Variance**
4. **Rolling Mean-Variance**
5. **Your Custom Portfolio**

The DRL environment and PPO training script are included in the repository for the next implementation stage.

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
| BIL | US Treasury bills / cash proxy |
| SHY | 1-3 year US Treasuries proxy |
| IEI | 3-7 year US Treasuries proxy |
| IEF | 7-10 year US Treasuries proxy |
| TLT | 20+ year US Treasuries proxy |
| GLD | Gold proxy |
| FXE | Euro currency proxy |
| FXY | Japanese yen currency proxy |
| FXB | British pound currency proxy |
| FXC | Canadian dollar currency proxy |
| FXA | Australian dollar currency proxy |
| UUP | US dollar index ETF proxy |

These are only convenient public proxies. They do **not** represent the reserve portfolio of any institution.

## Current implementation

The repository currently includes:

1. **Deterministic demo data** so the Streamlit app runs immediately after deployment.
2. **Live public market data mode** using `yfinance` for real-market-proxy experiments.
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
streamlit run app/streamlit_app.py
```

The Streamlit app also runs without CSV files because it generates deterministic demo data in memory. For live market data, select **Real market data via yfinance** in the sidebar.

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
src/data/            Demo data, live data download, feature engineering
src/portfolio/       Metrics, constraints, baselines
src/envs/            Custom Gymnasium environment
src/agents/          PPO training script
src/stress/          Stress scenario utilities
tests/               Unit tests
```

## Limitations

- Public ETFs and currency proxies are used instead of actual reserve assets.
- The live data mode depends on public upstream data availability.
- Transaction costs, liquidity haircuts, and mandate constraints are simplified.
- The RL model may overfit if used without strict out-of-sample validation.
- This is not an investment recommendation system.
- The app is a research prototype for interview and portfolio demonstration purposes.

## Future extensions

- Add SAC and TD3 agents.
- Add risk-parity and CVaR-optimisation benchmarks.
- Add FRED macro-financial features such as rates, inflation, and dollar index proxies.
- Add walk-forward retraining.
- Add richer stress-testing and attribution analysis.
- Add model explainability and regime analysis.

## Disclaimer

This project is a research and educational prototype. It uses publicly available market proxies and synthetic sample data. It does not represent the reserve portfolio, investment policy, internal systems, or recommendations of any central bank, the Bank for International Settlements, or any financial institution. The results are not investment advice.
