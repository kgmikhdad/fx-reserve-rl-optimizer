"""Generate lightweight CSV outputs for local exploration."""
from __future__ import annotations

from pathlib import Path

from src.data.demo_data import demo_dataset
from src.data.features import rolling_features
from src.portfolio.baselines import run_all_baselines, static_reserve_weights
from src.portfolio.metrics import metrics_table
from src.stress.stress_scenarios import default_stress_scenarios

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    prices, returns = demo_dataset()
    strategy_returns, _weights = run_all_baselines(returns)
    equity = (1.0 + strategy_returns).cumprod()
    metrics = metrics_table(strategy_returns)
    features = rolling_features(returns)

    sample = ROOT / "data" / "sample"
    reports = ROOT / "reports" / "tables"
    sample.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    prices.to_csv(sample / "sample_prices.csv")
    returns.to_csv(sample / "sample_returns.csv")
    features.to_csv(sample / "sample_features.csv")
    equity.to_csv(sample / "sample_equity_curves.csv")
    metrics.to_csv(sample / "sample_metrics.csv")
    strategy_returns.to_csv(sample / "sample_strategy_returns.csv")
    static_reserve_weights(returns).to_csv(sample / "sample_weights_static_reserve.csv")
    default_stress_scenarios().to_csv(sample / "sample_stress_scenarios.csv", index=False)

    strategy_returns.to_csv(reports / "benchmark_returns.csv")
    equity.to_csv(reports / "benchmark_equity_curves.csv")
    metrics.to_csv(reports / "benchmark_metrics.csv")
    print("Generated sample data and benchmark reports.")


if __name__ == "__main__":
    main()
