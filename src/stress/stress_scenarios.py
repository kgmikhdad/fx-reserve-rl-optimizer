"""Stylised stress scenarios for reserve portfolio diagnostics."""
from __future__ import annotations

import pandas as pd


def default_stress_scenarios() -> pd.DataFrame:
    """Return a small matrix of hypothetical stress losses by strategy.

    The numbers are illustrative only. They are intended for dashboard design,
    not for institutional stress calibration.
    """
    return pd.DataFrame(
        {
            "scenario": [
                "USD appreciation shock",
                "Gold selloff",
                "Global bond selloff",
                "Correlation breakdown",
            ],
            "Equal Weight": [-0.018, -0.012, -0.026, -0.021],
            "Static Reserve": [-0.012, -0.010, -0.018, -0.015],
            "Minimum Variance": [-0.008, -0.006, -0.011, -0.010],
            "Mean Variance": [-0.014, -0.009, -0.020, -0.017],
        }
    )
