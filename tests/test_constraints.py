from __future__ import annotations

import numpy as np

from src.portfolio.constraints import action_to_weights, validate_weights


def test_action_to_weights_sums_to_one() -> None:
    weights = action_to_weights(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_action_to_weights_respects_upper_bounds() -> None:
    weights = action_to_weights(np.array([10.0, 0.0, -2.0]), max_weights=np.array([0.5, 0.5, 0.5]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights <= 0.5 + 1e-8)


def test_validate_weights() -> None:
    assert validate_weights(np.array([0.3, 0.4, 0.3]))
