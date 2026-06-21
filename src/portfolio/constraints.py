"""Portfolio weight constraints and action projection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(raw_weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map any real-valued vector to non-negative weights summing to one."""
    x = np.asarray(raw_weights, dtype="float64")
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = exp_x.sum()
    if denom <= eps:
        return np.ones_like(x) / len(x)
    return exp_x / denom


def project_to_bounds(
    weights: np.ndarray,
    max_weights: np.ndarray | None = None,
    min_weights: np.ndarray | None = None,
    max_iter: int = 100,
) -> np.ndarray:
    """Project long-only weights into simple box bounds and renormalise.

    This is a practical projection for research prototypes. For stricter mandate
    constraints, replace with a convex optimisation projection.
    """
    w = np.asarray(weights, dtype="float64").copy()
    n = len(w)
    if min_weights is None:
        min_weights = np.zeros(n)
    if max_weights is None:
        max_weights = np.ones(n)
    min_w = np.asarray(min_weights, dtype="float64")
    max_w = np.asarray(max_weights, dtype="float64")
    if min_w.sum() > 1.0 or max_w.sum() < 1.0:
        raise ValueError("Infeasible weight bounds: lower bounds sum above one or upper bounds sum below one.")
    w = np.clip(w, min_w, max_w)
    for _ in range(max_iter):
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-10:
            break
        if gap > 0:
            room = max_w - w
            idx = room > 1e-12
            if not idx.any():
                break
            w[idx] += gap * room[idx] / room[idx].sum()
        else:
            room = w - min_w
            idx = room > 1e-12
            if not idx.any():
                break
            w[idx] += gap * room[idx] / room[idx].sum()
        w = np.clip(w, min_w, max_w)
    return w / w.sum()


def action_to_weights(raw_action: np.ndarray, max_weights: np.ndarray | None = None) -> np.ndarray:
    """Convert raw RL action vector into feasible portfolio weights."""
    return project_to_bounds(normalize_weights(raw_action), max_weights=max_weights)


def validate_weights(weights: pd.DataFrame | np.ndarray, atol: float = 1e-8) -> bool:
    arr = weights.to_numpy() if isinstance(weights, pd.DataFrame) else np.asarray(weights)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    non_negative = np.all(arr >= -atol)
    sums_to_one = np.allclose(arr.sum(axis=1), 1.0, atol=atol)
    finite = np.isfinite(arr).all()
    return bool(non_negative and sums_to_one and finite)


def concentration_penalty(weights: np.ndarray) -> float:
    """Herfindahl-style concentration penalty."""
    w = np.asarray(weights, dtype="float64")
    return float(np.sum(w**2))
