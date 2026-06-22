"""Forecast evaluation metrics.

Point-forecast metrics:
    mape  — Mean Absolute Percentage Error (fraction; multiply by 100 for %)
    rmse  — Root Mean Squared Error
    mae   — Mean Absolute Error

Probabilistic metrics:
    coverage      — fraction of y_true inside the [low, high] interval
    pinball_loss  — per-quantile pinball loss (a.k.a. quantile loss)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error (returned as a fraction)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def coverage(y_true: ArrayLike, y_low: ArrayLike, y_high: ArrayLike) -> float:
    """Fraction of y_true within [y_low, y_high]. For a 90% PI, should be ~0.90."""
    y_true = np.asarray(y_true)
    y_low  = np.asarray(y_low)
    y_high = np.asarray(y_high)
    return float(np.mean((y_true >= y_low) & (y_true <= y_high)))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """Pinball loss for a single quantile prediction (0 < quantile < 1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def summarize(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Convenience wrapper — returns all three point-forecast metrics."""
    return {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
    }
