"""Shared validation of portfolio weight vectors (CLAUDE.md section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_weights(
    weights: np.ndarray | pd.Series,
    tol: float = 1e-8,
    long_only: bool = True,
    max_weight: float | None = None,
    total: float = 1.0,
) -> None:
    """Raise ``ValueError`` unless ``weights`` form a valid portfolio.

    Parameters
    ----------
    weights : numpy.ndarray or pandas.Series
        Weight vector.
    tol : float, default 1e-8
        Absolute tolerance for the budget constraint, non-negativity and the cap.
    long_only : bool, default True
        Require every weight to be at least ``-tol``.
    max_weight : float, optional
        If given, require every weight to be at most ``max_weight + tol``.
    total : float, default 1.0
        Required sum of the weights (fully invested).
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("weights are empty")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or inf")
    if abs(w.sum() - total) > tol:
        raise ValueError(f"weights sum to {w.sum():.12f}, expected {total}")
    if long_only and w.min() < -tol:
        raise ValueError(f"long-only weights contain a negative value ({w.min():.3e})")
    if max_weight is not None and w.max() > max_weight + tol:
        raise ValueError(f"largest weight {w.max():.6f} exceeds the cap {max_weight}")
