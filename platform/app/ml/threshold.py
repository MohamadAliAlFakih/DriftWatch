"""Operating-threshold tuning for recall-constrained decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def find_highest_threshold_meeting_recall(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.75,
) -> dict[str, float]:
    """Find the highest threshold that still reaches the minimum recall.

    The threshold is returned as metadata and is intentionally not embedded in
    the sklearn pipeline. Serving should load the model and threshold separately.
    """
    if not 0 <= min_recall <= 1:
        raise ValueError({"error": "invalid_min_recall", "min_recall": min_recall})

    y_score = np.asarray(y_proba, dtype=float)
    if y_score.ndim != 1:
        raise ValueError({"error": "invalid_probability_shape", "shape": y_score.shape})

    y_array = np.asarray(y_true, dtype=int)
    if y_array.sum() == 0:
        raise ValueError({"error": "no_positive_examples", "min_recall": min_recall})

    thresholds = np.unique(np.concatenate(([0.0, 1.0], y_score)))
    best: dict[str, Any] | None = None

    for threshold in sorted(thresholds, reverse=True):
        y_pred = (y_score >= threshold).astype(int)
        recall = float(recall_score(y_array, y_pred, zero_division=0))
        if recall >= min_recall:
            best = {
                "threshold": float(threshold),
                "precision": float(precision_score(y_array, y_pred, zero_division=0)),
                "recall": recall,
                "f1": float(f1_score(y_array, y_pred, zero_division=0)),
            }
            break

    if best is None:
        raise ValueError(
            {
                "error": "no_threshold_meets_min_recall",
                "min_recall": min_recall,
            }
        )
    return best
