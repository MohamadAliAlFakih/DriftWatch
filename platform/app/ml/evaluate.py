"""Evaluation helpers for binary classifiers.

File summary:
- Converts predicted probabilities into binary labels using a supplied threshold.
- Computes the metric set used for model review and MLflow logging.
- Returns metrics as plain dictionaries so they can be saved as JSON or logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Return the small metric set needed for model review and MLflow."""
    y_score = np.asarray(y_proba, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
