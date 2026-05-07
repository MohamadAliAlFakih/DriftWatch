"""Threshold tuning tests.

File summary:
- Tests recall-constrained threshold selection.
- Verifies the highest valid threshold is chosen.
- Verifies impossible recall constraints raise a clear error.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.threshold import find_highest_threshold_meeting_recall


def test_returns_highest_threshold_satisfying_recall() -> None:
    """Verify the selector returns the highest threshold that satisfies recall."""
    y_true = pd.Series([1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.95, 0.80, 0.60, 0.20, 0.70, 0.10])

    result = find_highest_threshold_meeting_recall(y_true, y_proba, min_recall=0.75)

    assert result["threshold"] == 0.60
    assert result["recall"] >= 0.75


def test_raises_value_error_when_recall_is_impossible() -> None:
    """Verify threshold selection fails clearly when there are no positive labels."""
    y_true = pd.Series([0, 0, 0])
    y_proba = np.array([0.9, 0.4, 0.1])

    with pytest.raises(ValueError, match="no_positive_examples"):
        find_highest_threshold_meeting_recall(y_true, y_proba, min_recall=0.75)
