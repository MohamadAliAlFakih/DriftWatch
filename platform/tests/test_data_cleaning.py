"""Data cleaning tests for the bank marketing dataset.

File summary:
- Tests the project-specific bank marketing cleaning rules.
- Verifies target conversion from yes/no to 1/0.
- Verifies leakage, `pdays` sentinel handling, and the required 60/20/20 split.
"""

import pandas as pd

from app.ml.data import clean_bank_marketing_data, make_train_validation_test_split


def test_cleaning_applies_required_bank_marketing_rules() -> None:
    """Verify cleaning drops leakage columns and creates required `pdays` flags."""
    raw = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["unknown", "admin."],
            "duration": [100, 200],
            "pdays": [999, 4],
            "y": ["yes", "no"],
        }
    )

    cleaned = clean_bank_marketing_data(raw)

    assert "duration" not in cleaned.columns
    assert cleaned["y"].tolist() == [1, 0]
    assert cleaned["pdays_was_999"].tolist() == [1, 0]
    assert cleaned["never_contacted_flag"].tolist() == [1, 0]
    assert cleaned["pdays_clean"].tolist() == [0, 4]
    assert "unknown" in cleaned["job"].tolist()


def test_required_split_is_stratified_60_20_20() -> None:
    """Verify the project split follows the requirements PDF."""
    x = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 80 + [1] * 20)

    x_train, x_val, x_test, y_train, y_val, y_test = make_train_validation_test_split(
        x,
        y,
        random_state=42,
    )

    assert (len(x_train), len(x_val), len(x_test)) == (60, 20, 20)
    assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (12, 4, 4)
