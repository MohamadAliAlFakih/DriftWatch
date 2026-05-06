"""Data cleaning tests for the bank marketing dataset."""

import pandas as pd

from app.ml.data import clean_bank_marketing_data


def test_cleaning_applies_required_bank_marketing_rules() -> None:
    raw = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["unknown", "admin."],
            "duration": [100, 200],
            "pdays": [-1, 4],
            "y": ["yes", "no"],
        }
    )

    cleaned = clean_bank_marketing_data(raw)

    assert "duration" not in cleaned.columns
    assert cleaned["y"].tolist() == [1, 0]
    assert cleaned["pdays_was_minus_one"].tolist() == [1, 0]
    assert cleaned["never_contacted_flag"].tolist() == [1, 0]
    assert cleaned["pdays_clean"].tolist() == [0, 4]
    assert "unknown" in cleaned["job"].tolist()
