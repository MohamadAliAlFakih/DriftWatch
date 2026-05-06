"""Data loading, cleaning, and splitting for the UCI Bank Marketing model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "y"
DURATION_COL = "duration"
PDAYS_COL = "pdays"
PDAYS_SENTINEL = -1


def load_bank_marketing_data(path: str | Path) -> pd.DataFrame:
    """Load the bank marketing CSV from disk.

    UCI's bank marketing files are semicolon-separated, but ``sep=None`` keeps
    this loader tolerant if someone exports a comma-separated copy later.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            {
                "error": "dataset_not_found",
                "path": str(data_path),
                "hint": "Set DATA_PATH to the local bank marketing CSV.",
            }
        )
    return pd.read_csv(data_path, sep=None, engine="python")


def clean_bank_marketing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project cleaning rules without dropping useful rows."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if TARGET_COL not in cleaned.columns:
        raise ValueError({"error": "missing_target_column", "target_col": TARGET_COL})

    if DURATION_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[DURATION_COL])

    if PDAYS_COL in cleaned.columns:
        pdays_numeric = pd.to_numeric(cleaned[PDAYS_COL], errors="coerce")
        sentinel_mask = pdays_numeric == PDAYS_SENTINEL

        # The current bank-full.csv uses -1 for "not previously contacted".
        # We keep this as an explicit flag so the numeric pdays value can be
        # cleaned without losing the sentinel meaning.
        cleaned["pdays_was_minus_one"] = sentinel_mask.astype(int)
        cleaned["never_contacted_flag"] = cleaned["pdays_was_minus_one"]

        # Keep a numeric pdays value for modeling, with the sentinel replaced by
        # zero because the flag now carries the "never contacted" meaning.
        cleaned["pdays_clean"] = pdays_numeric.mask(sentinel_mask, 0)

    if cleaned[TARGET_COL].dtype == "object":
        cleaned[TARGET_COL] = cleaned[TARGET_COL].map({"yes": 1, "no": 0})

    if cleaned[TARGET_COL].isna().any():
        raise ValueError({"error": "invalid_target_values", "expected": ["yes", "no", 1, 0]})

    cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(int)
    return cleaned


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary target."""
    if target_col not in df.columns:
        raise ValueError({"error": "missing_target_column", "target_col": target_col})
    return df.drop(columns=[target_col]), df[target_col].astype(int)


def make_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the required stratified 70/30 train/test split."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
