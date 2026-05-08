"""Data loading, cleaning, and splitting for the UCI Bank Marketing model.

File summary:
- Loads the raw bank marketing CSV from disk.
- Applies project-specific cleaning rules before training or drift reference creation.
- Splits cleaned data into features and target.
- Creates the required stratified 60/20/20 train/validation/test split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "y"
DURATION_COL = "duration"
PDAYS_COL = "pdays"
PDAYS_SENTINEL = 999


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

        # UCI bank-additional-full.csv uses 999 for "not previously contacted".
        # Keep that fact as explicit model signal while making pdays_clean numeric.
        cleaned["pdays_was_999"] = sentinel_mask.astype(int)
        cleaned["never_contacted_flag"] = cleaned["pdays_was_999"]

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
    """Separate model features from the binary target column."""
    if target_col not in df.columns:
        raise ValueError({"error": "missing_target_column", "target_col": target_col})
    return df.drop(columns=[target_col]), df[target_col].astype(int)


def make_train_validation_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create the required stratified 60/20/20 train/validation/test split."""
    total = train_size + validation_size + test_size
    if round(total, 10) != 1.0:
        raise ValueError(
            {
                "error": "invalid_split_sizes",
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
            }
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )
    relative_test_size = test_size / (validation_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split for legacy callers."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
