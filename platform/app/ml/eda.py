"""Small EDA helpers used by the training notebook.

File summary:
- Provides quick summaries for columns, missing values, duplicates, and distributions.
- Separates numeric and categorical feature lists for later modeling steps.
- Computes target-rate summaries for categorical values during exploration.
- Keeps notebook EDA logic reusable instead of burying it inside notebook cells.
"""

from __future__ import annotations

import pandas as pd


def summarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column with type, null count, and unique count."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().to_numpy(),
            "unique_count": df.nunique(dropna=False).to_numpy(),
        }
    )


def get_numeric_categorical_columns(
    df: pd.DataFrame,
    target_col: str = "y",
) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical lists."""
    features = df.drop(columns=[target_col], errors="ignore")
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def categorical_cardinality_with_target_percentages(
    df: pd.DataFrame,
    target_col: str = "y",
) -> pd.DataFrame:
    """Summarize categorical cardinality and target rate by category value."""
    rows: list[dict[str, object]] = []
    _, categorical_columns = get_numeric_categorical_columns(df, target_col=target_col)
    working_df = df.copy()
    if working_df[target_col].dtype == "object":
        working_df[target_col] = working_df[target_col].map({"yes": 1, "no": 0})

    for column in categorical_columns:
        grouped = working_df.groupby(column, dropna=False)[target_col].agg(["count", "mean"])
        for category, values in grouped.iterrows():
            rows.append(
                {
                    "column": column,
                    "category": str(category),
                    "count": int(values["count"]),
                    "target_yes_percent": float(values["mean"] * 100),
                    "cardinality": int(df[column].nunique(dropna=False)),
                }
            )

    return pd.DataFrame(rows)


def numeric_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric distribution statistics useful for quick EDA."""
    numeric_df = df.select_dtypes(include=["number", "bool"])
    summary = numeric_df.agg(["mean", "median", "min", "max", "skew"]).T
    return summary.reset_index(names="column")


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for each column."""
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.to_numpy(),
            "missing_percent": (missing_count / len(df) * 100).to_numpy(),
        }
    )


def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row duplicate count summary."""
    duplicate_count = int(df.duplicated().sum())
    return pd.DataFrame(
        {
            "row_count": [len(df)],
            "duplicate_count": [duplicate_count],
            "duplicate_percent": [float(duplicate_count / len(df) * 100) if len(df) else 0.0],
        }
    )
