"""Prediction schema artifact generation.

File summary:
- Builds the `schema.json` artifact used by the serving validator.
- Excludes the target and leakage-prone columns from prediction inputs.
- Captures required fields, data types, categorical allowed values, and JSON-schema shape.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def infer_prediction_schema(
    df: pd.DataFrame,
    target_col: str = "y",
    excluded_columns: tuple[str, ...] = ("duration",),
) -> dict[str, Any]:
    """Create a JSON-serializable schema for serving-time inputs."""
    feature_df = df.drop(columns=[target_col, *excluded_columns], errors="ignore")
    fields: list[dict[str, Any]] = []

    for column in feature_df.columns:
        series = feature_df[column]
        field: dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "required": True,
        }
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            field["allowed_values"] = sorted(str(value) for value in series.dropna().unique())
        fields.append(field)

    return {
        "schema_version": 1,
        "target_excluded": target_col,
        "excluded_columns": list(excluded_columns),
        "threshold_location": "threshold.json",
        "required_fields": [field["name"] for field in fields],
        "features": fields,
        "json_schema": _to_json_schema(fields),
    }


def _to_json_schema(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a lightweight JSON schema description from feature metadata."""
    properties: dict[str, Any] = {}
    for field in fields:
        json_type = "number"
        if "allowed_values" in field:
            json_type = "string"
        elif field["dtype"] in {"int64", "int32", "bool"}:
            json_type = "integer"

        properties[field["name"]] = {"type": json_type}
        if "allowed_values" in field:
            properties[field["name"]]["enum"] = field["allowed_values"]

    return {
        "type": "object",
        "required": [field["name"] for field in fields],
        "properties": properties,
        "additionalProperties": False,
    }
