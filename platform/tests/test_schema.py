"""Prediction schema tests.

File summary:
- Tests generation of the serving-time prediction schema artifact.
- Verifies target and leakage columns are excluded.
- Verifies categorical allowed values are captured.
"""

import pandas as pd

from app.ml.schema import infer_prediction_schema


def test_schema_excludes_target_and_duration_and_captures_categories() -> None:
    """Verify schema generation includes safe features and categorical values."""
    data = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["unknown", "admin."],
            "duration": [100, 200],
            "pdays": [-1, 4],
            "pdays_was_minus_one": [1, 0],
            "y": [1, 0],
        }
    )

    schema = infer_prediction_schema(data)

    assert "y" not in schema["required_fields"]
    assert "duration" not in schema["required_fields"]
    assert {"age", "job", "pdays", "pdays_was_minus_one"}.issubset(
        schema["required_fields"]
    )
    assert schema["threshold_location"] == "threshold.json"

    job_field = next(field for field in schema["features"] if field["name"] == "job")
    assert "unknown" in job_field["allowed_values"]
