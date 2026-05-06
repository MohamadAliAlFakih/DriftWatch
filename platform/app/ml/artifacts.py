"""Artifact helpers for the model_v1 bundle."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib

IMPORTANT_PACKAGES = ("sklearn", "pandas", "numpy", "mlflow", "joblib")


def compute_file_md5(path: str | Path) -> str:
    """Compute an MD5 hash for dataset tracking."""
    return _compute_hash(path, hashlib.md5())


def compute_file_sha256(path: str | Path) -> str:
    """Compute a SHA-256 hash for model artifact integrity."""
    return _compute_hash(path, hashlib.sha256())


def save_model_joblib(model: Any, path: str | Path) -> Path:
    """Save the sklearn pipeline as the serving model artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def create_environment_fingerprint() -> dict[str, Any]:
    """Capture runtime versions needed to reproduce the model."""
    packages: dict[str, str] = {}
    for package_name in IMPORTANT_PACKAGES:
        lookup_name = "scikit-learn" if package_name == "sklearn" else package_name
        try:
            packages[package_name] = metadata.version(lookup_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = "not-installed"

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dictionary as pretty, stable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def create_model_card(
    *,
    dataset_name: str,
    dataset_hash: str,
    row_count: int,
    column_count: int,
    model_class: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    threshold: dict[str, float],
    environment_fingerprint: dict[str, Any],
    artifact_hash: str,
) -> str:
    """Create a concise Markdown model card for the registered model."""
    return f"""# DriftWatch Bank Marketing Model

## Dataset
- Name: {dataset_name}
- MD5 hash: `{dataset_hash}`
- Shape: {row_count} rows, {column_count} columns
- Target: `y`, where `yes` maps to 1 and `no` maps to 0

## Training Setup
- Split strategy: stratified 70/30 train/test, random_state=42
- Cross-validation: stratified folds on the training split
- Leakage warning: `duration` is dropped because it is known only after a call ends
- `pdays` sentinel: `pdays == -1` becomes `pdays_was_minus_one`, `never_contacted_flag`,
  and `pdays_clean`
- `unknown` treatment: preserved as a real categorical value

## Model
- Class: {model_class}
- Hyperparameters:
```json
{json.dumps(hyperparameters, indent=2, sort_keys=True)}
```

## Final Test Metrics
```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

## Operating Threshold
```json
{json.dumps(threshold, indent=2, sort_keys=True)}
```

## Environment Fingerprint
```json
{json.dumps(environment_fingerprint, indent=2, sort_keys=True)}
```

## Artifact Integrity
- SHA-256: `{artifact_hash}`

## Intended Use
- Score bank marketing leads for subscription propensity and support drift monitoring.

## Not Intended Use
- Do not use as the only basis for customer treatment, credit decisions, or regulated actions.

## Limitations
- The model reflects historical campaign data and may drift when economic conditions change.
- Recall-focused thresholding can increase false positives.
- New categories at serving time are ignored by the one-hot encoder.
"""


def _compute_hash(path: str | Path, digest: Any) -> str:
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(digest.hexdigest())
