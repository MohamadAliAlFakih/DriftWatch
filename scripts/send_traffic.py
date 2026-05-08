"""DriftWatch demo traffic generator.

Two-phase script that drives the Friday demo end-to-end:
  Phase 1 — 50 NORMAL prediction requests + drift check (establishes baseline)
  Phase 2 — 200 SHIFTED prediction requests + drift check (trips drift webhook)

The drift "shift" mutates one numeric and one categorical column per the PDF
("Shift one numeric and one categorical live during the demo"). For the
20-column bank-additional-full.csv the numeric we shift is `euribor3m` (the
PDF's named macroeconomic indicator) and the categorical is `job`.

Run:
    uv run --project scripts scripts/send_traffic.py
    uv run --project scripts scripts/send_traffic.py --base-url http://localhost:8001
    uv run --project scripts scripts/send_traffic.py --normal 50 --shifted 200 --no-pause

Open the dashboard FIRST (http://localhost:8501) so the audience sees the
empty/calm state before this script trips drift.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "platform" / "data" / "bank-additional-full.csv"
)

# Drift configuration — PDF page 2 names euribor3m as the recommended numeric to shift.
# +3.0 simulates a sudden 3-percentage-point rate hike, way past the medium PSI threshold.
SHIFT_NUMERIC_COL = "euribor3m"
SHIFT_NUMERIC_DELTA = 3.0
SHIFT_CATEGORICAL_COL = "job"
SHIFT_CATEGORICAL_VALUE = "student"    # over-represent one category to trip chi-square

# pdays sentinel handling — must mirror platform/app/ml/data.py so derived fields
# match what the model was trained on. bank-additional-full.csv uses 999.
PDAYS_SENTINEL = 999

# Integer columns in the schema; floats look like 25.0 in pandas but the platform
# validator wants real ints, so we cast back before serializing.
INTEGER_COLUMNS = {
    "age",
    "campaign",
    "pdays",
    "previous",
    "pdays_was_999",
    "never_contacted_flag",
    "pdays_clean",
}


def load_rows(data_path: Path) -> pd.DataFrame:
    """Load and prep rows so each one is a valid /predict payload."""
    if not data_path.exists():
        sys.exit(f"data file not found: {data_path}")

    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Drop the columns the model was trained without.
    df = df.drop(columns=[c for c in ("y", "duration") if c in df.columns])

    # Add the three derived pdays fields the schema requires (bank-additional uses 999).
    pdays_numeric = pd.to_numeric(df["pdays"], errors="coerce")
    sentinel_mask = pdays_numeric == PDAYS_SENTINEL
    df["pdays_was_999"] = sentinel_mask.astype(int)
    df["never_contacted_flag"] = df["pdays_was_999"]
    df["pdays_clean"] = pdays_numeric.mask(sentinel_mask, 0).astype(int)

    return df


def shift_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Mutate one numeric and one categorical column to simulate drift."""
    shifted = df.copy()
    if SHIFT_NUMERIC_COL in shifted.columns:
        shifted[SHIFT_NUMERIC_COL] = shifted[SHIFT_NUMERIC_COL] + SHIFT_NUMERIC_DELTA
    if SHIFT_CATEGORICAL_COL in shifted.columns:
        shifted[SHIFT_CATEGORICAL_COL] = SHIFT_CATEGORICAL_VALUE
    return shifted


def post_predictions(client: httpx.Client, base_url: str, rows: pd.DataFrame, label: str) -> None:
    """POST each row to /api/v1/predict, with a one-line progress dot stream."""
    url = f"{base_url}/api/v1/predict"
    sent = 0
    failed = 0
    print(f"  {label}: posting {len(rows)} predictions to {url}")
    for _, row in rows.iterrows():
        payload = _row_to_payload(row)
        try:
            response = client.post(url, json=payload, timeout=10.0)
            if response.status_code >= 400:
                failed += 1
                if failed <= 3:
                    # show first few failures so misconfiguration is obvious
                    print(f"\n    !! {response.status_code} {response.text[:200]}")
            else:
                sent += 1
        except httpx.HTTPError as exc:
            failed += 1
            if failed <= 3:
                print(f"\n    !! transport error: {exc}")
        if (sent + failed) % 25 == 0:
            print(".", end="", flush=True)
    print(f"\n  {label} done: sent={sent} failed={failed}")


def trigger_drift_check(client: httpx.Client, base_url: str, label: str) -> dict[str, Any]:
    """Hit /api/v1/drift/check and return the parsed response."""
    url = f"{base_url}/api/v1/drift/check"
    response = client.post(url, timeout=30.0)
    response.raise_for_status()
    body = response.json()
    severity = body.get("severity", "?")
    previous = body.get("previous_severity", "?")
    alert = body.get("alert")
    print(f"  {label}: severity={severity}  previous={previous}  webhook_fired={alert is not None}")
    if alert:
        print(f"     event_id={alert.get('event_id')}  status={alert.get('status')}")
    return body


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a pandas row to a JSON-safe dict matching the platform schema."""
    payload: dict[str, Any] = {}
    for col, value in row.items():
        if pd.isna(value):
            continue
        col_str = str(col)
        # The schema is strict about int vs float. Integer columns must arrive as
        # Python ints; pandas often loads small integers as floats (e.g. 25.0).
        if col_str in INTEGER_COLUMNS:
            payload[col_str] = int(value)
        elif isinstance(value, (int, bool)):
            payload[col_str] = int(value)
        elif isinstance(value, float):
            payload[col_str] = float(value)
        else:
            payload[col_str] = str(value)
    return payload


def main() -> None:
    """Drive the two-phase demo: warmup → baseline → shift → trip."""
    parser = argparse.ArgumentParser(description="DriftWatch demo traffic generator")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="platform base URL")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to bank CSV")
    # 250 fills the rolling window (drift_window_size=200) with calm baseline data.
    parser.add_argument("--normal", type=int, default=250, help="number of normal predictions")
    parser.add_argument("--shifted", type=int, default=250, help="number of shifted predictions")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="don't pause between phases (useful for automated runs)",
    )
    args = parser.parse_args()

    df = load_rows(args.data)
    if len(df) < args.normal + args.shifted:
        sys.exit(
            f"dataset has {len(df)} rows but script needs {args.normal + args.shifted}"
        )

    # Random sample so Phase 1 matches the training distribution (the CSV is
    # date-sorted; head() would always look drifted).
    sampled = df.sample(args.normal + args.shifted, random_state=42).reset_index(drop=True)
    normal_rows = sampled.iloc[: args.normal]
    shifted_rows = shift_rows(sampled.iloc[args.normal : args.normal + args.shifted])

    with httpx.Client() as client:
        print("Phase 1 — establish calm baseline")
        post_predictions(client, args.base_url, normal_rows, "phase 1")
        trigger_drift_check(client, args.base_url, "phase 1 drift check")

        if not args.no_pause:
            input("\n  Press Enter to trigger drift (Phase 2)...")

        print("\nPhase 2 — shifted traffic to trip drift")
        post_predictions(client, args.base_url, shifted_rows, "phase 2")
        # Small delay so the platform commits the last few prediction rows
        # before the drift check runs over the rolling window.
        time.sleep(1)
        trigger_drift_check(client, args.base_url, "phase 2 drift check")

    print("\nDone. Open http://localhost:8501 to watch the dashboard react.")


if __name__ == "__main__":
    main()