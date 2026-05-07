"""Drift statistics and alerting.

File summary:
- Builds or loads reference feature/output distributions from training data.
- Compares recent prediction inputs and outputs against that reference window.
- Computes numeric PSI, categorical chi-square drift, and output PSI.
- Saves drift reports and sends webhook alerts when severity meaningfully changes.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import DriftAlert, DriftReport, Prediction, ReferenceStatistics
from app.ml.artifacts import compute_file_md5
from app.ml.data import clean_bank_marketing_data, load_bank_marketing_data
from app.models.drift import DriftCheckResponse, ReferenceStatsResponse
from app.services.webhook_service import WebhookService

SEVERITY_ORDER = {
    "insufficient_data": 0,
    "none": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}
ALERT_SEVERITIES = {"medium", "high", "critical"}


class DriftService:
    """Coordinate reference statistics, drift checks, report persistence, and alerting."""

    def __init__(self, settings: Settings) -> None:
        """Store platform settings used by drift calculations and webhooks."""
        self.settings = settings

    def recompute_reference(self, db: Session) -> ReferenceStatsResponse:
        """Rebuild active reference statistics from the configured training dataset."""
        raw = load_bank_marketing_data(self.settings.data_path)
        cleaned = clean_bank_marketing_data(raw)
        features = cleaned.drop(columns=["y"], errors="ignore")
        numeric_stats: dict[str, Any] = {}
        categorical_stats: dict[str, Any] = {}

        for column in features.columns:
            series = features[column]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
                quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
                if quantiles.size < 2:
                    quantiles = np.array([float(values.min()), float(values.max()) + 1])
                numeric_stats[column] = {
                    "values": values.tolist(),
                    "bin_edges": quantiles.tolist(),
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            else:
                counts = series.astype(str).value_counts(normalize=True).to_dict()
                categorical_stats[column] = {str(k): float(v) for k, v in counts.items()}

        output_counts = cleaned["y"].astype(int).value_counts(normalize=True).to_dict()
        output_stats = {str(k): float(v) for k, v in output_counts.items()}

        db.query(ReferenceStatistics).filter(ReferenceStatistics.is_active.is_(True)).update(
            {"is_active": False}
        )
        row = ReferenceStatistics(
            model_name=self.settings.mlflow_registered_model_name,
            model_version=self.settings.model_version_label,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            output_stats=output_stats,
            dataset_hash=compute_file_md5(self.settings.data_path),
            is_active=True,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return ReferenceStatsResponse(
            reference_id=str(row.id),
            model_name=row.model_name,
            model_version=row.model_version,
            numeric_features=list(numeric_stats),
            categorical_features=list(categorical_stats),
        )

    def check_drift(self, db: Session) -> DriftCheckResponse:
        """Compare recent predictions to reference stats and optionally emit an alert."""
        predictions = (
            db.query(Prediction)
            .order_by(desc(Prediction.created_at))
            .limit(self.settings.drift_window_size)
            .all()
        )
        predictions = list(reversed(predictions))
        previous = db.query(DriftReport).order_by(desc(DriftReport.created_at)).first()
        previous_severity = previous.severity if previous else None

        if len(predictions) < self.settings.drift_min_window_size:
            return DriftCheckResponse(
                drift_report_id=None,
                severity="insufficient_data",
                previous_severity=previous_severity,
                window_size=len(predictions),
                numeric_psi={},
                categorical_chi2={},
                output_drift={},
            )

        reference = self._get_or_create_reference(db)
        recent_inputs = pd.DataFrame([p.input_json for p in predictions])
        numeric_psi = self._numeric_psi(reference.numeric_stats, recent_inputs)
        categorical = self._categorical_chi2(reference.categorical_stats, recent_inputs)
        output = self._output_drift(reference.output_stats or {}, predictions)
        severity = self._overall_severity(numeric_psi, categorical, output)

        report = DriftReport(
            model_name=self.settings.mlflow_registered_model_name,
            model_version=predictions[-1].model_version,
            window_start=predictions[0].created_at,
            window_end=predictions[-1].created_at,
            window_size=len(predictions),
            numeric_psi=numeric_psi,
            categorical_chi2=categorical,
            output_drift=output,
            severity=severity,
            previous_severity=previous_severity,
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        alert_info = None
        if (
            severity in ALERT_SEVERITIES
            and severity != previous_severity
            and previous_severity != "insufficient_data"
        ):
            alert = self._create_alert(db, report)
            # bridge sync drift_service into async webhook delivery (Engineering Standards Ch. 1)
            asyncio.run(WebhookService(self.settings).send_drift_alert(alert))
            db.commit()
            db.refresh(alert)
            alert_info = {
                "event_id": alert.event_id,
                "status": alert.status,
                "response_status": alert.response_status,
                "error_message": alert.error_message,
            }

        return DriftCheckResponse(
            drift_report_id=str(report.id),
            severity=severity,
            previous_severity=previous_severity,
            window_size=len(predictions),
            numeric_psi=numeric_psi,
            categorical_chi2=categorical,
            output_drift=output,
            alert=alert_info,
        )

    def list_reports(self, db: Session, limit: int = 25) -> list[dict[str, Any]]:
        """Return recent drift reports in compact dictionary form."""
        rows = db.query(DriftReport).order_by(desc(DriftReport.created_at)).limit(limit).all()
        return [
            {
                "id": str(row.id),
                "model_name": row.model_name,
                "model_version": row.model_version,
                "window_size": row.window_size,
                "severity": row.severity,
                "previous_severity": row.previous_severity,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def _get_or_create_reference(self, db: Session) -> ReferenceStatistics:
        """Return the active reference stats row, creating it when missing."""
        row = (
            db.query(ReferenceStatistics)
            .filter(ReferenceStatistics.is_active.is_(True))
            .order_by(desc(ReferenceStatistics.created_at))
            .first()
        )
        if row is None:
            self.recompute_reference(db)
            row = (
                db.query(ReferenceStatistics)
                .filter(ReferenceStatistics.is_active.is_(True))
                .order_by(desc(ReferenceStatistics.created_at))
                .first()
            )
        assert row is not None
        return row

    def _numeric_psi(
        self, reference_stats: dict[str, Any], recent_inputs: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute PSI and severity for numeric features present in recent inputs."""
        out: dict[str, Any] = {}
        for column, stats in reference_stats.items():
            if column not in recent_inputs:
                continue
            current = pd.to_numeric(recent_inputs[column], errors="coerce").dropna()
            value = psi(np.array(stats["values"]), current.to_numpy(), stats["bin_edges"])
            out[column] = {
                "psi": value,
                "severity": self._psi_severity(value),
            }
        return out

    def _categorical_chi2(
        self, reference_stats: dict[str, Any], recent_inputs: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute chi-square drift flags for categorical features."""
        out: dict[str, Any] = {}
        for column, proportions in reference_stats.items():
            if column not in recent_inputs:
                continue
            current_counts = recent_inputs[column].astype(str).value_counts()
            categories = sorted(set(proportions) | set(current_counts.index))
            observed = np.array([float(current_counts.get(cat, 0.0)) for cat in categories])
            total = max(float(observed.sum()), 1.0)
            expected = np.array([float(proportions.get(cat, 0.0)) * total for cat in categories])
            expected = np.clip(expected, 1e-6, None)
            expected = expected * (observed.sum() / expected.sum())
            stat, pvalue = chisquare(f_obs=observed, f_exp=expected)
            out[column] = {
                "chi2": float(stat),
                "pvalue": float(pvalue),
                "drifted": bool(pvalue < self.settings.chi2_pvalue_threshold),
            }
        return out

    def _output_drift(
        self, reference_output: dict[str, float], predictions: list[Prediction]
    ) -> dict[str, Any]:
        """Compute PSI and severity for the predicted output distribution."""
        current_values = np.array([p.prediction for p in predictions], dtype=int)
        ref_values: list[int] = []
        for key, proportion in reference_output.items():
            ref_values.extend([int(key)] * max(int(proportion * 1000), 1))
        value = psi(np.array(ref_values), current_values, [-0.5, 0.5, 1.5])
        return {
            "psi": value,
            "severity": self._psi_severity(value, output=True),
        }

    def _overall_severity(
        self,
        numeric_psi: dict[str, Any],
        categorical_chi2: dict[str, Any],
        output_drift: dict[str, Any],
    ) -> str:
        """Collapse feature-level drift signals into one overall severity label."""
        severities = [item["severity"] for item in numeric_psi.values()]
        severities.append(output_drift.get("severity", "none"))
        drifted_cats = sum(1 for item in categorical_chi2.values() if item["drifted"])
        if drifted_cats >= 4:
            severities.append("high")
        elif drifted_cats >= 2:
            severities.append("medium")
        elif drifted_cats == 1:
            severities.append("low")
        return max(severities or ["none"], key=lambda item: SEVERITY_ORDER[item])

    def _psi_severity(self, value: float, *, output: bool = False) -> str:
        """Map a PSI value to the configured severity thresholds."""
        if output and value >= self.settings.output_drift_psi_threshold:
            return "medium"
        if value >= self.settings.psi_high_threshold:
            return "high"
        if value >= self.settings.psi_medium_threshold:
            return "medium"
        if value >= self.settings.psi_low_threshold:
            return "low"
        return "none"

    def _create_alert(self, db: Session, report: DriftReport) -> DriftAlert:
        """Create and persist the webhook alert payload for one drift report."""
        event_id = f"drift_{uuid.uuid4().hex}"
        payload = {
            "contract_version": "v1",
            "event_id": event_id,
            "drift_report_id": str(report.id),
            "model_name": report.model_name,
            "model_version": report.model_version,
            "severity": report.severity,
            "previous_severity": report.previous_severity,
            "window": {
                "size": report.window_size,
                "start": report.window_start.isoformat() if report.window_start else None,
                "end": report.window_end.isoformat() if report.window_end else None,
            },
            "signals": {
                "numeric_psi": report.numeric_psi,
                "categorical_chi2": report.categorical_chi2,
                "output_drift": report.output_drift,
            },
            "recommended_actions": ["replay_test_set", "retrain_candidate"],
            "created_at": datetime.now(UTC).isoformat(),
        }
        alert = DriftAlert(
            drift_report_id=report.id,
            event_id=event_id,
            severity=report.severity,
            status="pending",
            webhook_payload=payload,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert


def psi(reference: np.ndarray, current: np.ndarray, bin_edges: list[float]) -> float:
    """Population Stability Index, using the notebook's clipped-bin formula."""
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.array(bin_edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_p = np.histogram(reference, bins=edges)[0] / max(reference.size, 1)
    cur_p = np.histogram(current, bins=edges)[0] / max(current.size, 1)
    eps = 1e-6
    ref_p = np.clip(ref_p, eps, 1.0)
    cur_p = np.clip(cur_p, eps, 1.0)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
