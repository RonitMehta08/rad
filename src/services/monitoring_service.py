"""Model monitoring service: data drift (PSI/KS/chi²) and accuracy tracking.

Owner: P5 (Dashboard & API) on top of P2's monitoring module
Reference: MASTER_PROMPT §5.3 Module 4, USP 8
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.monitoring.drift_detector import DriftDetector
from src.services.model_registry import PROJECT_ROOT, ModelRegistry
from src.utils.exceptions import ModelNotAvailableError

DEFAULT_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DRIFT_FEATURES: list[str] = [
    "current_queue_length_total", "current_queue_length_same_modality", "arrival_rate_last_15min",
    "rolling_avg_wait_30min", "utilization_rate", "emergency_patients_in_queue",
    "minutes_since_department_opened", "modality_encoded", "visit_type_encoded", "urgency_encoded",
]
MAX_DRIFT_ROWS: int = 5_000


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise ModelNotAvailableError(
            f"{path} not found. Run: python -m src.data.generator && python -m src.data.preprocessor"
        )
    return pd.read_parquet(path)


def run_drift_report(
    registry: ModelRegistry,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    current: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare a current window (default: test split) against training data.

    Also runs ADWIN/CUSUM concept-drift tests on the model's absolute errors
    over the current window when a wait-time model is available.

    Returns:
        Serializable drift report with per-feature results.
    """
    train = _load_split(processed_dir, "train").tail(MAX_DRIFT_ROWS)
    current = current if current is not None else _load_split(processed_dir, "test")
    features = [f for f in DRIFT_FEATURES if f in train.columns and f in current.columns]
    detector = DriftDetector(train[features], numerical_features=features)

    errors = None
    predictor = registry.try_wait_time_predictor()
    if predictor is not None and TARGET_COLUMN in current.columns:
        served = current[current[TARGET_COLUMN].notna()]
        X = served[[c for c in FEATURE_COLUMNS if c in served.columns]]
        errors = np.abs(served[TARGET_COLUMN].to_numpy() - predictor.predict_batch(X))

    report = detector.run_full_drift_check(current[features], prediction_errors=errors)
    out = asdict(report)
    for r in out["drift_results"]:
        r["statistic"] = float(r["statistic"]) if r["statistic"] is not None else None
        r["p_value"] = float(r["p_value"]) if r["p_value"] is not None else None
        r["is_drifted"] = bool(r["is_drifted"])
    if errors is not None:
        out["current_window_mae"] = float(np.mean(errors))
    return out


def rolling_accuracy(registry: ModelRegistry, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Daily MAE of the deployed wait-time model on the held-out test period."""
    predictor = registry.wait_time_predictor
    test = _load_split(processed_dir, "test")
    test = test[test[TARGET_COLUMN].notna()].copy()
    X = test[[c for c in FEATURE_COLUMNS if c in test.columns]]
    test["abs_error"] = np.abs(test[TARGET_COLUMN].to_numpy() - predictor.predict_batch(X))
    test["date"] = pd.to_datetime(test["registration_time"]).dt.date
    return test.groupby("date")["abs_error"].agg(mae="mean", patients="count").reset_index()
