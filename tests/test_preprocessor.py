"""Tests for the feature engineering pipeline — especially leakage safety."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.feature_builder import build_feature_frame
from src.data.preprocessor import (
    FEATURE_COLUMNS,
    compute_lag_features,
    compute_rolling_features,
    run_preprocessing_pipeline,
)

T0 = datetime(2024, 1, 1, 9, 0)


def _patients(rows: list[tuple[int, int | None, float | None]]) -> pd.DataFrame:
    """rows: (registration minute, imaging-start minute, wait)."""
    return pd.DataFrame({
        "registration_time": [T0 + timedelta(minutes=r) for r, _, _ in rows],
        "imaging_start_time": [T0 + timedelta(minutes=s) if s is not None else pd.NaT for _, s, _ in rows],
        "actual_wait_time_minutes": [w for _, _, w in rows],
        "showed_up": [w is not None for _, _, w in rows],
    })


def test_should_not_include_own_wait_in_rolling_average() -> None:
    df = compute_rolling_features(_patients([(0, 100, 999.0)]))
    assert np.isnan(df.loc[0, "rolling_avg_wait_30min"])


def test_should_only_use_waits_of_patients_already_imaged() -> None:
    # Patient A registered at 0 and was imaged at 50 (wait 50). Patient B registers at 20:
    # A is still waiting, so A's wait must NOT be visible to B.
    df = _patients([(0, 50, 50.0), (20, 70, 50.0), (60, 80, 20.0)])
    rolled = compute_rolling_features(df)
    lagged = compute_lag_features(df)
    assert np.isnan(rolled.loc[1, "rolling_avg_wait_30min"])
    assert np.isnan(lagged.loc[1, "wait_time_last_served_patient"])
    # At minute 60 A (imaged at 50) is known
    assert rolled.loc[2, "rolling_avg_wait_30min"] == 50.0
    assert lagged.loc[2, "wait_time_last_served_patient"] == 50.0


def test_should_exclude_noshows_from_arrival_rate() -> None:
    df = compute_rolling_features(_patients([(0, 5, 5.0), (1, None, None), (2, 10, 8.0)]))
    assert df.loc[2, "arrival_rate_last_15min"] == 2 / 15


def test_should_split_chronologically_and_fill_with_train_medians() -> None:
    rng = np.random.default_rng(0)
    n = 400
    reg = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    df = pd.DataFrame({
        "registration_time": reg,
        "imaging_start_time": [r + timedelta(minutes=float(w)) for r, w in zip(reg, rng.uniform(1, 60, n))],
        "actual_wait_time_minutes": rng.uniform(1, 60, n),
        "showed_up": True,
        "hour_of_day": [r.hour for r in reg], "day_of_week": [r.weekday() for r in reg],
        "modality": rng.choice(["xray", "ct", "mri", "ultrasound"], n),
        "shift_type": "morning", "exam_complexity": "simple", "visit_type": "walk_in",
        "current_queue_length_total": rng.integers(0, 20, n),
    })
    train, val, test, enc = run_preprocessing_pipeline(df, return_encoders=True)
    assert train["registration_time"].max() <= val["registration_time"].min()
    assert val["registration_time"].max() <= test["registration_time"].min()
    assert not train[[c for c in FEATURE_COLUMNS if c in train]].isna().any().any()
    assert set(enc["target_encodings"]["modality_target_enc"]) <= {"xray", "ct", "mri", "ultrasound"}


def test_should_build_full_feature_row_from_raw_patient() -> None:
    row = build_feature_frame({"modality": "mri", "urgency": "emergency", "timestamp": "2024-01-01T10:15:00"})
    assert list(row.columns) == FEATURE_COLUMNS
    assert row.loc[0, "urgency_encoded"] == 2
    assert row.loc[0, "is_monday"] == 1
    assert row.loc[0, "minutes_since_department_opened"] == 135
