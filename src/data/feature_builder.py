"""Inference-time feature builder.

Turns a raw patient description plus (optional) live department context into
the exact engineered feature row the wait-time / no-show models were trained on,
reusing the same encoding functions as the offline preprocessor so training and
serving cannot drift apart.

Owner: P1 (Data & Config Lead)
Consumers: P2 (predictors), P3 (dispatcher), P5 (API / dashboard)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from src.data.indian_context import generate_holiday_features, get_weather_category
from src.data.preprocessor import (
    FEATURE_COLUMNS,
    apply_target_encodings,
    compute_interaction_features,
    encode_categoricals,
    encode_cyclical,
)
from src.utils.constants import DEPARTMENT_OPEN_HOUR

MINUTES_PER_HOUR: int = 60
MORNING_SHIFT_END_HOUR: int = 14
AFTERNOON_SHIFT_END_HOUR: int = 20

# Raw patient fields and their defaults when the caller omits them.
PATIENT_DEFAULTS: dict[str, Any] = {
    "modality": "xray",
    "visit_type": "walk_in",
    "urgency": "routine",
    "exam_complexity": "simple",
    "gender": "M",
    "age_group": "18-40",
    "insurance_type": "government",
    "distance_category": "local",
    "requires_contrast": False,
    "requires_prep": False,
    "previous_no_show_count": 0,
    "previous_appointment_count": 0,
    "appointment_lead_time_days": 0,
    "equipment_under_maintenance": False,
}


def _shift_for_hour(hour: int) -> str:
    """Map an hour of day to the staff shift name used in training data."""
    if hour < MORNING_SHIFT_END_HOUR:
        return "morning"
    if hour < AFTERNOON_SHIFT_END_HOUR:
        return "afternoon"
    return "evening"


def _calendar_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive time/calendar features from ``timestamp`` (or hour/day fields, or now)."""
    ts = raw.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts is None:
        ts = datetime.now()
    hour = int(raw.get("hour_of_day", ts.hour))
    minute = int(raw.get("minute", ts.minute if "hour_of_day" not in raw else 0))
    day_of_week = int(raw.get("day_of_week", ts.weekday()))
    day: date = ts.date()
    return {
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": day_of_week >= 5,
        "is_monday": day_of_week == 0,
        "is_holiday": bool(raw.get("is_holiday", generate_holiday_features(day)["is_holiday"])),
        "weather_category": raw.get("weather_category", get_weather_category(day).value),
        "shift_type": raw.get("shift_type", _shift_for_hour(hour)),
        "minutes_since_department_opened": raw.get(
            "minutes_since_department_opened",
            max(0, (hour - DEPARTMENT_OPEN_HOUR) * MINUTES_PER_HOUR + minute),
        ),
    }


def build_feature_frame(
    raw: dict[str, Any],
    encoders: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a single-row model-ready feature DataFrame.

    Args:
        raw: Patient fields (see ``PATIENT_DEFAULTS``) plus optional queue/resource
            context using the training column names, e.g.
            ``current_queue_length_same_modality``, ``num_machines_available``,
            ``rolling_avg_wait_30min``. An optional ``timestamp`` drives calendar features.
        encoders: Output of ``fit_feature_encoders`` (``feature_encoders.json``).
            Supplies target encodings and training medians for unknown context.

    Returns:
        DataFrame with exactly ``FEATURE_COLUMNS`` (one row).
    """
    medians = (encoders or {}).get("feature_medians", {})
    row: dict[str, Any] = {**PATIENT_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}
    row.update({k: v for k, v in _calendar_fields(row).items() if k not in raw or raw[k] is None})
    row.setdefault("is_repeat_patient", row["previous_appointment_count"] > 0)

    df = pd.DataFrame([row])
    df = encode_cyclical(df, "hour_of_day", period=24)
    df = encode_cyclical(df, "day_of_week", period=7)
    df = encode_categoricals(df)

    # Unknown live context -> typical (training median) department state.
    for col in FEATURE_COLUMNS:
        if col not in df.columns or pd.isna(df.at[0, col]):
            df[col] = medians.get(col, 0.0)

    df = compute_interaction_features(df)
    if encoders:
        df = apply_target_encodings(df, encoders)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)
    return df[FEATURE_COLUMNS].astype(float)
