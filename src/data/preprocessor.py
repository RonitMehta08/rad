"""Feature engineering pipeline for RadQueue AI.

Transforms raw patient records into ML-ready feature matrices with cyclical
encoding, rolling statistics, interaction features, lag features, and
target encoding.

Owner: P1 (Data & Config Lead)
Consumers: P2 (model training), P5 (inference pipeline)
Reference: MASTER_PROMPT §7.1 Step 2

Usage:
    python -m src.data.preprocessor --input data/generated/patients.csv \\
        --output data/processed/ --config config/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.constants import TRAIN_RATIO, VAL_RATIO
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN: str = "actual_wait_time_minutes"


# ---------------------------------------------------------------------------
# Cyclical Encoding (§7.1 Step 2)
# ---------------------------------------------------------------------------


def encode_cyclical(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Add sin/cos cyclical features for a periodic column.

    Args:
        df: Input DataFrame (modified in-place).
        column: Name of the column to encode.
        period: Period of the cycle (24 for hours, 7 for days).

    Returns:
        DataFrame with added {column}_sin and {column}_cos columns.
    """
    values = df[column].astype(float)
    df[f"{column}_sin"] = np.sin(2 * math.pi * values / period)
    df[f"{column}_cos"] = np.cos(2 * math.pi * values / period)
    return df


# ---------------------------------------------------------------------------
# Known-Wait Features (leakage-safe rolling + lag statistics, §7.1 Step 2)
# ---------------------------------------------------------------------------
#
# A patient's own wait (and the waits of patients still in the queue) are not
# observable when we predict at registration time. Wait-based features are
# therefore computed only from patients whose imaging had already STARTED
# strictly before the current patient's registration time.

ROLLING_WINDOWS_MINUTES: dict[str, int] = {"rolling_avg_wait_30min": 30, "rolling_avg_wait_1hr": 60}
ARRIVAL_RATE_WINDOW_MINUTES: int = 15
LAST_N_SERVED: int = 3
NANOSECONDS_PER_MINUTE: int = 60 * 1_000_000_000


def _to_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime-like Series to int64 nanoseconds."""
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype("int64")


def _known_wait_lookup(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (imaging_start_ns, waits, cumulative_waits) of served patients sorted by imaging start."""
    if "imaging_start_time" not in df.columns or TARGET_COLUMN not in df.columns:
        return np.array([], dtype="int64"), np.array([]), np.array([0.0])
    served = df.loc[df[TARGET_COLUMN].notna() & df["imaging_start_time"].notna(),
                    ["imaging_start_time", TARGET_COLUMN]]
    served = served.sort_values("imaging_start_time")
    waits = served[TARGET_COLUMN].to_numpy(dtype=float)
    return _to_ns(served["imaging_start_time"]), waits, np.concatenate([[0.0], np.cumsum(waits)])


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-safe rolling statistics.

    * ``rolling_avg_wait_{30min,1hr}``: mean wait of patients whose imaging started
      within the window before this patient's registration (NaN if none).
    * ``arrival_rate_last_15min``: patients who actually arrived (no-shows excluded)
      in the 15 minutes up to and including this registration, per minute.

    Args:
        df: DataFrame with ``registration_time`` (and ideally ``imaging_start_time``).

    Returns:
        DataFrame sorted by registration_time with rolling feature columns added.
    """
    df = df.sort_values("registration_time").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["registration_time"]):
        df["registration_time"] = pd.to_datetime(df["registration_time"])
    reg_ns = _to_ns(df["registration_time"])

    starts, _, cum = _known_wait_lookup(df)
    hi = np.searchsorted(starts, reg_ns, side="left")
    for col, minutes in ROLLING_WINDOWS_MINUTES.items():
        lo = np.searchsorted(starts, reg_ns - minutes * NANOSECONDS_PER_MINUTE, side="left")
        count = hi - lo
        df[col] = np.where(count > 0, (cum[hi] - cum[lo]) / np.maximum(count, 1), np.nan)

    if "showed_up" in df.columns:
        arrived = df["showed_up"].astype(bool).to_numpy()
    else:
        arrived = np.ones(len(df), dtype=bool)
    arrived_ns = np.sort(reg_ns[arrived])
    upper = np.searchsorted(arrived_ns, reg_ns, side="right")
    lower = np.searchsorted(arrived_ns, reg_ns - ARRIVAL_RATE_WINDOW_MINUTES * NANOSECONDS_PER_MINUTE, side="right")
    df["arrival_rate_last_15min"] = (upper - lower) / float(ARRIVAL_RATE_WINDOW_MINUTES)
    return df


# ---------------------------------------------------------------------------
# Lag Features (§7.1 Step 2)
# ---------------------------------------------------------------------------


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-safe lag features.

    ``wait_time_last_served_patient`` / ``wait_time_last_3_avg`` use the most
    recent patients whose imaging started before this registration.

    Args:
        df: DataFrame (any order; returned sorted by registration_time).

    Returns:
        DataFrame with lag feature columns added.
    """
    df = df.sort_values("registration_time").reset_index(drop=True)
    reg_ns = _to_ns(df["registration_time"])
    starts, waits, cum = _known_wait_lookup(df)
    hi = np.searchsorted(starts, reg_ns, side="left")
    lo = np.maximum(hi - LAST_N_SERVED, 0)
    has_any = hi > 0
    if len(waits):
        df["wait_time_last_served_patient"] = np.where(has_any, waits[np.maximum(hi - 1, 0)], np.nan)
    else:
        df["wait_time_last_served_patient"] = np.nan
    df["wait_time_last_3_avg"] = np.where(has_any, (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1), np.nan)

    if "current_queue_length_total" in df.columns:
        ql = df["current_queue_length_total"]
        # Approximate 15-min queue change using a lag of ~5 registrations (known at prediction time)
        df["queue_length_change_last_15min"] = (ql - ql.shift(5).fillna(ql)).astype(int)
    else:
        df["queue_length_change_last_15min"] = 0

    return df


# ---------------------------------------------------------------------------
# Interaction Features (§7.1 Step 2)
# ---------------------------------------------------------------------------


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features between existing columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with interaction features added.
    """
    # Utilization rate
    if "patients_in_service_count" in df.columns and "num_machines_available" in df.columns:
        machines = df["num_machines_available"].clip(lower=1)
        df["utilization_rate"] = (df["patients_in_service_count"] / machines).clip(0, 1)
    else:
        df["utilization_rate"] = 0.0

    # Congestion indicator: queue_length * utilization_rate
    if "current_queue_length_total" in df.columns:
        df["congestion_indicator"] = df["current_queue_length_total"] * df["utilization_rate"]
    else:
        df["congestion_indicator"] = 0.0

    # Modality-hour interaction
    if "modality" in df.columns and "hour_of_day" in df.columns:
        df["modality_hour_interaction"] = df["modality"].astype(str) + "_" + df["hour_of_day"].astype(str)

    # Visit-type-queue interaction
    if "visit_type" in df.columns and "current_queue_length_total" in df.columns:
        visit_map = {"walk_in": 1, "scheduled": 0, "emergency": 2}
        df["visit_queue_interaction"] = (
            df["visit_type"].map(visit_map).fillna(0) * df["current_queue_length_total"]
        )

    # Estimated remaining service for current patients
    if "patients_in_service_count" in df.columns and "avg_service_time_last_5_patients" in df.columns:
        df["estimated_remaining_service_minutes"] = (
            df["patients_in_service_count"] * df["avg_service_time_last_5_patients"] * 0.5
        )
    else:
        df["estimated_remaining_service_minutes"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Target Encoding (§7.1 Step 2)
# ---------------------------------------------------------------------------


def compute_target_encoding(
    df: pd.DataFrame,
    column: str,
    target: str = "actual_wait_time_minutes",
    smoothing: float = 10.0,
) -> tuple[pd.Series, dict[str, float]]:
    """Compute smoothed target encoding for a categorical column.

    Uses Bayesian smoothing: encoded = (count * mean_cat + smoothing * global_mean) / (count + smoothing)

    Args:
        df: Training DataFrame.
        column: Categorical column to encode.
        target: Target column name.
        smoothing: Smoothing factor (higher = more regularization).

    Returns:
        Tuple of (encoded Series, mapping dict for later application).
    """
    valid = df[[column, target]].dropna(subset=[target])
    global_mean = valid[target].mean()

    agg = valid.groupby(column)[target].agg(["mean", "count"])
    smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    mapping = smoothed.to_dict()

    encoded = df[column].map(mapping).fillna(global_mean)
    return encoded, mapping


# ---------------------------------------------------------------------------
# Categorical Encoding
# ---------------------------------------------------------------------------


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode or ordinal-encode categorical columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with encoded categorical features.
    """
    # Ordinal encode where order matters
    urgency_map = {"routine": 0, "urgent": 1, "emergency": 2}
    if "urgency" in df.columns:
        df["urgency_encoded"] = df["urgency"].map(urgency_map).fillna(0).astype(int)

    complexity_map = {"simple": 0, "moderate": 1, "complex": 2}
    if "exam_complexity" in df.columns:
        df["exam_complexity_encoded"] = df["exam_complexity"].map(complexity_map).fillna(0).astype(int)

    visit_map = {"scheduled": 0, "walk_in": 1, "emergency": 2}
    if "visit_type" in df.columns:
        df["visit_type_encoded"] = df["visit_type"].map(visit_map).fillna(0).astype(int)

    # Label encode modality (tree models handle this natively)
    modality_map = {"xray": 0, "ct": 1, "mri": 2, "ultrasound": 3}
    if "modality" in df.columns:
        df["modality_encoded"] = df["modality"].map(modality_map).fillna(0).astype(int)

    # Binary flags
    for col in ("requires_contrast", "requires_prep", "is_weekend", "is_holiday",
                "is_monday", "equipment_under_maintenance", "is_repeat_patient"):
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Gender
    if "gender" in df.columns:
        df["gender_encoded"] = df["gender"].map({"M": 0, "F": 1, "Other": 2}).fillna(0).astype(int)

    # Shift
    shift_map = {"morning": 0, "afternoon": 1, "evening": 2}
    if "shift_type" in df.columns:
        df["shift_type_encoded"] = df["shift_type"].map(shift_map).fillna(0).astype(int)

    # Age group
    age_map = {"0-18": 0, "18-40": 1, "40-60": 2, "60+": 3}
    if "age_group" in df.columns:
        df["age_group_encoded"] = df["age_group"].map(age_map).fillna(1).astype(int)

    # Insurance
    ins_map = {"government": 0, "private": 1, "self_pay": 2}
    if "insurance_type" in df.columns:
        df["insurance_type_encoded"] = df["insurance_type"].map(ins_map).fillna(0).astype(int)

    # Distance
    dist_map = {"local": 0, "city": 1, "outstation": 2}
    if "distance_category" in df.columns:
        df["distance_category_encoded"] = df["distance_category"].map(dist_map).fillna(0).astype(int)

    # Weather
    weather_map = {"summer": 0, "monsoon": 1, "winter": 2}
    if "weather_category" in df.columns:
        df["weather_category_encoded"] = df["weather_category"].map(weather_map).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Outlier Handling
# ---------------------------------------------------------------------------


def cap_outliers(df: pd.DataFrame, column: str, percentile: float = 99.0) -> pd.DataFrame:
    """Cap outliers at a given percentile (IQR method).

    Args:
        df: Input DataFrame (modified in-place).
        column: Column to cap.
        percentile: Upper percentile for capping.

    Returns:
        DataFrame with outliers capped.
    """
    if column in df.columns:
        cap_value = df[column].quantile(percentile / 100)
        df[column] = df[column].clip(upper=cap_value)
    return df


# ---------------------------------------------------------------------------
# Time-Based Train/Val/Test Split (§7.1 Step 3)
# ---------------------------------------------------------------------------


def time_based_split(
    df: pd.DataFrame,
    time_column: str = "registration_time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically: 70% train, 15% val, 15% test.

    This is a time-based split (NOT random) — essential for realistic evaluation.

    Args:
        df: Full dataset sorted by time.
        time_column: Datetime column to sort by.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values(time_column).reset_index(drop=True)
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature Column Selection
# ---------------------------------------------------------------------------

# Columns to keep as ML features (excludes timestamps, IDs, raw categoricals)
FEATURE_COLUMNS: list[str] = [
    # Encoded categoricals
    "modality_encoded", "visit_type_encoded", "urgency_encoded",
    "exam_complexity_encoded", "gender_encoded", "age_group_encoded",
    "insurance_type_encoded", "shift_type_encoded",
    "distance_category_encoded", "weather_category_encoded",
    # Binary flags
    "requires_contrast", "requires_prep", "is_weekend", "is_holiday",
    "is_monday", "equipment_under_maintenance", "is_repeat_patient",
    # Numeric — queue state
    "current_queue_length_total", "current_queue_length_same_modality",
    "patients_in_service_count", "avg_service_time_last_5_patients",
    "time_since_last_patient_served_minutes", "emergency_patients_in_queue",
    # Numeric — resources
    "num_machines_available", "num_technologists_on_duty", "num_radiologists_on_duty",
    # Numeric — patient
    "previous_no_show_count", "appointment_lead_time_days", "previous_appointment_count",
    # Time features
    "minutes_since_department_opened",
    # Cyclical
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    # Rolling
    "rolling_avg_wait_1hr", "rolling_avg_wait_30min", "arrival_rate_last_15min",
    # Lag
    "wait_time_last_served_patient", "wait_time_last_3_avg", "queue_length_change_last_15min",
    # Interaction
    "utilization_rate", "congestion_indicator", "visit_queue_interaction",
    "estimated_remaining_service_minutes",
    # Target encoding
    "modality_target_enc", "shift_type_target_enc", "exam_complexity_target_enc",
]

NOSHOW_TARGET_COLUMN: str = "showed_up"
ENCODERS_FILENAME: str = "feature_encoders.json"
OUTLIER_CAP_PERCENTILE: float = 99.0
TARGET_ENCODED_COLUMNS: list[tuple[str, str]] = [
    ("modality", "modality_target_enc"),
    ("shift_type", "shift_type_target_enc"),
    ("exam_complexity", "exam_complexity_target_enc"),
]


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def fit_feature_encoders(train_df: pd.DataFrame, smoothing: float = 10.0) -> dict[str, Any]:
    """Fit everything inference needs to rebuild features for a new patient.

    Args:
        train_df: Training split (after feature engineering, before column selection).
        smoothing: Target-encoding smoothing factor.

    Returns:
        JSON-serialisable dict with target-encoding maps, the global target mean
        and per-feature training medians (defaults for unknown context).
    """
    target_encodings: dict[str, dict[str, float]] = {}
    for col_name, enc_name in TARGET_ENCODED_COLUMNS:
        if col_name in train_df.columns:
            _, mapping = compute_target_encoding(train_df, col_name, TARGET_COLUMN, smoothing)
            target_encodings[enc_name] = {str(k): float(v) for k, v in mapping.items()}
    medians = {
        col: float(train_df[col].median())
        for col in FEATURE_COLUMNS
        if col in train_df.columns
        and pd.api.types.is_numeric_dtype(train_df[col])
        and train_df[col].notna().any()
    }
    return {
        "target_encodings": target_encodings,
        "global_mean": float(train_df[TARGET_COLUMN].dropna().mean()),
        "feature_medians": medians,
    }


def apply_target_encodings(df: pd.DataFrame, encoders: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted target encodings (unseen categories map to the training global mean)."""
    for col_name, enc_name in TARGET_ENCODED_COLUMNS:
        mapping = encoders["target_encodings"].get(enc_name)
        if mapping is not None and col_name in df.columns:
            df[enc_name] = df[col_name].astype(str).map(mapping).fillna(encoders["global_mean"]).astype(float)
    return df


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    return_encoders: bool = False,
) -> Any:
    """Run the complete feature engineering pipeline.

    Args:
        df: Raw patient records DataFrame.
        config: Optional model config dictionary.
        return_encoders: If True, also return the fitted feature encoders.

    Returns:
        Tuple of (train, val, test) DataFrames with feature columns + target,
        plus the encoders dict when ``return_encoders`` is True.
    """
    logger.info(f"Starting preprocessing pipeline on {len(df)} records")

    # 1. Parse timestamps
    for col in ["registration_time", "imaging_start_time", "imaging_end_time",
                 "departure_time", "queue_entry_time", "prep_start_time",
                 "reporting_start_time", "reporting_end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Cyclical + categorical encoding
    df = encode_cyclical(df, "hour_of_day", period=24)
    df = encode_cyclical(df, "day_of_week", period=7)
    df = encode_categoricals(df)

    # 3. Rolling + lag features (leakage-safe; only waits known at registration time)
    df = compute_rolling_features(df)
    df = compute_lag_features(df)

    # 4. Interaction features
    df = compute_interaction_features(df)

    # 5. Time-based split
    train_df, val_df, test_df = time_based_split(df)

    # 6. Cap target outliers using the TRAINING percentile only (no test information)
    cap_value = train_df[TARGET_COLUMN].quantile(OUTLIER_CAP_PERCENTILE / 100)
    for split in (train_df, val_df, test_df):
        split[TARGET_COLUMN] = split[TARGET_COLUMN].clip(upper=cap_value)

    # 7. Target encoding + medians (fit on train, apply to all splits)
    smoothing = 10.0
    if config and "feature_engineering" in config:
        smoothing = config["feature_engineering"].get("target_encoding_smoothing", 10.0)
    encoders = fit_feature_encoders(train_df, smoothing)
    for split in (train_df, val_df, test_df):
        apply_target_encodings(split, encoders)

    # 8. Fill missing values with TRAINING medians
    for split in (train_df, val_df, test_df):
        for col in FEATURE_COLUMNS:
            if col in split.columns:
                split[col] = split[col].fillna(encoders["feature_medians"].get(col, 0.0))

    # 9. Select final columns
    available_features = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    extra_cols = [TARGET_COLUMN, NOSHOW_TARGET_COLUMN, "patient_id", "registration_time", "modality", "visit_type"]
    keep_cols = available_features + [c for c in extra_cols if c in train_df.columns]

    train_df = train_df[keep_cols].copy()
    val_df = val_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()

    logger.info(f"Preprocessing complete. Features: {len(available_features)}")
    logger.info(f"Final shapes — train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    if return_encoders:
        return train_df, val_df, test_df, encoders
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/generated/patients.csv"),
                        help="Input CSV file path")
    parser.add_argument("--output", type=Path, default=Path("data/processed"),
                        help="Output directory for parquet files")
    parser.add_argument("--config", type=Path, default=Path("config/model_config.yaml"),
                        help="Model config YAML path")
    args = parser.parse_args()

    # Load config
    config: dict[str, Any] = {}
    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(args.input, parse_dates=["registration_time"])
    logger.info(f"Loaded {len(df)} records from {args.input}")

    # Run pipeline
    train_df, val_df, test_df, encoders = run_preprocessing_pipeline(df, config, return_encoders=True)

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.output / "train.parquet", index=False)
    val_df.to_parquet(args.output / "val.parquet", index=False)
    test_df.to_parquet(args.output / "test.parquet", index=False)
    with open(args.output / ENCODERS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2)

    logger.info(f"Saved processed splits to {args.output}")


if __name__ == "__main__":
    main()
