"""No-show prediction training — XGBoost classifier with SMOTE and Platt calibration.

Owner: P2 (ML Engineer)
Consumers: P3 (scheduler overbooking), P5 (dashboard)
Reference: MASTER_PROMPT §7.2, §16 Cmd 7

Usage:
    python -m src.models.noshow.trainer --device gpu --optuna-trials 50 \\
        --data data/processed/ --output models/noshow/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.data.preprocessor import NOSHOW_TARGET_COLUMN
from src.utils.constants import RANDOM_SEED
from src.utils.logger import get_logger
from src.utils.metrics import compute_classification_metrics

logger = get_logger(__name__)

# Features relevant to no-show prediction (subset of full features)
NOSHOW_FEATURES: list[str] = [
    "appointment_lead_time_days",
    "modality_encoded",
    "hour_of_day_sin", "hour_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
    "age_group_encoded",
    "gender_encoded",
    "previous_no_show_count",
    "previous_appointment_count",
    "insurance_type_encoded",
    "is_repeat_patient",
    "weather_category_encoded",
    "distance_category_encoded",
    "is_weekend",
    "is_holiday",
    "is_monday",
    "exam_complexity_encoded",
]

# Only pre-booked (scheduled) appointments can be no-shows; walk-ins and
# emergencies are present by definition and would make the task trivial.
SCHEDULED_VISIT_CODE: int = 0  # visit_type_encoded for "scheduled"


def _get_noshow_feature_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Extract no-show features and binary target.

    Args:
        df: Processed DataFrame.

    Returns:
        Tuple of (X, y, feature_names).
    """
    available = [c for c in NOSHOW_FEATURES if c in df.columns]

    if NOSHOW_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{NOSHOW_TARGET_COLUMN}' not found in DataFrame")

    if "visit_type_encoded" in df.columns:
        df = df.loc[df["visit_type_encoded"] == SCHEDULED_VISIT_CODE]

    X = df[available].copy()
    y = df[NOSHOW_TARGET_COLUMN].copy()

    # showed_up=True → not a no-show → label=0; showed_up=False → no-show → label=1
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    # Invert: we predict no-show (1 = no-show, 0 = showed up)
    y = 1 - y  # showed_up=True(1)→0(showed), showed_up=False(0)→1(noshow)

    X = X.fillna(0)
    return X, y, available


# ---------------------------------------------------------------------------
# SMOTE Oversampling
# ---------------------------------------------------------------------------


def _apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to balance classes.

    Args:
        X: Feature matrix.
        y: Binary target.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE applied: {len(X)} → {len(X_res)} samples")
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except ImportError:
        logger.warning("imblearn not installed. Skipping SMOTE. Install with: pip install imbalanced-learn")
        return X, y


# ---------------------------------------------------------------------------
# XGBoost Classifier Training with Optuna
# ---------------------------------------------------------------------------


def train_noshow_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    device: str = "gpu",
    n_trials: int = 50,
    use_smote: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Train XGBoost no-show classifier with Optuna and Platt calibration.

    Args:
        train_df: Training data.
        val_df: Validation data.
        device: 'gpu' or 'cpu'.
        n_trials: Optuna trials.
        use_smote: Whether to apply SMOTE.
        seed: Random seed.

    Returns:
        Tuple of (best_model, calibrator, best_params, val_metrics).
    """
    import xgboost as xgb

    X_train, y_train, feature_names = _get_noshow_feature_target(train_df)
    X_val, y_val, _ = _get_noshow_feature_target(val_df)

    logger.info(f"No-show training: {len(X_train)} train, {len(X_val)} val, "
                f"no-show rate: {y_train.mean():.2%}")

    # Apply SMOTE
    if use_smote:
        X_train_sm, y_train_sm = _apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 2000,
            "random_state": seed,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }

        if device == "gpu":
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "hist"

        model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(
            X_train_sm, y_train_sm,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        from sklearn.metrics import roc_auc_score
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"No-show Optuna complete. Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params
    best_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "random_state": seed,
        **study.best_params,
    }
    if device == "gpu":
        best_params["tree_method"] = "hist"
        best_params["device"] = "cuda"
    else:
        best_params["tree_method"] = "hist"

    best_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=30)
    best_model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Platt calibration (§7.2 Step 2) — fitted on validation; judged on the test split later
    logger.info("Applying Platt calibration...")
    calibrator = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid", cv=5)
    calibrator.fit(X_val, y_val)

    # Choose the decision threshold that maximises F1 on validation, then report at it
    y_proba = calibrator.predict_proba(X_val)[:, 1]
    decision_threshold = compute_classification_metrics(y_val.values, y_proba)["optimal_threshold"]
    val_metrics = compute_classification_metrics(y_val.values, y_proba, threshold=decision_threshold)
    logger.info(f"No-show validation metrics: AUC-ROC={val_metrics['auc_roc']:.4f}, "
                f"AUC-PR={val_metrics['auc_pr']:.4f}, F1={val_metrics['f1']:.4f}")

    return best_model, calibrator, best_params, val_metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for no-show model training."""
    parser = argparse.ArgumentParser(description="Train no-show prediction model")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/noshow"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.data / "train.parquet")
    val_df = pd.read_parquet(args.data / "val.parquet")

    model, calibrator, params, metrics = train_noshow_model(
        train_df, val_df, device=args.device,
        n_trials=args.optuna_trials, seed=args.seed,
    )

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "xgboost_noshow.pkl")
    joblib.dump(calibrator, args.output / "calibrator.pkl")

    # Save metadata
    meta: dict[str, Any] = {
        "params": {k: str(v) for k, v in params.items()},
        "decision_threshold": metrics["threshold_used"],
        "training_population": "scheduled appointments only",
        "metrics": metrics,
    }
    test_path = args.data / "test.parquet"
    if test_path.exists():
        X_test, y_test, _ = _get_noshow_feature_target(pd.read_parquet(test_path))
        test_proba = calibrator.predict_proba(X_test)[:, 1]
        meta["test_metrics"] = compute_classification_metrics(
            y_test.values, test_proba, threshold=metrics["threshold_used"],
        )
        logger.info(f"No-show TEST metrics: AUC-ROC={meta['test_metrics']['auc_roc']:.4f}, "
                    f"AUC-PR={meta['test_metrics']['auc_pr']:.4f}, F1={meta['test_metrics']['f1']:.4f}")
    with open(args.output / "noshow_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"No-show model saved to {args.output}")


if __name__ == "__main__":
    main()
