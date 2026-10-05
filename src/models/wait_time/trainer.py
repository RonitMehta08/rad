"""Wait-time prediction training pipeline with Optuna hyperparameter search.

Supports LightGBM, XGBoost, and Elastic Net with GPU acceleration,
early stopping, and MLflow logging.

Owner: P2 (ML Engineer)
Consumers: P2 (ensemble, evaluator), P5 (dashboard model info)
Reference: MASTER_PROMPT §7.1 Step 3, §16 Cmd 4-5

Usage:
    python -m src.models.wait_time.trainer --model lightgbm --device gpu \\
        --optuna-trials 100 --data data/processed/ --output models/wait_time/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import ENCODERS_FILENAME, FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.constants import (
    EARLY_STOPPING_PATIENCE,
    OPTUNA_N_TRIALS_DEFAULT,
    RANDOM_SEED,
)
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_regression_metrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature / Target Extraction
# ---------------------------------------------------------------------------


def _get_feature_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Extract feature matrix and target vector from a processed DataFrame.

    Args:
        df: Processed DataFrame from the preprocessor.

    Returns:
        Tuple of (X, y, feature_names).
    """
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_COLUMN].copy()

    # Drop rows where target is NaN
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    # Fill any remaining NaN features with 0
    X = X.fillna(0)

    return X, y, available


# ---------------------------------------------------------------------------
# Held-out Evaluation Helpers
# ---------------------------------------------------------------------------

RESIDUAL_LOWER_QUANTILE: float = 0.05
RESIDUAL_UPPER_QUANTILE: float = 0.95


def predict_with_package(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict with a plain model or an Elastic Net ``{'scaler', 'model'}`` package."""
    if isinstance(model, dict) and "scaler" in model:
        return model["model"].predict(model["scaler"].transform(X))
    return model.predict(X)


def compute_evaluation_extras(
    model: Any,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute validation residual quantiles (for prediction intervals) and test metrics.

    Args:
        model: Trained model or Elastic Net package.
        val_df: Validation split.
        test_df: Held-out test split, or None if unavailable.

    Returns:
        Dict with ``residual_quantiles`` and (if test data given) ``test_metrics``.
    """
    X_val, y_val, _ = _get_feature_target(val_df)
    residuals = y_val.values - np.clip(predict_with_package(model, X_val), 0, None)
    extras: dict[str, Any] = {
        "residual_quantiles": {
            "q05": float(np.quantile(residuals, RESIDUAL_LOWER_QUANTILE)),
            "q95": float(np.quantile(residuals, RESIDUAL_UPPER_QUANTILE)),
        },
    }
    if test_df is not None:
        X_test, y_test, _ = _get_feature_target(test_df)
        test_preds = np.clip(predict_with_package(model, X_test), 0, None)
        extras["test_metrics"] = compute_all_regression_metrics(y_test.values, test_preds)
    return extras


def copy_feature_encoders(data_dir: Path, output_dir: Path) -> None:
    """Copy ``feature_encoders.json`` next to the model so inference can rebuild features."""
    src = data_dir / ENCODERS_FILENAME
    if src.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, output_dir / ENCODERS_FILENAME)
    else:
        logger.warning(f"{src} not found — re-run the preprocessor so the API can accept raw patient inputs")


# ---------------------------------------------------------------------------
# LightGBM Trainer
# ---------------------------------------------------------------------------


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    device: str = "gpu",
    n_trials: int = OPTUNA_N_TRIALS_DEFAULT,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train LightGBM with Optuna hyperparameter optimization.

    Args:
        train_df: Training data (processed).
        val_df: Validation data (processed).
        device: 'gpu' or 'cpu'.
        n_trials: Number of Optuna trials.
        seed: Random seed.

    Returns:
        Tuple of (best_model, best_params, validation_metrics).
    """
    import lightgbm as lgb

    X_train, y_train, feature_names = _get_feature_target(train_df)
    X_val, y_val, _ = _get_feature_target(val_df)

    logger.info(f"LightGBM training: {X_train.shape[0]} train, {X_val.shape[0]} val, "
                f"{len(feature_names)} features, device={device}")

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_estimators": 5000,
            "random_state": seed,
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        if device == "gpu":
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(y_val.values - preds)))
        return mae

    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"LightGBM Optuna complete. Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params
    best_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "n_estimators": 5000,
        "random_state": seed,
        **study.best_params,
    }
    if device == "gpu":
        best_params["device"] = "gpu"
        best_params["gpu_platform_id"] = 0
        best_params["gpu_device_id"] = 0

    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = best_model.predict(X_val)
    val_metrics = compute_all_regression_metrics(y_val.values, val_preds)
    logger.info(f"LightGBM validation metrics: {val_metrics}")

    return best_model, best_params, val_metrics


# ---------------------------------------------------------------------------
# XGBoost Trainer
# ---------------------------------------------------------------------------


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    device: str = "gpu",
    n_trials: int = OPTUNA_N_TRIALS_DEFAULT,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train XGBoost with Optuna hyperparameter optimization.

    Args:
        train_df: Training data (processed).
        val_df: Validation data (processed).
        device: 'gpu' or 'cpu'.
        n_trials: Number of Optuna trials.
        seed: Random seed.

    Returns:
        Tuple of (best_model, best_params, validation_metrics).
    """
    import xgboost as xgb

    X_train, y_train, feature_names = _get_feature_target(train_df)
    X_val, y_val, _ = _get_feature_target(val_df)

    logger.info(f"XGBoost training: {X_train.shape[0]} train, {X_val.shape[0]} val, "
                f"{len(feature_names)} features, device={device}")

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "n_estimators": 5000,
            "random_state": seed,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        if device == "gpu":
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "hist"

        model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_PATIENCE)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(y_val.values - preds)))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"XGBoost Optuna complete. Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params
    best_params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "n_estimators": 5000,
        "random_state": seed,
        **study.best_params,
    }
    if device == "gpu":
        best_params["tree_method"] = "hist"
        best_params["device"] = "cuda"
    else:
        best_params["tree_method"] = "hist"

    best_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=EARLY_STOPPING_PATIENCE)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = best_model.predict(X_val)
    val_metrics = compute_all_regression_metrics(y_val.values, val_preds)
    logger.info(f"XGBoost validation metrics: {val_metrics}")

    return best_model, best_params, val_metrics


# ---------------------------------------------------------------------------
# Elastic Net Trainer (Baseline — CPU only)
# ---------------------------------------------------------------------------


def train_elastic_net(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Train Elastic Net baseline via GridSearchCV.

    Args:
        train_df: Training data.
        val_df: Validation data.
        seed: Random seed.

    Returns:
        Tuple of (best_model_pipeline, best_params, validation_metrics).

    Note:
        Returns a dict with 'scaler' and 'model' keys since Elastic Net
        requires feature scaling.
    """
    from sklearn.model_selection import GridSearchCV

    X_train, y_train, feature_names = _get_feature_target(train_df)
    X_val, y_val, _ = _get_feature_target(val_df)

    logger.info(f"Elastic Net training: {X_train.shape[0]} train, {X_val.shape[0]} val")

    # Scale features (required for regularized linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    grid = GridSearchCV(
        ElasticNet(random_state=seed, max_iter=10000),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    logger.info(f"Elastic Net best params: {best_params}")

    val_preds = best_model.predict(X_val_scaled)
    val_metrics = compute_all_regression_metrics(y_val.values, val_preds)
    logger.info(f"Elastic Net validation metrics: {val_metrics}")

    # Package scaler with model
    model_package = {"scaler": scaler, "model": best_model}

    return model_package, best_params, val_metrics


# ---------------------------------------------------------------------------
# Save / Load Utilities
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    output_dir: Path,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    feature_names: list[str],
    extras: dict[str, Any] | None = None,
) -> Path:
    """Save a trained model, its parameters, and metrics.

    Args:
        model: Trained model object.
        output_dir: Directory to save into.
        model_name: Model identifier (e.g., 'lightgbm_best').
        params: Best hyperparameters.
        metrics: Validation metrics.
        feature_names: Feature column names used.
        extras: Optional additional metadata (test metrics, residual quantiles).

    Returns:
        Path to the saved model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "params": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in params.items()},
        "metrics": metrics,
        **(extras or {}),
        "feature_names": feature_names,
    }
    meta_path = output_dir / f"{model_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save feature names separately for easy access
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return model_path


def load_model(model_path: Path) -> Any:
    """Load a saved model from disk.

    Args:
        model_path: Path to the .pkl file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train wait-time prediction models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lightgbm", "xgboost", "elastic_net", "all"],
                        help="Model to train")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Device for tree model training")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_N_TRIALS_DEFAULT,
                        help="Number of Optuna trials")
    parser.add_argument("--data", type=Path, default=Path("data/processed"),
                        help="Directory with train/val/test parquet files")
    parser.add_argument("--output", type=Path, default=Path("models/wait_time"),
                        help="Output directory for saved models")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # Load data
    train_df = pd.read_parquet(args.data / "train.parquet")
    val_df = pd.read_parquet(args.data / "val.parquet")
    test_path = args.data / "test.parquet"
    test_df = pd.read_parquet(test_path) if test_path.exists() else None
    _, _, feature_names = _get_feature_target(train_df)
    copy_feature_encoders(args.data, args.output)

    models_to_train = [args.model] if args.model != "all" else ["lightgbm", "xgboost", "elastic_net"]

    for model_name in models_to_train:
        logger.info(f"--- Training {model_name} ---")

        if model_name == "lightgbm":
            model, params, metrics = train_lightgbm(
                train_df, val_df, device=args.device,
                n_trials=args.optuna_trials, seed=args.seed,
            )
        elif model_name == "xgboost":
            model, params, metrics = train_xgboost(
                train_df, val_df, device=args.device,
                n_trials=args.optuna_trials, seed=args.seed,
            )
        elif model_name == "elastic_net":
            model, params, metrics = train_elastic_net(train_df, val_df, seed=args.seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        extras = compute_evaluation_extras(model, val_df, test_df)
        save_model(model, args.output, f"{model_name}_best", params, metrics, feature_names, extras)
        test_mae = extras.get("test_metrics", {}).get("mae", float("nan"))
        logger.info(f"{model_name} training complete. Validation MAE: {metrics['mae']:.4f}, "
                    f"Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
