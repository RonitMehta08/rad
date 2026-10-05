"""Stacking ensemble — Ridge meta-learner on OOF predictions from base models.

Owner: P2 (ML Engineer)
Consumers: P2 (evaluator), P5 (dashboard)
Reference: MASTER_PROMPT §7.1 Step 3 — Model 4

Usage:
    python -m src.models.wait_time.ensemble --data data/processed/ \\
        --models models/wait_time/ --output models/wait_time/ensemble_best.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.constants import RANDOM_SEED
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_regression_metrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# OOF Prediction Generator
# ---------------------------------------------------------------------------


def _generate_oof_predictions(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    is_elastic_net: bool = False,
) -> np.ndarray:
    """Generate out-of-fold predictions for stacking.

    Args:
        model: A model object that supports .fit() and .predict().
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of CV folds.
        seed: Random seed for KFold.
        is_elastic_net: If True, applies StandardScaler before fitting.

    Returns:
        Array of OOF predictions (same length as X).
    """
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train = X.iloc[train_idx].copy()
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx].copy()

        if is_elastic_net:
            scaler = StandardScaler()
            X_fold_train = pd.DataFrame(
                scaler.fit_transform(X_fold_train),
                columns=X_fold_train.columns,
                index=X_fold_train.index,
            )
            X_fold_val = pd.DataFrame(
                scaler.transform(X_fold_val),
                columns=X_fold_val.columns,
                index=X_fold_val.index,
            )

        import copy
        fold_model = copy.deepcopy(model)

        # Fit with early stopping for tree models
        try:
            fold_model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y.iloc[val_idx])],
                verbose=False,
            )
        except TypeError:
            # ElasticNet and models without eval_set
            fold_model.fit(X_fold_train, y_fold_train)

        oof_preds[val_idx] = fold_model.predict(X_fold_val)

    return oof_preds


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------


class StackingEnsemble:
    """Stacking ensemble with Ridge meta-learner.

    Combines LightGBM, XGBoost, and ElasticNet predictions using
    a Ridge regression meta-learner trained on OOF predictions.

    Attributes:
        base_models: Dict of {name: model} for base learners.
        meta_learner: Trained Ridge meta-learner.
        feature_names: Feature column names.
    """

    def __init__(self) -> None:
        self.base_models: dict[str, Any] = {}
        self.meta_learner: RidgeCV | None = None
        self.feature_names: list[str] = []
        self._is_elastic_net: dict[str, bool] = {}
        self._scalers: dict[str, Any] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_dir: Path,
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Fit the stacking ensemble.

        Loads base models, generates OOF predictions, and trains the meta-learner.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            model_dir: Directory containing base model .pkl files.
            n_splits: Number of CV folds for OOF generation.

        Returns:
            Validation metrics dictionary.
        """
        available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
        self.feature_names = available

        # Prepare data
        X_train = train_df[available].fillna(0)
        y_train = train_df[TARGET_COLUMN].copy()
        valid_mask = y_train.notna()
        X_train = X_train.loc[valid_mask].reset_index(drop=True)
        y_train = y_train.loc[valid_mask].reset_index(drop=True)

        X_val = val_df[available].fillna(0)
        y_val = val_df[TARGET_COLUMN].copy()
        valid_mask_val = y_val.notna()
        X_val = X_val.loc[valid_mask_val].reset_index(drop=True)
        y_val = y_val.loc[valid_mask_val].reset_index(drop=True)

        # Load base models
        oof_matrix_train = []
        val_pred_matrix = []
        model_names: list[str] = []

        for model_file in sorted(model_dir.glob("*_best.pkl")):
            if "ensemble" in model_file.stem:
                continue  # Skip self

            name = model_file.stem.replace("_best", "")
            raw_model = joblib.load(model_file)

            is_enet = isinstance(raw_model, dict) and "scaler" in raw_model
            self._is_elastic_net[name] = is_enet

            if is_enet:
                inner_model = raw_model["model"]
                self._scalers[name] = raw_model["scaler"]
                self.base_models[name] = inner_model
            else:
                self.base_models[name] = raw_model

            logger.info(f"Generating OOF predictions for {name}")
            oof = _generate_oof_predictions(
                self.base_models[name], X_train, y_train,
                n_splits=n_splits, is_elastic_net=is_enet,
            )
            oof_matrix_train.append(oof)

            # Validation predictions
            if is_enet:
                X_val_scaled = self._scalers[name].transform(X_val)
                val_pred = self.base_models[name].predict(X_val_scaled)
            else:
                val_pred = self.base_models[name].predict(X_val)
            val_pred_matrix.append(val_pred)

            model_names.append(name)

        if not model_names:
            raise RuntimeError(f"No base models found in {model_dir}")

        # Stack OOF predictions as meta-features
        meta_X_train = np.column_stack(oof_matrix_train)
        meta_X_val = np.column_stack(val_pred_matrix)

        # Train meta-learner
        self.meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_learner.fit(meta_X_train, y_train.values)

        logger.info(f"Meta-learner trained. Alpha={self.meta_learner.alpha_:.4f}")
        logger.info(f"Meta-learner weights: {dict(zip(model_names, self.meta_learner.coef_))}")

        # Evaluate on validation
        val_preds = self.meta_learner.predict(meta_X_val)
        val_preds = np.clip(val_preds, 0, None)
        metrics = compute_all_regression_metrics(y_val.values, val_preds)
        logger.info(f"Ensemble validation metrics: {metrics}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted wait times.
        """
        if self.meta_learner is None:
            raise RuntimeError("Ensemble has not been fitted yet")

        # Align features
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names].fillna(0)

        # Get base model predictions
        base_preds: list[np.ndarray] = []
        for name, model in self.base_models.items():
            if self._is_elastic_net.get(name, False):
                X_scaled = self._scalers[name].transform(X)
                base_preds.append(model.predict(X_scaled))
            else:
                base_preds.append(model.predict(X))

        meta_X = np.column_stack(base_preds)
        preds = self.meta_learner.predict(meta_X)
        return np.clip(preds, 0, None)

    def predict_single(self, features: dict[str, Any]) -> float:
        """Predict for a single patient.

        Args:
            features: Dict of feature values.

        Returns:
            Predicted wait time in minutes.
        """
        X = pd.DataFrame([features])
        return float(self.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ensemble training."""
    parser = argparse.ArgumentParser(description="Train stacking ensemble")
    parser.add_argument("--data", type=Path, default=Path("data/processed"),
                        help="Directory with train/val parquet files")
    parser.add_argument("--models", type=Path, default=Path("models/wait_time"),
                        help="Directory with base model .pkl files")
    parser.add_argument("--output", type=Path, default=Path("models/wait_time/ensemble_best.pkl"),
                        help="Output path for ensemble model")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.data / "train.parquet")
    val_df = pd.read_parquet(args.data / "val.parquet")
    test_path = args.data / "test.parquet"
    test_df = pd.read_parquet(test_path) if test_path.exists() else None

    ensemble = StackingEnsemble()
    metrics = ensemble.fit(train_df, val_df, args.models)

    from src.models.wait_time.trainer import compute_evaluation_extras, copy_feature_encoders
    extras = compute_evaluation_extras(ensemble, val_df, test_df)
    copy_feature_encoders(args.data, args.output.parent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, args.output)
    logger.info(f"Saved ensemble to {args.output}")

    # Save metadata
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "stacking_ensemble",
            "base_models": list(ensemble.base_models.keys()),
            "meta_learner_alpha": float(ensemble.meta_learner.alpha_) if ensemble.meta_learner else None,
            "meta_learner_weights": {
                name: float(w) for name, w in zip(ensemble.base_models.keys(), ensemble.meta_learner.coef_)
            } if ensemble.meta_learner else {},
            "metrics": metrics,
            **extras,
        }, f, indent=2)

    logger.info(f"Ensemble MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    # Ensure pickle stores the fully-qualified class path instead of __main__.
    # Both the sys.modules alias and __module__ patch are needed so pickle's
    # identity check (same object at the resolved path) passes.
    import sys
    sys.modules["src.models.wait_time.ensemble"] = sys.modules[__name__]
    StackingEnsemble.__module__ = "src.models.wait_time.ensemble"
    main()
