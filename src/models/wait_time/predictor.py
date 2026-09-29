"""Wait-time inference pipeline — load model, prepare features, predict, explain.

Owner: P2 (ML Engineer)
Consumers: P3 (scheduler integration), P5 (dashboard prediction page)
Reference: MASTER_PROMPT §5.3 Module 1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.wait_time.ensemble import StackingEnsemble  # noqa: F401 — needed for joblib.load
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Predictor Class
# ---------------------------------------------------------------------------


class WaitTimePredictor:
    """Inference pipeline for wait-time prediction.

    Loads a trained model, prepares input features, and returns
    predictions with optional SHAP explanations.

    Attributes:
        model: The loaded ML model.
        feature_names: Ordered feature column names the model expects.
        model_name: Human-readable model identifier.
    """

    def __init__(
        self,
        model_path: Path | str,
        feature_names_path: Path | str | None = None,
    ) -> None:
        """Initialize the predictor by loading model artifacts.

        Args:
            model_path: Path to the .pkl model file.
            feature_names_path: Path to feature_names.json. If None,
                infers from model_path's parent directory.

        Raises:
            FileNotFoundError: If model or feature_names file is missing.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_name = model_path.stem
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model '{self.model_name}' from {model_path}")

        # Handle Elastic Net package (dict with 'scaler' and 'model')
        self._is_elastic_net = isinstance(self.model, dict) and "scaler" in self.model
        if self._is_elastic_net:
            self._scaler = self.model["scaler"]
            self._model_inner = self.model["model"]
        else:
            self._scaler = None
            self._model_inner = self.model

        # Load feature names
        if feature_names_path is None:
            feature_names_path = model_path.parent / "feature_names.json"
        feature_names_path = Path(feature_names_path)

        if feature_names_path.exists():
            with open(feature_names_path, encoding="utf-8") as f:
                self.feature_names: list[str] = json.load(f)
        else:
            logger.warning(f"feature_names.json not found at {feature_names_path}, using defaults")
            self.feature_names = FEATURE_COLUMNS

    def _prepare_features(self, patient_features: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Align input features to the model's expected columns.

        Args:
            patient_features: Either a dict of features or a single-row DataFrame.

        Returns:
            Single-row DataFrame with columns matching self.feature_names.
        """
        if isinstance(patient_features, dict):
            df = pd.DataFrame([patient_features])
        else:
            df = patient_features.copy()

        # Add missing columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Select only expected features in order
        df = df[self.feature_names]

        # Fill NaN
        df = df.fillna(0)

        return df

    def predict(self, patient_features: dict[str, Any] | pd.DataFrame) -> float:
        """Predict wait time in minutes for a single patient.

        Args:
            patient_features: Dictionary or DataFrame of engineered features.

        Returns:
            Predicted wait time in minutes (non-negative).
        """
        X = self._prepare_features(patient_features)

        if self._is_elastic_net:
            X_scaled = self._scaler.transform(X)
            pred = float(self._model_inner.predict(X_scaled)[0])
        else:
            pred = float(self._model_inner.predict(X)[0])

        # Clamp to non-negative
        return max(0.0, round(pred, 1))

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict wait times for a batch of patients.

        Args:
            features_df: DataFrame with one row per patient.

        Returns:
            Array of predicted wait times.
        """
        X = self._prepare_features(features_df)

        if self._is_elastic_net:
            X_scaled = self._scaler.transform(X)
            preds = self._model_inner.predict(X_scaled)
        else:
            preds = self._model_inner.predict(X)

        return np.clip(preds, 0, None)

    def predict_with_confidence(
        self,
        patient_features: dict[str, Any] | pd.DataFrame,
        confidence_level: float = 0.90,
    ) -> dict[str, float]:
        """Predict wait time with a pseudo-confidence interval.

        Uses the model's prediction ± a percentage-based margin as a
        simple confidence interval proxy (tree models don't natively
        produce prediction intervals).

        Args:
            patient_features: Input features.
            confidence_level: Desired confidence level (0.0-1.0).

        Returns:
            Dict with keys: predicted, lower_bound, upper_bound.
        """
        pred = self.predict(patient_features)

        # Heuristic: ±20% for 90% CI, scale proportionally
        margin_fraction = 0.20 * (confidence_level / 0.90)
        margin = pred * margin_fraction

        return {
            "predicted": pred,
            "lower_bound": max(0.0, round(pred - margin, 1)),
            "upper_bound": round(pred + margin, 1),
            "confidence_level": confidence_level,
        }
