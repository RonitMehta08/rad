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

from src.data.feature_builder import build_feature_frame
from src.data.preprocessor import ENCODERS_FILENAME, FEATURE_COLUMNS
from src.models.wait_time.ensemble import StackingEnsemble  # noqa: F401 — needed for joblib.load
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fallback relative margin when no residual quantiles were saved with the model.
FALLBACK_INTERVAL_FRACTION: float = 0.20
DEFAULT_CONFIDENCE_LEVEL: float = 0.90


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

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

        # Encoders let us accept raw patient descriptions (see src.data.feature_builder)
        self.encoders = load_json_if_exists(model_path.parent / ENCODERS_FILENAME)
        if self.encoders is None:
            logger.warning(
                f"{ENCODERS_FILENAME} not found next to {model_path}; raw inputs will use "
                "untrained defaults. Re-run the preprocessor + trainer to create it."
            )

        # Empirical validation residual quantiles -> honest prediction intervals
        metadata = (
            load_json_if_exists(model_path.parent / f"{self.model_name}_metadata.json")
            or load_json_if_exists(model_path.with_suffix(".json"))
            or {}
        )
        self.metadata = metadata
        self.residual_quantiles: dict[str, float] = metadata.get("residual_quantiles", {})

    def _prepare_features(self, patient_features: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Align input features to the model's expected columns.

        Args:
            patient_features: Either a dict of features or a single-row DataFrame.

        Returns:
            Single-row DataFrame with columns matching self.feature_names.
        """
        if isinstance(patient_features, dict):
            is_engineered = all(col in patient_features for col in self.feature_names)
            if not is_engineered:
                # Raw patient description (e.g. from API, dashboard or dispatcher)
                return build_feature_frame(patient_features, self.encoders)[self.feature_names]
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

    def prepare_features(self, patient_features: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Public alias: the exact model-ready feature row used for prediction."""
        return self._prepare_features(patient_features)

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
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> dict[str, float]:
        """Predict wait time with an empirical prediction interval.

        The interval adds the 5th/95th percentiles of validation residuals
        (actual - predicted) saved by the trainer. If those are missing it falls
        back to a +/-20% heuristic and says so via ``interval_method``.

        Args:
            patient_features: Input features (raw or engineered).
            confidence_level: Nominal coverage; only 0.90 is calibrated.

        Returns:
            Dict with keys: predicted, lower_bound, upper_bound, confidence_level,
            interval_method.
        """
        pred = self.predict(patient_features)

        if "q05" in self.residual_quantiles and "q95" in self.residual_quantiles:
            lower = pred + self.residual_quantiles["q05"]
            upper = pred + self.residual_quantiles["q95"]
            method = "validation_residual_quantiles"
        else:
            margin = pred * FALLBACK_INTERVAL_FRACTION * (confidence_level / DEFAULT_CONFIDENCE_LEVEL)
            lower, upper = pred - margin, pred + margin
            method = "heuristic_pct"

        return {
            "predicted": pred,
            "lower_bound": max(0.0, round(lower, 1)),
            "upper_bound": round(max(upper, pred), 1),
            "confidence_level": confidence_level,
            "interval_method": method,
        }
