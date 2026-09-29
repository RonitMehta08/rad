"""No-show probability inference pipeline.

Owner: P2 (ML Engineer)
Consumers: P3 (scheduler overbooking), P5 (dashboard)
Reference: MASTER_PROMPT §7.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.noshow.trainer import NOSHOW_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NoShowPredictor:
    """Inference pipeline for no-show probability prediction.

    Uses the calibrated classifier for well-calibrated probability estimates.

    Attributes:
        model: Raw XGBoost classifier.
        calibrator: Platt-calibrated classifier.
        feature_names: Expected feature columns.
    """

    def __init__(
        self,
        model_path: Path | str,
        calibrator_path: Path | str | None = None,
    ) -> None:
        """Load the no-show model and calibrator.

        Args:
            model_path: Path to xgboost_noshow.pkl.
            calibrator_path: Path to calibrator.pkl. If None, inferred.

        Raises:
            FileNotFoundError: If model files are missing.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"No-show model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded no-show model from {model_path}")

        # Load calibrator
        if calibrator_path is None:
            calibrator_path = model_path.parent / "calibrator.pkl"
        calibrator_path = Path(calibrator_path)

        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
            self._use_calibrator = True
            logger.info(f"Loaded Platt calibrator from {calibrator_path}")
        else:
            self.calibrator = None
            self._use_calibrator = False
            logger.warning("No calibrator found — using raw model probabilities")

        self.feature_names = NOSHOW_FEATURES

    def _prepare_features(self, features: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """Align input features to expected columns.

        Args:
            features: Input features as dict or DataFrame.

        Returns:
            Aligned single-row DataFrame.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        available = [c for c in self.feature_names if c in df.columns]
        return df[available].fillna(0)

    def predict_probability(self, features: dict[str, Any] | pd.DataFrame) -> float:
        """Predict no-show probability for a single patient.

        Args:
            features: Patient features.

        Returns:
            Calibrated no-show probability (0.0 to 1.0).
        """
        X = self._prepare_features(features)

        if self._use_calibrator:
            proba = self.calibrator.predict_proba(X)[0, 1]
        else:
            proba = self.model.predict_proba(X)[0, 1]

        return float(np.clip(proba, 0.0, 1.0))

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict no-show probabilities for a batch of patients.

        Args:
            features_df: DataFrame with one row per patient.

        Returns:
            Array of calibrated no-show probabilities.
        """
        X = self._prepare_features(features_df)

        if self._use_calibrator:
            probas = self.calibrator.predict_proba(X)[:, 1]
        else:
            probas = self.model.predict_proba(X)[:, 1]

        return np.clip(probas, 0.0, 1.0)

    def should_overbook(
        self,
        features: dict[str, Any] | pd.DataFrame,
        threshold: float = 0.5,
        max_overbooking_fraction: float = 0.15,
        current_overbooking_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Decide whether to overbook a slot based on no-show risk.

        Args:
            features: Patient features.
            threshold: Probability threshold above which overbooking is triggered.
            max_overbooking_fraction: Safety limit on total overbooking.
            current_overbooking_rate: Current overbooking level.

        Returns:
            Dict with keys: no_show_probability, should_overbook,
            overbooking_blocked_by_safety, reason.
        """
        prob = self.predict_probability(features)
        over_threshold = prob >= threshold
        within_safety = current_overbooking_rate < max_overbooking_fraction

        return {
            "no_show_probability": round(prob, 3),
            "should_overbook": over_threshold and within_safety,
            "overbooking_blocked_by_safety": over_threshold and not within_safety,
            "reason": (
                f"No-show probability {prob:.1%} "
                + ("exceeds" if over_threshold else "below")
                + f" threshold ({threshold:.0%})"
                + (f"; overbooking rate {current_overbooking_rate:.0%} at safety limit"
                   if over_threshold and not within_safety else "")
            ),
        }
