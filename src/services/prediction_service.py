"""Wait-time and no-show prediction service (prediction + SHAP + clinician text).

Owner: P5 (Dashboard & API)
Reference: MASTER_PROMPT §5.3 Module 1-2, USP 7
"""

from __future__ import annotations

from typing import Any

from src.services.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_SHAP_FEATURES: int = 6


def predict_wait_time(registry: ModelRegistry, raw_patient: dict[str, Any], explain: bool = True) -> dict[str, Any]:
    """Predict a patient's wait with an empirical interval and SHAP explanation.

    Args:
        registry: Model registry.
        raw_patient: Raw patient fields + optional live context (see feature_builder).
        explain: Whether to compute the SHAP explanation.

    Returns:
        Dict with predicted_wait_minutes, lower/upper bound, interval_method,
        model_name and (optionally) explanation {contributions, clinician_text}.

    Raises:
        ModelNotAvailableError: If no trained wait-time model exists.
    """
    predictor = registry.wait_time_predictor
    result = predictor.predict_with_confidence(raw_patient)
    response: dict[str, Any] = {
        "predicted_wait_minutes": result["predicted"],
        "lower_bound_minutes": result["lower_bound"],
        "upper_bound_minutes": result["upper_bound"],
        "confidence_level": result["confidence_level"],
        "interval_method": result["interval_method"],
        "model_name": predictor.model_name,
    }
    if explain:
        explainer = registry.explainer
        features = predictor.prepare_features(raw_patient)
        explanation = explainer.explain_single_prediction(
            features[explainer.feature_names] if explainer.feature_names else features,
            result["predicted"],
            top_n=TOP_SHAP_FEATURES,
        )
        explanation["explained_model"] = getattr(explainer, "model_name", "tree_model")
        response["explanation"] = explanation
    return response


def predict_noshow(registry: ModelRegistry, raw_patient: dict[str, Any]) -> dict[str, Any]:
    """No-show probability and overbooking recommendation for a scheduled appointment.

    Raises:
        ModelNotAvailableError: If the no-show model is not trained.
    """
    predictor = registry.noshow_predictor
    decision = predictor.should_overbook(raw_patient)
    decision["decision_threshold"] = round(predictor.decision_threshold, 3)
    return decision
