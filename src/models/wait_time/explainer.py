"""SHAP explanations — global feature importance and per-sample waterfall plots.

Generates both technical SHAP visualizations and clinician-friendly text
explanations for each prediction.

Owner: P2 (ML Engineer)
Consumers: P5 (Wait Time Predictor dashboard page, SHAP display component)
Reference: MASTER_PROMPT §7.1 Step 5, USP 7, §16 Cmd 8

Usage:
    python -m src.models.wait_time.explainer \\
        --model models/wait_time/lightgbm_best.pkl \\
        --data data/processed/test.parquet --output reports/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Human-Readable Feature Name Mapping
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "current_queue_length_total": "Total queue length",
    "current_queue_length_same_modality": "Queue length (same modality)",
    "patients_in_service_count": "Patients currently being served",
    "avg_service_time_last_5_patients": "Avg service time (last 5 patients)",
    "time_since_last_patient_served_minutes": "Time since last patient served",
    "emergency_patients_in_queue": "Emergency patients in queue",
    "num_machines_available": "Available machines",
    "num_technologists_on_duty": "Technologists on duty",
    "num_radiologists_on_duty": "Radiologists on duty",
    "modality_encoded": "Imaging modality",
    "visit_type_encoded": "Visit type",
    "urgency_encoded": "Clinical urgency",
    "exam_complexity_encoded": "Exam complexity",
    "hour_of_day_sin": "Time of day (cyclic)",
    "hour_of_day_cos": "Time of day (cyclic)",
    "day_of_week_sin": "Day of week (cyclic)",
    "day_of_week_cos": "Day of week (cyclic)",
    "is_holiday": "Holiday",
    "is_monday": "Monday (peak day)",
    "is_weekend": "Weekend",
    "rolling_avg_wait_1hr": "Avg wait time (last 1 hour)",
    "rolling_avg_wait_30min": "Avg wait time (last 30 min)",
    "arrival_rate_last_15min": "Recent arrival rate",
    "utilization_rate": "Resource utilization",
    "congestion_indicator": "Congestion level",
    "requires_contrast": "Contrast injection required",
    "requires_prep": "Patient preparation required",
    "appointment_lead_time_days": "Appointment lead time (days)",
    "previous_no_show_count": "Previous no-shows",
    "equipment_under_maintenance": "Equipment under maintenance",
    "minutes_since_department_opened": "Time since dept opened",
    "estimated_remaining_service_minutes": "Est. remaining service time",
    "wait_time_last_served_patient": "Last patient's wait time",
    "wait_time_last_3_avg": "Avg wait (last 3 patients)",
    "queue_length_change_last_15min": "Queue change (last 15 min)",
    "visit_queue_interaction": "Visit type × queue interaction",
}


# ---------------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------------


class WaitTimeExplainer:
    """SHAP-based explanation engine for wait-time predictions.

    Generates global importance plots, per-sample waterfall explanations,
    and clinician-friendly text summaries.
    """

    def __init__(self, model: Any, feature_names: list[str] | None = None) -> None:
        """Initialize the explainer.

        Args:
            model: Trained tree model (LightGBM or XGBoost).
            feature_names: Ordered feature names. If None, inferred.
        """
        self.model = model
        self.feature_names = feature_names or []

        # Handle Elastic Net package
        if isinstance(model, dict) and "model" in model:
            self._inner_model = model["model"]
        else:
            self._inner_model = model

        self.explainer: shap.TreeExplainer | None = None

    def _ensure_explainer(self, X: pd.DataFrame | np.ndarray) -> None:
        """Lazily initialize the SHAP TreeExplainer."""
        if self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self._inner_model)
            except Exception:
                logger.warning("TreeExplainer failed, falling back to KernelExplainer (slow)")
                background = shap.sample(X, min(100, len(X)))
                self.explainer = shap.KernelExplainer(self._inner_model.predict, background)

    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values for a feature matrix.

        Args:
            X: Feature DataFrame.

        Returns:
            SHAP Explanation object.
        """
        self._ensure_explainer(X)
        shap_values = self.explainer(X)
        return shap_values

    def get_global_importance(
        self, X: pd.DataFrame, top_n: int = 15,
    ) -> list[dict[str, Any]]:
        """Compute global feature importance via mean |SHAP|.

        Args:
            X: Feature DataFrame (typically test or validation set).
            top_n: Number of top features to return.

        Returns:
            List of dicts with keys: feature, display_name, mean_abs_shap.
        """
        shap_vals = self.compute_shap_values(X)
        mean_abs = np.abs(shap_vals.values).mean(axis=0)

        feature_names = self.feature_names if self.feature_names else list(X.columns)
        importance = sorted(
            zip(feature_names, mean_abs),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "mean_abs_shap": round(float(val), 4),
            }
            for feat, val in importance
        ]

    def explain_single_prediction(
        self,
        X_single: pd.DataFrame | dict[str, Any],
        predicted_wait: float,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Generate a full explanation for a single prediction.

        Includes SHAP feature contributions and a clinician-friendly text.

        Args:
            X_single: Single-row feature DataFrame or dict.
            predicted_wait: The model's predicted wait time.
            top_n: Number of top contributing features to highlight.

        Returns:
            Dict with keys: predicted_wait, base_value, contributions,
            clinician_text.
        """
        if isinstance(X_single, dict):
            X_single = pd.DataFrame([X_single])

        feature_names = self.feature_names if self.feature_names else list(X_single.columns)

        # Align columns
        for col in feature_names:
            if col not in X_single.columns:
                X_single[col] = 0
        X_single = X_single[feature_names].fillna(0)

        self._ensure_explainer(X_single)
        shap_vals = self.explainer(X_single)

        base_value = float(shap_vals.base_values[0])
        shap_values_arr = shap_vals.values[0]

        # Build sorted contributions
        contributions = sorted(
            zip(feature_names, shap_values_arr, X_single.iloc[0].values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        contribution_list = [
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "feature_value": round(float(val), 2) if isinstance(val, (int, float, np.floating)) else str(val),
                "shap_value": round(float(sv), 2),
                "direction": "increases" if sv > 0 else "decreases",
            }
            for feat, sv, val in contributions[:top_n]
        ]

        # Generate clinician-friendly text
        clinician_text = _generate_clinician_text(predicted_wait, contribution_list)

        return {
            "predicted_wait_minutes": predicted_wait,
            "base_value": round(base_value, 2),
            "contributions": contribution_list,
            "clinician_text": clinician_text,
        }

    def save_global_plots(
        self,
        X: pd.DataFrame,
        output_dir: Path,
        max_display: int = 15,
    ) -> list[Path]:
        """Generate and save global SHAP plots.

        Args:
            X: Feature matrix (test or val set).
            output_dir: Directory to save plots.
            max_display: Max features to display.

        Returns:
            List of saved file paths.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        shap_vals = self.compute_shap_values(X)

        # 1. Summary bar plot
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_vals, max_display=max_display, show=False)
        plt.tight_layout()
        path = output_dir / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)
        logger.info(f"Saved SHAP summary bar plot to {path}")

        # 2. Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_vals, max_display=max_display, show=False)
        plt.tight_layout()
        path = output_dir / "shap_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)
        logger.info(f"Saved SHAP beeswarm plot to {path}")

        # 3. Feature importance bar chart
        importance = self.get_global_importance(X, top_n=max_display)
        from src.utils.plotting import create_bar_comparison, save_figure
        fig = create_bar_comparison(
            labels=[i["display_name"] for i in reversed(importance)],
            values=[i["mean_abs_shap"] for i in reversed(importance)],
            title="Feature Importance (Mean |SHAP|)",
            x_label="Mean |SHAP Value|",
        )
        path = output_dir / "feature_importance.html"
        save_figure(fig, path)
        saved.append(path)

        return saved


# ---------------------------------------------------------------------------
# Clinician-Friendly Text Generator (USP 7)
# ---------------------------------------------------------------------------


def _generate_clinician_text(
    predicted_wait: float,
    contributions: list[dict[str, Any]],
) -> str:
    """Translate SHAP contributions into clinician-friendly language.

    Args:
        predicted_wait: Predicted wait time in minutes.
        contributions: Top SHAP contributions (from explain_single_prediction).

    Returns:
        Plain-English explanation string.

    Example output:
        "This patient's predicted wait is 47 minutes. The main contributors are:
        Current MRI queue length (12 patients) adding +18 min, Walk-in arrival
        during peak hour adding +14 min, offset by 2 MRI machines available
        saving -8 min."
    """
    lines: list[str] = [
        f"This patient's predicted wait is **{predicted_wait:.0f} minutes**."
    ]

    increasing = [c for c in contributions if c["shap_value"] > 0]
    decreasing = [c for c in contributions if c["shap_value"] < 0]

    if increasing:
        parts = []
        for c in increasing[:3]:
            parts.append(
                f"**{c['display_name']}** (value: {c['feature_value']}) "
                f"adding +{abs(c['shap_value']):.0f} min"
            )
        lines.append("The main factors increasing wait time: " + "; ".join(parts) + ".")

    if decreasing:
        parts = []
        for c in decreasing[:2]:
            parts.append(
                f"**{c['display_name']}** (value: {c['feature_value']}) "
                f"saving {c['shap_value']:.0f} min"
            )
        lines.append("Offsetting factors: " + "; ".join(parts) + ".")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for SHAP explanation generation."""
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--model", type=Path, required=True, help="Path to model .pkl")
    parser.add_argument("--data", type=Path, required=True, help="Path to test parquet")
    parser.add_argument("--output", type=Path, default=Path("reports/figures"),
                        help="Output directory for plots")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples for SHAP computation")
    args = parser.parse_args()

    # Load model
    model = joblib.load(args.model)
    test_df = pd.read_parquet(args.data)

    # Prepare features
    available = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    valid_mask = test_df[TARGET_COLUMN].notna()
    X_test = test_df.loc[valid_mask, available].fillna(0)

    # Subsample for speed
    if len(X_test) > args.max_samples:
        X_test = X_test.sample(args.max_samples, random_state=42)

    explainer = WaitTimeExplainer(model, feature_names=available)
    saved = explainer.save_global_plots(X_test, args.output)

    # Save global importance as JSON
    importance = explainer.get_global_importance(X_test)
    imp_path = args.output / "feature_importance.json"
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2)

    logger.info(f"SHAP explanations complete. Saved {len(saved) + 1} files to {args.output}")


if __name__ == "__main__":
    main()
