"""No-show model evaluation — AUC-ROC, AUC-PR, confusion matrix, threshold analysis.

Owner: P2 (ML Engineer)
Consumers: P5 (Model Performance dashboard page)
Reference: MASTER_PROMPT §7.2 Step 3, §12.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.noshow.predictor import NoShowPredictor
from src.models.noshow.trainer import NOSHOW_FEATURES, _get_noshow_feature_target
from src.utils.constants import (
    TARGET_NOSHOW_AUC_PR,
    TARGET_NOSHOW_AUC_ROC,
    TARGET_NOSHOW_F1,
)
from src.utils.logger import get_logger
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import apply_radqueue_theme, save_figure

logger = get_logger(__name__)


class NoShowEvaluator:
    """Evaluation suite for the no-show prediction model."""

    def __init__(self, predictor: NoShowPredictor) -> None:
        self.predictor = predictor

    def evaluate_on_test_set(self, test_df: pd.DataFrame) -> dict[str, Any]:
        """Run full evaluation on the test set.

        Args:
            test_df: Test DataFrame.

        Returns:
            Dict with overall metrics, target comparison, and threshold analysis.
        """
        X_test, y_test, feature_names = _get_noshow_feature_target(test_df)

        y_proba = self.predictor.predict_batch(X_test)
        metrics = compute_classification_metrics(y_test.values, y_proba)

        # Target comparison
        target_comparison = {
            "auc_roc": {
                "actual": metrics["auc_roc"],
                "target": TARGET_NOSHOW_AUC_ROC,
                "passed": metrics["auc_roc"] >= TARGET_NOSHOW_AUC_ROC,
            },
            "auc_pr": {
                "actual": metrics["auc_pr"],
                "target": TARGET_NOSHOW_AUC_PR,
                "passed": metrics["auc_pr"] >= TARGET_NOSHOW_AUC_PR,
            },
            "f1": {
                "actual": metrics["f1"],
                "target": TARGET_NOSHOW_F1,
                "passed": metrics["f1"] >= TARGET_NOSHOW_F1,
            },
        }

        # Threshold sweep
        threshold_analysis: list[dict[str, float]] = []
        for thresh in np.arange(0.1, 0.9, 0.05):
            t_metrics = compute_classification_metrics(y_test.values, y_proba, threshold=float(thresh))
            threshold_analysis.append({
                "threshold": round(float(thresh), 2),
                "f1": t_metrics["f1"],
                "precision": t_metrics["precision"],
                "recall": t_metrics["recall"],
            })

        results = {
            "test_size": len(y_test),
            "noshow_rate": float(y_test.mean()),
            "metrics": metrics,
            "target_comparison": target_comparison,
            "threshold_analysis": threshold_analysis,
        }

        logger.info(f"No-show evaluation: AUC-ROC={metrics['auc_roc']:.4f}, "
                     f"AUC-PR={metrics['auc_pr']:.4f}, F1={metrics['f1']:.4f}")
        return results

    def generate_evaluation_plots(
        self,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> list[Path]:
        """Generate ROC, PR, and confusion matrix plots.

        Args:
            test_df: Test DataFrame.
            output_dir: Directory to save plots.

        Returns:
            List of saved file paths.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.metrics import precision_recall_curve, roc_curve

        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        X_test, y_test, _ = _get_noshow_feature_target(test_df)
        y_proba = self.predictor.predict_batch(X_test)

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics = compute_classification_metrics(y_test.values, y_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr.tolist(), y=tpr.tolist(),
            mode="lines",
            line={"color": "#6366F1", "width": 2},
            name=f"ROC (AUC={metrics['auc_roc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line={"dash": "dash", "color": "#94A3B8"},
            name="Random",
        ))
        apply_radqueue_theme(fig,
                             title="ROC Curve — No-Show Prediction",
                             xaxis_title="False Positive Rate",
                             yaxis_title="True Positive Rate")
        path = output_dir / "roc_curve.html"
        save_figure(fig, path)
        saved.append(path)

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall.tolist(), y=precision.tolist(),
            mode="lines",
            line={"color": "#10B981", "width": 2},
            name=f"PR (AUC={metrics['auc_pr']:.3f})",
        ))
        apply_radqueue_theme(fig,
                             title="Precision-Recall Curve — No-Show Prediction",
                             xaxis_title="Recall",
                             yaxis_title="Precision")
        path = output_dir / "pr_curve.html"
        save_figure(fig, path)
        saved.append(path)

        # 3. Confusion Matrix Heatmap
        cm = np.array(metrics["confusion_matrix"])
        fig = go.Figure(data=go.Heatmap(
            z=cm.tolist(),
            x=["Showed Up", "No-Show"],
            y=["Showed Up", "No-Show"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            colorscale=[[0, "#1E293B"], [1, "#6366F1"]],
            showscale=False,
        ))
        apply_radqueue_theme(fig,
                             title="Confusion Matrix — No-Show Model",
                             xaxis_title="Predicted",
                             yaxis_title="Actual")
        path = output_dir / "confusion_matrix.html"
        save_figure(fig, path)
        saved.append(path)

        logger.info(f"Generated {len(saved)} no-show evaluation plots")
        return saved
