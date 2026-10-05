"""Wait-time model evaluation — metrics, per-modality breakdown, residual analysis.

Owner: P2 (ML Engineer)
Consumers: P5 (Model Performance dashboard page)
Reference: MASTER_PROMPT §7.1 Step 4, §12.1, §16 Cmd 10

Usage:
    python -m src.models.wait_time.evaluator --models models/ \\
        --data data/processed/test.parquet --output reports/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.wait_time.predictor import WaitTimePredictor
from src.utils.constants import (
    TARGET_MAE_MINUTES,
    TARGET_MAPE_PERCENT,
    TARGET_R_SQUARED,
    TARGET_RMSE_MINUTES,
)
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_regression_metrics
from src.utils.plotting import (
    MODALITY_COLORS,
    apply_radqueue_theme,
    create_scatter_pred_vs_actual,
    save_figure,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------


class WaitTimeEvaluator:
    """Full evaluation suite for wait-time prediction models.

    Computes overall metrics, per-modality breakdowns, residual analysis,
    and generates evaluation plots.
    """

    def __init__(self, predictor: WaitTimePredictor) -> None:
        self.predictor = predictor

    def evaluate_on_test_set(
        self, test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run full evaluation on the held-out test set.

        Args:
            test_df: Test DataFrame with features and target.

        Returns:
            Dict with keys: overall_metrics, per_modality_metrics,
            per_visit_type_metrics, residual_stats, target_comparison.
        """
        available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]
        X_test = test_df[available_features].fillna(0)
        y_test = test_df[TARGET_COLUMN].dropna()

        # Align X to valid y
        valid_idx = y_test.index
        X_test = X_test.loc[valid_idx]

        y_pred = self.predictor.predict_batch(X_test)
        y_true = y_test.values

        # Overall metrics
        overall = compute_all_regression_metrics(y_true, y_pred)

        # Residuals
        residuals = y_true - y_pred
        residual_stats = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "skewness": float(pd.Series(residuals).skew()),
            "kurtosis": float(pd.Series(residuals).kurtosis()),
            "pct_overestimate": float(np.mean(residuals < 0) * 100),
            "pct_underestimate": float(np.mean(residuals > 0) * 100),
        }

        # Per-modality breakdown
        per_modality: dict[str, dict[str, float]] = {}
        if "modality" in test_df.columns:
            for mod in test_df.loc[valid_idx, "modality"].unique():
                mod_mask = test_df.loc[valid_idx, "modality"] == mod
                if mod_mask.sum() > 10:
                    per_modality[str(mod)] = compute_all_regression_metrics(
                        y_true[mod_mask.values], y_pred[mod_mask.values]
                    )

        # Per-visit-type breakdown
        per_visit: dict[str, dict[str, float]] = {}
        visit_col = "visit_type" if "visit_type" in test_df.columns else None
        if visit_col:
            for vt in test_df.loc[valid_idx, visit_col].unique():
                vt_mask = test_df.loc[valid_idx, visit_col] == vt
                if vt_mask.sum() > 10:
                    per_visit[str(vt)] = compute_all_regression_metrics(
                        y_true[vt_mask.values], y_pred[vt_mask.values]
                    )

        # Target comparison
        target_comparison = {
            "mae_vs_target": {
                "actual": overall["mae"],
                "target": TARGET_MAE_MINUTES,
                "passed": overall["mae"] <= TARGET_MAE_MINUTES,
            },
            "rmse_vs_target": {
                "actual": overall["rmse"],
                "target": TARGET_RMSE_MINUTES,
                "passed": overall["rmse"] <= TARGET_RMSE_MINUTES,
            },
            "r_squared_vs_target": {
                "actual": overall["r_squared"],
                "target": TARGET_R_SQUARED,
                "passed": overall["r_squared"] >= TARGET_R_SQUARED,
            },
            "mape_vs_target": {
                "actual": overall["mape"],
                "target": TARGET_MAPE_PERCENT,
                "passed": overall["mape"] <= TARGET_MAPE_PERCENT,
            },
        }

        results = {
            "model_name": self.predictor.model_name,
            "test_size": len(y_true),
            "overall_metrics": overall,
            "per_modality_metrics": per_modality,
            "per_visit_type_metrics": per_visit,
            "residual_stats": residual_stats,
            "target_comparison": target_comparison,
        }

        logger.info(f"Evaluation complete for {self.predictor.model_name}: MAE={overall['mae']:.3f}")
        return results

    def generate_evaluation_plots(
        self,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> list[Path]:
        """Generate all evaluation plots and save them.

        Args:
            test_df: Test DataFrame.
            output_dir: Directory to save plots.

        Returns:
            List of paths to generated plot files.
        """
        import plotly.graph_objects as go

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]
        X_test = test_df[available_features].fillna(0)
        y_test = test_df[TARGET_COLUMN].dropna()
        valid_idx = y_test.index
        X_test = X_test.loc[valid_idx]

        y_pred = self.predictor.predict_batch(X_test)
        y_true = y_test.values

        # 1. Prediction vs Actual scatter
        fig = create_scatter_pred_vs_actual(
            y_true.tolist(), y_pred.tolist(),
            title=f"Predicted vs Actual — {self.predictor.model_name}",
        )
        path = output_dir / "prediction_vs_actual.html"
        save_figure(fig, path)
        saved_paths.append(path)

        # 2. Residual plot
        residuals = y_true - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred.tolist(), y=residuals.tolist(),
            mode="markers",
            marker={"color": "#6366F1", "size": 4, "opacity": 0.5},
            name="Residuals",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        apply_radqueue_theme(fig,
                             title=f"Residual Plot — {self.predictor.model_name}",
                             xaxis_title="Predicted Wait Time (min)",
                             yaxis_title="Residual (Actual - Predicted)")
        path = output_dir / "residual_plot.html"
        save_figure(fig, path)
        saved_paths.append(path)

        # 3. Error distribution histogram
        abs_errors = np.abs(residuals)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=abs_errors.tolist(), nbinsx=50,
            marker_color="#10B981", opacity=0.8,
            name="Absolute Error",
        ))
        apply_radqueue_theme(fig,
                             title="Absolute Error Distribution",
                             xaxis_title="Absolute Error (min)",
                             yaxis_title="Count")
        path = output_dir / "error_distribution.html"
        save_figure(fig, path)
        saved_paths.append(path)

        # 4. Per-modality violin plot
        if "modality" in test_df.columns:
            modalities = test_df.loc[valid_idx, "modality"].values
            fig = go.Figure()
            for mod in sorted(set(modalities)):
                mask = modalities == mod
                if mask.sum() > 5:
                    fig.add_trace(go.Violin(
                        y=np.abs(residuals[mask]).tolist(),
                        name=str(mod).upper(),
                        marker_color=MODALITY_COLORS.get(str(mod), "#6366F1"),
                        box_visible=True,
                        meanline_visible=True,
                    ))
            apply_radqueue_theme(fig,
                                 title="Absolute Error by Modality",
                                 yaxis_title="Absolute Error (min)")
            path = output_dir / "modality_violin.html"
            save_figure(fig, path)
            saved_paths.append(path)

        logger.info(f"Generated {len(saved_paths)} evaluation plots in {output_dir}")
        return saved_paths


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate wait-time models on test set")
    parser.add_argument("--models", type=Path, default=Path("models"),
                        help="Root models directory")
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.parquet"),
                        help="Test set parquet file")
    parser.add_argument("--output", type=Path, default=Path("reports"),
                        help="Output directory for reports and figures")
    args = parser.parse_args()

    test_df = pd.read_parquet(args.data)

    # Evaluate each available model
    model_dir = args.models / "wait_time"
    all_results: list[dict[str, Any]] = []

    for model_file in model_dir.glob("*_best.pkl"):
        predictor = WaitTimePredictor(model_file)
        evaluator = WaitTimeEvaluator(predictor)

        results = evaluator.evaluate_on_test_set(test_df)
        all_results.append(results)

        evaluator.generate_evaluation_plots(test_df, args.output / "figures")

    # Save combined report
    report_path = args.output / "evaluation_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Evaluation report saved to {report_path}")

    # Print summary
    for r in all_results:
        m = r["overall_metrics"]
        logger.info(f"{r['model_name']}: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, "
                     f"R²={m['r_squared']:.3f}, MAPE={m['mape']:.1f}%")


if __name__ == "__main__":
    main()
