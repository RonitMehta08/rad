"""Rolling model performance tracking and degradation alerting.

Owner: P2 (ML Engineer)
Consumers: P5 (Model Performance dashboard page)
Reference: MASTER_PROMPT §5.3 Module 4
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.constants import TARGET_MAE_MINUTES
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_regression_metrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEGRADATION_THRESHOLD_PERCENT = 20.0  # Alert if MAE increases >20% over window
ROLLING_WINDOW_SIZE = 500  # Number of recent predictions to track
SHORT_WINDOW_SIZE = 100


# ---------------------------------------------------------------------------
# Performance Record
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSnapshot:
    """A single snapshot of model performance metrics."""

    timestamp: str
    window_size: int
    mae: float
    rmse: float
    r_squared: float
    mape: float
    per_modality_mae: dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationAlert:
    """Alert when model performance degrades beyond threshold."""

    timestamp: str
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: str  # "warning" or "critical"
    message: str = ""


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------


class PerformanceTracker:
    """Tracks model prediction accuracy over time with rolling windows.

    Maintains a buffer of recent (prediction, actual) pairs and computes
    rolling metrics. Triggers degradation alerts when accuracy drops.

    Attributes:
        model_name: Name of the model being tracked.
        baseline_mae: Baseline MAE from initial evaluation.
    """

    def __init__(
        self,
        model_name: str = "wait_time_model",
        baseline_mae: float = TARGET_MAE_MINUTES,
        window_size: int = ROLLING_WINDOW_SIZE,
    ) -> None:
        self.model_name = model_name
        self.baseline_mae = baseline_mae
        self.window_size = window_size

        # Rolling buffers
        self._predictions: deque[float] = deque(maxlen=window_size)
        self._actuals: deque[float] = deque(maxlen=window_size)
        self._modalities: deque[str] = deque(maxlen=window_size)
        self._timestamps: deque[str] = deque(maxlen=window_size)

        # History of snapshots
        self.snapshots: list[PerformanceSnapshot] = []
        self.alerts: list[DegradationAlert] = []

    def record_prediction(
        self,
        predicted: float,
        actual: float,
        modality: str = "unknown",
        timestamp: str | None = None,
    ) -> None:
        """Record a single prediction-actual pair.

        Args:
            predicted: Model's predicted wait time.
            actual: Actual observed wait time.
            modality: Imaging modality for this prediction.
            timestamp: ISO timestamp (auto-generated if None).
        """
        ts = timestamp or datetime.now(tz=timezone.utc).isoformat()
        self._predictions.append(predicted)
        self._actuals.append(actual)
        self._modalities.append(modality)
        self._timestamps.append(ts)

    def record_batch(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        modalities: list[str] | None = None,
    ) -> None:
        """Record a batch of prediction-actual pairs.

        Args:
            predictions: Array of predictions.
            actuals: Array of actual values.
            modalities: Optional list of modalities per prediction.
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        mods = modalities or ["unknown"] * len(predictions)

        for pred, act, mod in zip(predictions, actuals, mods):
            self.record_prediction(float(pred), float(act), mod, ts)

    def compute_current_metrics(self) -> PerformanceSnapshot | None:
        """Compute metrics on the current rolling window.

        Returns:
            PerformanceSnapshot or None if insufficient data.
        """
        if len(self._predictions) < 10:
            return None

        preds = np.array(self._predictions)
        acts = np.array(self._actuals)
        mods = list(self._modalities)

        metrics = compute_all_regression_metrics(acts, preds)

        # Per-modality breakdown
        per_modality: dict[str, float] = {}
        mod_series = pd.Series(mods)
        for mod in mod_series.unique():
            mask = mod_series == mod
            if mask.sum() >= 5:
                mod_preds = preds[mask.values]
                mod_acts = acts[mask.values]
                per_modality[str(mod)] = float(np.mean(np.abs(mod_acts - mod_preds)))

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            window_size=len(preds),
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            r_squared=metrics["r_squared"],
            mape=metrics["mape"],
            per_modality_mae=per_modality,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def check_for_degradation(self) -> list[DegradationAlert]:
        """Check if model performance has degraded beyond threshold.

        Compares current rolling MAE against the baseline and recent history.

        Returns:
            List of new degradation alerts (empty if healthy).
        """
        snapshot = self.compute_current_metrics()
        if snapshot is None:
            return []

        new_alerts: list[DegradationAlert] = []

        # Check MAE against baseline
        if self.baseline_mae > 0:
            change_pct = ((snapshot.mae - self.baseline_mae) / self.baseline_mae) * 100

            if change_pct > DEGRADATION_THRESHOLD_PERCENT:
                severity = "critical" if change_pct > DEGRADATION_THRESHOLD_PERCENT * 2 else "warning"
                alert = DegradationAlert(
                    timestamp=snapshot.timestamp,
                    metric_name="mae",
                    current_value=snapshot.mae,
                    baseline_value=self.baseline_mae,
                    change_percent=change_pct,
                    severity=severity,
                    message=(
                        f"MAE degraded by {change_pct:.1f}% "
                        f"(baseline: {self.baseline_mae:.2f}, current: {snapshot.mae:.2f}). "
                        + ("Retraining recommended." if severity == "critical"
                           else "Monitor closely.")
                    ),
                )
                new_alerts.append(alert)
                self.alerts.append(alert)
                logger.warning(alert.message)

        # Check short-term trend (recent 100 vs. earlier)
        if len(self._predictions) >= SHORT_WINDOW_SIZE * 2:
            preds = np.array(self._predictions)
            acts = np.array(self._actuals)

            recent_mae = float(np.mean(np.abs(acts[-SHORT_WINDOW_SIZE:] - preds[-SHORT_WINDOW_SIZE:])))
            earlier_mae = float(np.mean(np.abs(
                acts[-SHORT_WINDOW_SIZE * 2:-SHORT_WINDOW_SIZE] -
                preds[-SHORT_WINDOW_SIZE * 2:-SHORT_WINDOW_SIZE]
            )))

            if earlier_mae > 0:
                trend_change = ((recent_mae - earlier_mae) / earlier_mae) * 100
                if trend_change > 15:
                    alert = DegradationAlert(
                        timestamp=snapshot.timestamp,
                        metric_name="mae_trend",
                        current_value=recent_mae,
                        baseline_value=earlier_mae,
                        change_percent=trend_change,
                        severity="warning",
                        message=(
                            f"Short-term MAE trending upward: "
                            f"{earlier_mae:.2f} → {recent_mae:.2f} ({trend_change:+.1f}%)"
                        ),
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)

        return new_alerts

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of current model performance status.

        Returns:
            Dict with current metrics, trend info, and alerts.
        """
        current = self.compute_current_metrics()

        summary: dict[str, Any] = {
            "model_name": self.model_name,
            "total_predictions_tracked": len(self._predictions),
            "baseline_mae": self.baseline_mae,
            "current_metrics": None,
            "recent_alerts": [],
            "status": "healthy",
        }

        if current:
            summary["current_metrics"] = {
                "mae": current.mae,
                "rmse": current.rmse,
                "r_squared": current.r_squared,
                "mape": current.mape,
                "per_modality_mae": current.per_modality_mae,
            }

        # Recent alerts (last 10)
        summary["recent_alerts"] = [
            {"timestamp": a.timestamp, "metric": a.metric_name,
             "severity": a.severity, "message": a.message}
            for a in self.alerts[-10:]
        ]

        # Overall status
        critical_alerts = [a for a in self.alerts[-5:] if a.severity == "critical"]
        warning_alerts = [a for a in self.alerts[-5:] if a.severity == "warning"]
        if critical_alerts:
            summary["status"] = "critical"
        elif warning_alerts:
            summary["status"] = "warning"

        return summary
