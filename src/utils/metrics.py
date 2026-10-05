"""Custom evaluation metric functions for RadQueue AI.

Owner: P1 (Data & Config Lead)
Consumers: P2 (model evaluation), P5 (dashboard metrics display)
Reference: MASTER_PROMPT §12.1, §12.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Regression Metrics (Wait-Time Prediction — §12.1)
# ---------------------------------------------------------------------------


def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error in minutes.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        MAE value.
    """
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error in minutes.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_r_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of Determination (R²).

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        R² score (1.0 = perfect, 0.0 = mean-only model).
    """
    return float(r2_score(y_true, y_pred))


def compute_mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error (%).

    Filters out zero ground-truth values to avoid division by zero.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5%).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    mask = y_true_arr > 0
    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def compute_median_ae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median Absolute Error in minutes.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        Median AE value.
    """
    return float(median_absolute_error(y_true, y_pred))


def compute_p90_ae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """90th Percentile Absolute Error — measures tail performance.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        P90 absolute error in minutes.
    """
    abs_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.percentile(abs_errors, 90))


def compute_all_regression_metrics(
    y_true: ArrayLike, y_pred: ArrayLike
) -> dict[str, float]:
    """Compute the full regression metric suite per §12.1.

    Args:
        y_true: Ground-truth wait times.
        y_pred: Predicted wait times.

    Returns:
        Dictionary with keys: mae, rmse, r_squared, mape, median_ae, p90_ae.
    """
    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "r_squared": compute_r_squared(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "median_ae": compute_median_ae(y_true, y_pred),
        "p90_ae": compute_p90_ae(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Classification Metrics (No-Show Prediction — §12.2)
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute classification metrics for the no-show model.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred_proba: Predicted probabilities.
        threshold: Classification threshold (default 0.5).

    Returns:
        Dictionary with auc_roc, auc_pr, f1, precision, recall,
        confusion_matrix, and optimal_threshold.
    """
    from sklearn.metrics import (
        auc,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_pred_proba, dtype=np.float64)
    y_pred_binary = (y_proba_arr >= threshold).astype(int)

    # Precision-recall curve for AUC-PR
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_arr, y_proba_arr)
    auc_pr = float(auc(recall_curve, precision_curve))

    # Find optimal threshold (max F1)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    optimal_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(pr_thresholds[min(optimal_idx, len(pr_thresholds) - 1)])

    return {
        "auc_roc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        "auc_pr": auc_pr,
        "f1": float(f1_score(y_true_arr, y_pred_binary)),
        "precision": float(precision_score(y_true_arr, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_binary, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_binary).tolist(),
        "optimal_threshold": optimal_threshold,
        "threshold_used": threshold,
    }


# ---------------------------------------------------------------------------
# Scheduling Performance Metrics (§12.3)
# ---------------------------------------------------------------------------


def compute_scheduling_metrics(
    wait_times: ArrayLike,
    utilization_rates: ArrayLike,
    emergency_response_times: ArrayLike | None = None,
    max_routine_wait: float = 90.0,
) -> dict[str, float]:
    """Compute scheduling optimization KPIs.

    Args:
        wait_times: Array of patient wait times in minutes.
        utilization_rates: Array of resource utilization rates (0-1).
        emergency_response_times: Optional emergency response times.
        max_routine_wait: Threshold for starvation detection (default 90 min).

    Returns:
        Dictionary of scheduling performance metrics.
    """
    wt = np.asarray(wait_times, dtype=np.float64)
    ur = np.asarray(utilization_rates, dtype=np.float64)

    metrics: dict[str, float] = {
        "avg_wait_time": float(np.mean(wt)),
        "max_wait_time": float(np.max(wt)),
        "median_wait_time": float(np.median(wt)),
        "std_wait_time": float(np.std(wt)),
        "p90_wait_time": float(np.percentile(wt, 90)),
        "avg_utilization": float(np.mean(ur)),
        "starvation_count": int(np.sum(wt > max_routine_wait)),
        "total_patients": len(wt),
    }

    if emergency_response_times is not None:
        ert = np.asarray(emergency_response_times, dtype=np.float64)
        metrics["avg_emergency_response"] = float(np.mean(ert))
        metrics["max_emergency_response"] = float(np.max(ert))
        metrics["emergency_compliance_rate"] = float(np.mean(ert <= 10.0))

    return metrics
