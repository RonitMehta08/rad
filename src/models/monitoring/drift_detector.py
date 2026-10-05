"""Data and concept drift detection — PSI, KS-test, Chi-squared, ADWIN, CUSUM.

Owner: P2 (ML Engineer)
Consumers: P5 (Model Performance dashboard page — drift monitoring section)
Reference: MASTER_PROMPT §5.3 Module 4
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Drift Result Types
# ---------------------------------------------------------------------------

PSI_THRESHOLD_LOW = 0.10  # No significant drift
PSI_THRESHOLD_HIGH = 0.20  # Significant drift — retrain recommended


@dataclass
class DriftResult:
    """Result of a single drift test on one feature."""

    feature_name: str
    test_name: str
    statistic: float
    p_value: float | None
    is_drifted: bool
    severity: str  # "none", "warning", "critical"
    details: str = ""


@dataclass
class DriftReport:
    """Aggregated drift report across all features."""

    timestamp: str = ""
    total_features_tested: int = 0
    features_drifted: int = 0
    drift_results: list[DriftResult] = field(default_factory=list)
    overall_status: str = "healthy"  # "healthy", "warning", "critical"
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures the shift in distribution of a feature between the
    reference (training) period and the current (production) period.

    Args:
        reference: Feature values from training/reference period.
        current: Feature values from current/production period.
        n_bins: Number of bins for discretization.

    Returns:
        PSI value. <0.1 = stable, 0.1-0.2 = slight shift, >0.2 = significant drift.
    """
    # Create bins from reference distribution
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    # Normalize to proportions
    ref_pct = ref_counts / len(ref_clean)
    cur_pct = cur_counts / len(cur_clean)

    # Avoid log(0) — add small epsilon
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# KS Test
# ---------------------------------------------------------------------------


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    significance: float = 0.05,
) -> DriftResult:
    """Kolmogorov-Smirnov test for distribution shift on a numerical feature.

    Args:
        reference: Reference distribution values.
        current: Current distribution values.
        significance: Significance level for p-value threshold.

    Returns:
        DriftResult with test outcome.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return DriftResult(
            feature_name="", test_name="ks_test",
            statistic=0.0, p_value=1.0,
            is_drifted=False, severity="none",
            details="Insufficient data for KS test",
        )

    ks_stat, p_value = stats.ks_2samp(ref_clean, cur_clean)

    is_drifted = p_value < significance
    severity = "critical" if p_value < 0.01 else ("warning" if is_drifted else "none")

    return DriftResult(
        feature_name="",
        test_name="ks_test",
        statistic=float(ks_stat),
        p_value=float(p_value),
        is_drifted=is_drifted,
        severity=severity,
        details=f"KS statistic={ks_stat:.4f}, p-value={p_value:.6f}",
    )


# ---------------------------------------------------------------------------
# Chi-Squared Test (Categorical Features)
# ---------------------------------------------------------------------------


def compute_chi_squared(
    reference: np.ndarray,
    current: np.ndarray,
    significance: float = 0.05,
) -> DriftResult:
    """Chi-squared test for distribution shift on a categorical feature.

    Args:
        reference: Reference category values.
        current: Current category values.
        significance: Significance level.

    Returns:
        DriftResult with test outcome.
    """
    ref_series = pd.Series(reference).value_counts()
    cur_series = pd.Series(current).value_counts()

    # Align categories
    all_cats = sorted(set(ref_series.index) | set(cur_series.index))
    ref_counts = np.array([ref_series.get(c, 0) for c in all_cats], dtype=float)
    cur_counts = np.array([cur_series.get(c, 0) for c in all_cats], dtype=float)

    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return DriftResult(
            feature_name="", test_name="chi_squared",
            statistic=0.0, p_value=1.0,
            is_drifted=False, severity="none",
            details="Insufficient data for chi-squared test",
        )

    # Scale reference to match current total
    expected = ref_counts * (cur_counts.sum() / ref_counts.sum())
    expected = np.clip(expected, 1e-6, None)

    chi2_stat, p_value = stats.chisquare(cur_counts, f_exp=expected)

    is_drifted = p_value < significance
    severity = "critical" if p_value < 0.01 else ("warning" if is_drifted else "none")

    return DriftResult(
        feature_name="",
        test_name="chi_squared",
        statistic=float(chi2_stat),
        p_value=float(p_value),
        is_drifted=is_drifted,
        severity=severity,
        details=f"Chi² statistic={chi2_stat:.4f}, p-value={p_value:.6f}",
    )


# ---------------------------------------------------------------------------
# ADWIN (Adaptive Windowing) — Simplified
# ---------------------------------------------------------------------------


def detect_adwin_drift(
    values: np.ndarray,
    delta: float = 0.002,
) -> DriftResult:
    """Simplified ADWIN concept drift detection.

    Compares the mean of the first half vs. the second half of a stream.
    For a full ADWIN implementation, use river.drift.ADWIN.

    Args:
        values: Stream of prediction errors or metric values.
        delta: Significance threshold.

    Returns:
        DriftResult indicating whether concept drift was detected.
    """
    if len(values) < 20:
        return DriftResult(
            feature_name="predictions", test_name="adwin",
            statistic=0.0, p_value=None,
            is_drifted=False, severity="none",
            details="Insufficient data for ADWIN",
        )

    mid = len(values) // 2
    first_half = values[:mid]
    second_half = values[mid:]

    mean_diff = abs(np.mean(second_half) - np.mean(first_half))
    pooled_std = np.sqrt(np.var(first_half) / len(first_half) + np.var(second_half) / len(second_half))

    if pooled_std < 1e-10:
        is_drifted = False
    else:
        z_score = mean_diff / pooled_std
        is_drifted = z_score > stats.norm.ppf(1 - delta)

    severity = "critical" if is_drifted else "none"

    return DriftResult(
        feature_name="predictions",
        test_name="adwin",
        statistic=float(mean_diff),
        p_value=None,
        is_drifted=is_drifted,
        severity=severity,
        details=f"Mean shift={mean_diff:.4f}, pooled_std={pooled_std:.4f}",
    )


# ---------------------------------------------------------------------------
# CUSUM (Cumulative Sum)
# ---------------------------------------------------------------------------


def detect_cusum_drift(
    values: np.ndarray,
    threshold: float = 5.0,
    drift_magnitude: float = 1.0,
) -> DriftResult:
    """CUSUM change-point detection for gradual drift.

    Args:
        values: Stream of values (e.g., prediction errors).
        threshold: CUSUM decision threshold (h).
        drift_magnitude: Expected minimal drift magnitude (k).

    Returns:
        DriftResult indicating whether a change point was detected.
    """
    if len(values) < 10:
        return DriftResult(
            feature_name="predictions", test_name="cusum",
            statistic=0.0, p_value=None,
            is_drifted=False, severity="none",
            details="Insufficient data for CUSUM",
        )

    mean_val = np.mean(values[:len(values) // 2])  # reference mean
    half_k = drift_magnitude / 2

    cusum_pos = 0.0
    cusum_neg = 0.0
    max_cusum = 0.0

    for v in values:
        cusum_pos = max(0, cusum_pos + (v - mean_val) - half_k)
        cusum_neg = max(0, cusum_neg - (v - mean_val) - half_k)
        max_cusum = max(max_cusum, cusum_pos, cusum_neg)

    is_drifted = max_cusum > threshold
    severity = "critical" if max_cusum > threshold * 2 else ("warning" if is_drifted else "none")

    return DriftResult(
        feature_name="predictions",
        test_name="cusum",
        statistic=float(max_cusum),
        p_value=None,
        is_drifted=is_drifted,
        severity=severity,
        details=f"Max CUSUM={max_cusum:.4f}, threshold={threshold}",
    )


# ---------------------------------------------------------------------------
# Full Drift Detection Pipeline
# ---------------------------------------------------------------------------


class DriftDetector:
    """Comprehensive drift detection across all features and predictions.

    Runs PSI + KS-test on numerical features, chi-squared on categoricals,
    and ADWIN + CUSUM on prediction streams.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Initialize with reference (training) data.

        Args:
            reference_data: Training data for baseline distributions.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
        """
        self.reference_data = reference_data
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []

        # Auto-detect if not provided
        if not self.numerical_features and not self.categorical_features:
            for col in reference_data.columns:
                if reference_data[col].dtype in ("float64", "float32", "int64", "int32"):
                    self.numerical_features.append(col)
                elif reference_data[col].dtype == "object":
                    self.categorical_features.append(col)

    def run_full_drift_check(
        self,
        current_data: pd.DataFrame,
        prediction_errors: np.ndarray | None = None,
    ) -> DriftReport:
        """Run all drift detection tests.

        Args:
            current_data: Current/production data.
            prediction_errors: Optional array of recent prediction errors.

        Returns:
            Comprehensive DriftReport.
        """
        from datetime import datetime, timezone

        results: list[DriftResult] = []

        # Numerical features: PSI + KS
        for feat in self.numerical_features:
            if feat not in self.reference_data.columns or feat not in current_data.columns:
                continue

            ref_vals = self.reference_data[feat].values.astype(float)
            cur_vals = current_data[feat].values.astype(float)

            # PSI
            psi_val = compute_psi(ref_vals, cur_vals)
            psi_severity = (
                "critical" if psi_val > PSI_THRESHOLD_HIGH
                else ("warning" if psi_val > PSI_THRESHOLD_LOW else "none")
            )
            results.append(DriftResult(
                feature_name=feat, test_name="psi",
                statistic=psi_val, p_value=None,
                is_drifted=psi_val > PSI_THRESHOLD_HIGH,
                severity=psi_severity,
                details=f"PSI={psi_val:.4f}",
            ))

            # KS test
            ks_result = compute_ks_test(ref_vals, cur_vals)
            ks_result.feature_name = feat
            results.append(ks_result)

        # Categorical features: Chi-squared
        for feat in self.categorical_features:
            if feat not in self.reference_data.columns or feat not in current_data.columns:
                continue

            ref_vals = self.reference_data[feat].values
            cur_vals = current_data[feat].values

            chi_result = compute_chi_squared(ref_vals, cur_vals)
            chi_result.feature_name = feat
            results.append(chi_result)

        # Concept drift on predictions
        if prediction_errors is not None and len(prediction_errors) > 0:
            adwin_result = detect_adwin_drift(prediction_errors)
            results.append(adwin_result)

            cusum_result = detect_cusum_drift(prediction_errors)
            results.append(cusum_result)

        # Aggregate
        drifted_count = sum(1 for r in results if r.is_drifted)
        critical_count = sum(1 for r in results if r.severity == "critical")

        if critical_count > 0:
            overall = "critical"
            recommendation = "Significant drift detected. Model retraining recommended."
        elif drifted_count > len(results) * 0.3:
            overall = "warning"
            recommendation = "Multiple features showing drift. Monitor closely."
        else:
            overall = "healthy"
            recommendation = "No significant drift detected."

        report = DriftReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            total_features_tested=len(results),
            features_drifted=drifted_count,
            drift_results=results,
            overall_status=overall,
            recommendation=recommendation,
        )

        logger.info(f"Drift check: {drifted_count}/{len(results)} features drifted, "
                     f"status={overall}")
        return report
