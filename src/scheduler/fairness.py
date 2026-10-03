"""Rawlsian fairness constraints, starvation prevention, and equity scoring.

Inspired by the FAIR framework (2025):
- Evaluates equity across waiting patients so that low-priority patients do not starve.
- Implements Rawlsian maximin criteria (optimizing for the worst-off patient).
- Enforces strict waiting-time ceilings per clinical urgency level.

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §4, §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from src.scheduler.models import PatientState
from src.utils.constants import (
    UrgencyLevel,
    FAIRNESS_PENALTY_RATE,
    MAX_WAIT_ROUTINE_MINUTES,
    MAX_WAIT_URGENT_MINUTES,
    MAX_WAIT_EMERGENCY_MINUTES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

WAIT_LIMITS: dict[UrgencyLevel, float] = {
    UrgencyLevel.ROUTINE: float(MAX_WAIT_ROUTINE_MINUTES),
    UrgencyLevel.URGENT: float(MAX_WAIT_URGENT_MINUTES),
    UrgencyLevel.EMERGENCY: float(MAX_WAIT_EMERGENCY_MINUTES),
}


class FairnessEngine:
    """Calculates fairness penalties and enforces starvation constraints."""

    def __init__(
        self,
        penalty_rate: float = FAIRNESS_PENALTY_RATE,
        starvation_threshold_minutes: float = 60.0,
        wait_limits: dict[UrgencyLevel, float] | None = None,
    ) -> None:
        self.penalty_rate = penalty_rate
        self.starvation_threshold_minutes = starvation_threshold_minutes
        self.wait_limits = wait_limits or WAIT_LIMITS

    def compute_fairness_bonus(
        self, current_wait_minutes: float, avg_wait_minutes: float
    ) -> float:
        """Calculate dynamic fairness bonus for a patient waiting longer than average.

        Bonus = max(0, current_wait - avg_wait) * penalty_rate
        """
        excess_wait = max(0.0, current_wait_minutes - avg_wait_minutes)
        return excess_wait * self.penalty_rate

    def is_starving(self, patient: PatientState) -> bool:
        """Check if a routine or urgent patient is at risk of starvation."""
        if patient.urgency == UrgencyLevel.EMERGENCY:
            return False
        return patient.current_wait_minutes >= self.starvation_threshold_minutes

    def check_wait_limit_violation(
        self, patient: PatientState, total_wait_minutes: float
    ) -> bool:
        """Check whether total wait would breach urgency-specific maximum allowable wait."""
        limit = self.wait_limits.get(patient.urgency, 90.0)
        return total_wait_minutes > limit

    def get_starving_patients(
        self, queue: Sequence[PatientState]
    ) -> list[PatientState]:
        """Filter queue for patients exceeding starvation threshold."""
        return [p for p in queue if self.is_starving(p)]

    @staticmethod
    def compute_equity_metrics(waiting_times: Sequence[float]) -> dict[str, float]:
        """Compute equity and dispersion metrics for waiting times.

        Metrics:
        - avg_wait: Mean wait time
        - max_wait: Worst-off patient wait (Rawlsian metric)
        - std_dev: Standard deviation
        - gini_index: Gini inequality coefficient (0 = perfect equity, 1 = maximum inequity)
        """
        if not waiting_times:
            return {"avg_wait": 0.0, "max_wait": 0.0, "std_dev": 0.0, "gini": 0.0}

        arr = np.array(waiting_times, dtype=np.float64)
        n = len(arr)
        avg = float(np.mean(arr))
        max_val = float(np.max(arr))
        std = float(np.std(arr))

        # Efficient O(n log n) Gini calculation with O(n) memory
        if avg == 0 or n <= 1:
            gini = 0.0
        else:
            sorted_arr = np.sort(arr)
            total_sum = np.sum(sorted_arr)
            if total_sum == 0:
                gini = 0.0
            else:
                indices = np.arange(1, n + 1, dtype=np.float64)
                gini = float((2.0 * np.sum(indices * sorted_arr) / (n * total_sum)) - (n + 1.0) / n)
                gini = max(0.0, min(1.0, gini))

        return {
            "avg_wait": round(avg, 2),
            "max_wait": round(max_val, 2),
            "std_dev": round(std, 2),
            "gini": round(gini, 4),
        }
