"""Level 2: Real-time dynamic priority dispatch engine.

Implements:
1. Dynamic priority scoring = base_urgency + fairness_bonus + modality_factor + starvation_boost.
2. Multi-objective resource matching minimizing:
   alpha * avg_wait + beta * max_wait + gamma * wait_std_dev + delta * idle_penalty.
3. Queue re-evaluation upon resource availability events.

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.scheduler.config import SchedulerConfig, load_scheduler_config
from src.scheduler.fairness import FairnessEngine
from src.scheduler.models import Assignment, DepartmentState, PatientState, Resource
from src.utils.constants import (
    SERVICE_TIME_PARAMS,
    URGENCY_WEIGHTS,
    ModalityType,
    UrgencyLevel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MINUTES_PER_HOUR = 60


def build_scheduler_context(
    patient: PatientState, resource: Resource, state: DepartmentState
) -> dict[str, Any]:
    """Translate scheduler state into raw wait-time model inputs.

    Keys match the raw columns understood by ``src.data.feature_builder``.
    """
    same_modality = [p for p in state.waiting_patients if p.modality == patient.modality]
    machines = [r for r in state.resources if r.modality == patient.modality] or [resource]
    busy = sum(1 for r in machines if not r.is_available)
    return {
        "modality": patient.modality.value,
        "urgency": patient.urgency.value,
        "visit_type": patient.visit_type.value,
        "current_queue_length_same_modality": state.queue_length.get(patient.modality, len(same_modality)),
        "current_queue_length_total": sum(state.queue_length.values()) or len(state.waiting_patients),
        "emergency_patients_in_queue": sum(
            1 for p in state.waiting_patients if p.urgency == UrgencyLevel.EMERGENCY
        ),
        "patients_in_service_count": busy,
        "num_machines_available": len(machines),
        "minutes_since_department_opened": state.current_time_minutes,
        "hour_of_day": 8 + int(state.current_time_minutes // MINUTES_PER_HOUR),
    }



class RealTimeDispatcher:
    """Real-time dynamic priority dispatch optimizer for radiology resources."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        wait_time_predictor: Any | None = None,
    ) -> None:
        self.config = config or load_scheduler_config()
        self.wait_time_predictor = wait_time_predictor
        self.fairness_engine = FairnessEngine(
            penalty_rate=self.config.fairness_penalty_rate,
            starvation_threshold_minutes=self.config.starvation_threshold_minutes,
        )

        # Multi-objective weights aligned with config.milp.objective_weights (§7.3)
        weights = self.config.milp.objective_weights
        self.alpha = weights.avg_wait_time       # 0.40 (efficiency / avg wait)
        self.beta = weights.max_wait_time        # 0.25 (fairness / max wait ceiling)
        self.gamma = weights.wait_time_std_dev   # 0.15 (equity / dispersion variance)
        self.delta = weights.utilization         # 0.20 (resource efficiency)

    def compute_priority_score(
        self, patient: PatientState, state: DepartmentState
    ) -> float:
        """Compute dynamic priority score for a waiting patient.

        Score = base_urgency + fairness_bonus + modality_factor + starvation_boost
        Emergencies are additionally placed in a strict top tier by ``rank_queue``,
        so this score only orders patients within the emergency / non-emergency tiers.
        """
        base = float(URGENCY_WEIGHTS.get(patient.urgency, 1))

        # Fairness bonus: increases as patient waits longer than current department average
        wait_bonus = self.fairness_engine.compute_fairness_bonus(
            patient.current_wait_minutes, state.avg_wait_minutes
        )

        # Modality factor: slightly boosts patients in less congested modalities
        queue_len = state.queue_length.get(patient.modality, 0)
        modality_factor = 1.0 / (1.0 + float(queue_len))

        # Starvation kicker: boosts starving routine patients ahead of urgent, while preserving emergency primacy
        starvation_boost = 3.0 if self.fairness_engine.is_starving(patient) else 0.0

        return base + wait_bonus + modality_factor + starvation_boost

    def rank_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState,
        modality: ModalityType | None = None,
    ) -> list[tuple[PatientState, float]]:
        """Rank waiting patients by dynamic priority score in descending order."""
        candidates = [p for p in queue if modality is None or p.modality == modality]
        scored = [
            (p, self.compute_priority_score(p, state))
            for p in candidates
        ]
        # Emergencies always rank first (uncapped fairness bonuses must never let a
        # long-waiting routine/urgent patient overtake them); then by score, then arrival.
        return sorted(
            scored,
            key=lambda item: (
                0 if item[0].urgency == UrgencyLevel.EMERGENCY else 1,
                -item[1],
                item[0].arrival_time_minutes,
            ),
        )

    def predict_wait_for_resource(
        self, patient: PatientState, resource: Resource, state: DepartmentState
    ) -> float:
        """Predict expected wait time if patient is routed to given resource.

        When a ``WaitTimePredictor`` is attached, the live department state is
        translated into the model's real feature names (see
        ``build_scheduler_context``); otherwise an analytical estimate is used.
        """
        if self.wait_time_predictor is not None:
            try:
                raw = dict(patient.features)
                raw.update(build_scheduler_context(patient, resource, state))
                return float(self.wait_time_predictor.predict(raw))
            except Exception as e:  # predictor problems must be visible, not silent
                logger.warning(
                    f"Wait-time predictor failed for patient_id={patient.patient_id}, "
                    f"modality={patient.modality.value}; using analytical fallback: {e}"
                )

        # Analytical fallback: time until resource is free + patient's current wait
        time_to_free = max(0.0, resource.busy_until_minute - state.current_time_minutes)
        return round(patient.current_wait_minutes + time_to_free, 1)

    def estimate_cascade_delay(
        self, resource: Resource, patient: PatientState, state: DepartmentState
    ) -> float:
        """Estimate downstream delay imposed on other waiting patients in same modality."""
        downstream_count = state.queue_length.get(patient.modality, 1)
        est_duration = (
            patient.estimated_duration_minutes
            or SERVICE_TIME_PARAMS[patient.modality]["median_minutes"]
        )
        return float(est_duration * max(0, downstream_count - 1))

    def select_best_resource(
        self,
        patient: PatientState,
        available_resources: Sequence[Resource],
        state: DepartmentState,
    ) -> tuple[Resource | None, float, float]:
        """Select the optimal available resource for the patient using multi-objective scoring.

        Returns:
            Tuple of (chosen_resource, predicted_wait, score)
        """
        compatible = [
            r for r in available_resources
            if r.modality == patient.modality and (r.is_available or r.busy_until_minute <= state.current_time_minutes)
        ]

        if not compatible:
            return None, float("inf"), float("inf")

        best_resource: Resource | None = None
        best_score = float("inf")
        best_pred_wait = 0.0

        for res in compatible:
            pred_wait = self.predict_wait_for_resource(patient, res, state)
            cascade_proxy = self.estimate_cascade_delay(res, patient, state)
            max_wait_impact = max(pred_wait, state.avg_wait_minutes)
            idle_penalty = 1.0 - res.utilization_rate

            # Multi-objective score: alpha*avg_wait + beta*max_wait + gamma*std_proxy + delta*idle
            score = (
                self.alpha * pred_wait
                + self.beta * max_wait_impact
                + self.gamma * cascade_proxy
                + self.delta * idle_penalty
            )

            if score < best_score:
                best_score = score
                best_resource = res
                best_pred_wait = pred_wait

        return best_resource, best_pred_wait, best_score

    def dispatch_next(
        self,
        queue: Sequence[PatientState],
        resources: Sequence[Resource],
        state: DepartmentState,
        modality: ModalityType | None = None,
    ) -> Assignment | None:
        """Dynamically evaluate queue and dispatch highest priority patient to best resource."""
        ranked = self.rank_queue(queue, state, modality)
        if not ranked:
            return None

        for patient, _priority_score in ranked:
            best_res, pred_wait, obj_score = self.select_best_resource(
                patient, resources, state
            )
            if best_res is not None:
                duration = (
                    patient.estimated_duration_minutes
                    or SERVICE_TIME_PARAMS[patient.modality]["median_minutes"]
                )
                return Assignment(
                    patient_id=patient.patient_id,
                    resource_id=best_res.resource_id,
                    start_time_minutes=state.current_time_minutes,
                    estimated_duration_minutes=duration,
                    predicted_wait_minutes=pred_wait,
                    urgency=patient.urgency,
                    score=obj_score,
                )

        return None
