"""Level 3: Real-time event rescheduling and trigger handling.

Handles operational events:
1. Emergency preemption & cascade rescheduling.
2. No-show detection (pull-forward & controlled overbooking).
3. Equipment failure queue redistribution.
4. Surge detection (arrival rate > 2 sigma) and overflow protocol.

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §4, §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

from typing import Any

from src.scheduler.config import SchedulerConfig, load_scheduler_config
from src.scheduler.dispatcher import RealTimeDispatcher
from src.scheduler.models import Assignment, DepartmentState, PatientState, Resource
from src.utils.constants import OVERBOOKING_PROBABILITY_THRESHOLD, SERVICE_TIME_PARAMS, UrgencyLevel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReschedulingEngine:
    """Manages dynamic event triggers and downstream re-routing."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        dispatcher: RealTimeDispatcher | None = None,
        noshow_predictor: Any | None = None,
    ) -> None:
        self.config = config or load_scheduler_config()
        self.resched_cfg = self.config.rescheduling
        self.dispatcher = dispatcher or RealTimeDispatcher(config=self.config)
        self.noshow_predictor = noshow_predictor

    def handle_emergency_preemption(
        self,
        emergency_patient: PatientState,
        active_assignments: list[Assignment],
        waiting_queue: list[PatientState],
        resources: list[Resource],
        state: DepartmentState,
        active_patients: list[PatientState] | None = None,
    ) -> tuple[Assignment | None, PatientState | None]:
        """Preempt routine prep to serve an emergency arrival immediately.

        Returns:
            Tuple of (emergency_assignment, preempted_patient)
        """
        if not self.resched_cfg.preemption_enabled:
            logger.info("Preemption disabled by configuration")
            return None, None

        compatible_res_ids = {
            r.resource_id for r in resources if r.modality == emergency_patient.modality
        }

        # Look for active routine assignment in prep on a compatible machine
        candidate_assignment: Assignment | None = None
        for asgn in active_assignments:
            if asgn.resource_id in compatible_res_ids:
                if self.resched_cfg.preempt_only_routine:
                    if asgn.urgency == UrgencyLevel.ROUTINE:
                        candidate_assignment = asgn
                        break
                else:
                    if asgn.urgency != UrgencyLevel.EMERGENCY:
                        candidate_assignment = asgn
                        break

        # Find patient record
        preempted_patient: PatientState | None = None
        target_res_id: str | None = None

        if candidate_assignment is not None:
            target_res_id = candidate_assignment.resource_id
            search_pool = (active_patients or []) + (state.active_patients or []) + list(waiting_queue)
            for p in search_pool:
                if p.patient_id == candidate_assignment.patient_id:
                    preempted_patient = p
                    break

            if preempted_patient is None:
                # Fallback: create patient state representation from assignment
                preempted_patient = PatientState(
                    patient_id=candidate_assignment.patient_id,
                    modality=emergency_patient.modality,
                    urgency=candidate_assignment.urgency,
                    arrival_time_minutes=candidate_assignment.start_time_minutes,
                    current_wait_minutes=0.0,
                )

            # Remove displaced assignment from active
            active_assignments.remove(candidate_assignment)
        else:
            # Check if any compatible resource is directly available
            avail = [r for r in resources if r.modality == emergency_patient.modality and r.is_available]
            if avail:
                target_res_id = avail[0].resource_id

        if target_res_id is None:
            logger.warning("No resource available or preemptable for emergency")
            return None, None

        # Create emergency assignment
        dur = (
            emergency_patient.estimated_duration_minutes
            or SERVICE_TIME_PARAMS[emergency_patient.modality]["median_minutes"]
        )
        emergency_asgn = Assignment(
            patient_id=emergency_patient.patient_id,
            resource_id=target_res_id,
            start_time_minutes=state.current_time_minutes,
            estimated_duration_minutes=dur,
            predicted_wait_minutes=0.0,
            urgency=UrgencyLevel.EMERGENCY,
            score=0.0,
        )
        active_assignments.append(emergency_asgn)

        # Cascade reschedule preempted patient: elevate priority bonus to prevent double-delay
        if preempted_patient:
            preempted_patient.status = "preempted"
            preempted_patient.current_wait_minutes += dur
            logger.info(
                f"Emergency {emergency_patient.patient_id} preempted {preempted_patient.patient_id}"
            )

        return emergency_asgn, preempted_patient

    def handle_noshow_pullforward(
        self,
        scheduled_patient: PatientState,
        waiting_queue: list[PatientState],
        available_resources: list[Resource],
        state: DepartmentState,
    ) -> Assignment | None:
        """Pull next queued patient forward if scheduled patient is flagged as no-show."""
        if not self.resched_cfg.noshow_pullforward_enabled:
            return None

        # Check grace window: patient has not arrived within window minutes
        delay = state.current_time_minutes - scheduled_patient.arrival_time_minutes
        if delay < self.resched_cfg.noshow_detection_window_minutes:
            return None  # Not yet classified as no-show

        scheduled_patient.status = "noshow"
        logger.info(
            f"Patient {scheduled_patient.patient_id} marked as no-show after {delay:.1f}m delay"
        )

        # Pull forward next patient in queue matching the modality
        candidates = [
            p for p in waiting_queue
            if p.modality == scheduled_patient.modality and p.patient_id != scheduled_patient.patient_id
        ]
        if not candidates:
            return None

        # Dispatch next best candidate
        return self.dispatcher.dispatch_next(
            candidates, available_resources, state, modality=scheduled_patient.modality
        )

    def evaluate_controlled_overbooking(
        self,
        slot_patients_count: int,
        slot_capacity: int = 1,
        features: dict[str, Any] | None = None,
        historical_noshow_prob: float | None = None,
    ) -> bool:
        """Determine if a slot should be overbooked based on predicted no-show probability.

        Args:
            slot_patients_count: Total patients already scheduled in this slot.
            slot_capacity: Baseline single-slot capacity (typically 1).
            features: Patient features for ML model.
            historical_noshow_prob: Default or historical probability fallback.

        Returns:
            True if overbooking is permitted and within safety limits.
        """
        prob = 0.0
        if self.noshow_predictor is not None and features is not None:
            try:
                if hasattr(self.noshow_predictor, "predict_probability"):
                    prob = float(self.noshow_predictor.predict_probability(features))
                elif hasattr(self.noshow_predictor, "predict_proba"):
                    prob = float(self.noshow_predictor.predict_proba(features))
                else:
                    prob = float(self.noshow_predictor.predict(features))
            except Exception as e:
                logger.debug(f"NoShowPredictor fallback: {e}")
                prob = historical_noshow_prob or 0.20
        else:
            prob = historical_noshow_prob or 0.20

        max_overbook_frac = self.config.constraints.max_overbooking_fraction
        current_overbook_rate = max(
            0.0, float(slot_patients_count - slot_capacity) / float(max(1, slot_capacity))
        )

        # Overbook only when no-show risk is high and the overbooking rate is within the safety limit
        return prob >= OVERBOOKING_PROBABILITY_THRESHOLD and current_overbook_rate < max_overbook_frac

    def handle_equipment_failure(
        self,
        failed_resource_id: str,
        resources: list[Resource],
        active_assignments: list[Assignment],
        waiting_queue: list[PatientState],
        state: DepartmentState,
    ) -> list[Assignment]:
        """Redistribute queues and pending assignments when a machine fails."""
        if not self.resched_cfg.equipment_failure_redistribution:
            return []

        # Find failed machine
        failed_res = next((r for r in resources if r.resource_id == failed_resource_id), None)
        if failed_res is None:
            return []

        failed_res.is_available = False
        failed_modality = failed_res.modality
        logger.warning(f"Equipment failure triggered on {failed_resource_id} ({failed_modality})")

        # Extract assignments on the failed resource
        displaced_assignments = [
            a for a in active_assignments if a.resource_id == failed_resource_id
        ]
        for da in displaced_assignments:
            active_assignments.remove(da)

        # Remaining functional resources for this modality
        surviving_resources = [
            r for r in resources
            if r.modality == failed_modality and r.resource_id != failed_resource_id and r.is_available
        ]

        new_assignments: list[Assignment] = []
        if not surviving_resources:
            logger.error(f"No surviving machines for modality {failed_modality}")
            return new_assignments

        # Re-dispatch displaced patients using surviving resources
        displaced_patients = [
            p for p in waiting_queue
            if any(da.patient_id == p.patient_id for da in displaced_assignments)
        ]

        for p in displaced_patients:
            asgn = self.dispatcher.dispatch_next(
                [p], surviving_resources, state, modality=failed_modality
            )
            if asgn:
                new_assignments.append(asgn)
                active_assignments.append(asgn)

        return new_assignments

    def detect_surge(
        self,
        current_arrival_rate: float,
        mean_rate: float,
        std_rate: float,
    ) -> dict[str, Any]:
        """Detect whether current arrival rate triggers a 2-sigma surge overflow."""
        sigma_threshold = self.resched_cfg.surge_detection_sigma
        cutoff = mean_rate + (sigma_threshold * std_rate)
        is_surge = current_arrival_rate > cutoff

        return {
            "is_surge": is_surge,
            "current_rate": current_arrival_rate,
            "surge_cutoff": cutoff,
            "overflow_protocol_active": is_surge and self.resched_cfg.surge_overflow_protocol,
            "recommendation": (
                "Activate surge protocol: mobilize reserve staff and open buffer slots"
                if is_surge
                else "Normal OPD load"
            ),
        }
