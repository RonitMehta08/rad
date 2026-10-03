"""Radiology Dynamic Scheduling & Optimization Package.

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

from typing import Any, Sequence

from src.scheduler.config import SchedulerConfig, load_scheduler_config
from src.scheduler.dispatcher import RealTimeDispatcher
from src.scheduler.fairness import FairnessEngine
from src.scheduler.models import (
    Assignment,
    DepartmentState,
    PatientState,
    Resource,
)
from src.scheduler.optimizer import DayAheadOptimizer
from src.scheduler.policies import (
    BaseSchedulingPolicy,
    FCFSPolicy,
    PriorityPolicy,
    SJFPolicy,
    WavePolicy,
    get_policy,
)
from src.scheduler.rescheduler import ReschedulingEngine
from src.utils.constants import ModalityType, UrgencyLevel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RadiologyScheduler:
    """Unified facade coordinating Day-Ahead Optimization, Real-Time Dispatch,

    Rawlsian Fairness, and Dynamic Event Rescheduling.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        wait_time_predictor: Any | None = None,
        noshow_predictor: Any | None = None,
    ) -> None:
        self.config = config or load_scheduler_config()
        self.optimizer = DayAheadOptimizer(config=self.config)
        self.dispatcher = RealTimeDispatcher(
            config=self.config, wait_time_predictor=wait_time_predictor
        )
        self.fairness = FairnessEngine(
            penalty_rate=self.config.fairness_penalty_rate,
            starvation_threshold_minutes=self.config.starvation_threshold_minutes,
        )
        self.rescheduler = ReschedulingEngine(
            config=self.config,
            dispatcher=self.dispatcher,
            noshow_predictor=noshow_predictor,
        )

    def compute_priority_score(
        self, patient: PatientState, state: DepartmentState
    ) -> float:
        """Compute Level 2 dynamic priority score for a patient."""
        return self.dispatcher.compute_priority_score(patient, state)

    def dispatch_next(
        self,
        queue: Sequence[PatientState],
        resources: Sequence[Resource],
        state: DepartmentState,
        modality: ModalityType | None = None,
    ) -> Assignment | None:
        """Dynamically evaluate queue and dispatch highest priority patient to best resource."""
        return self.dispatcher.dispatch_next(queue, resources, state, modality)

    def optimize_day_ahead(
        self,
        patients: Sequence[PatientState],
        resources: Sequence[Resource],
    ) -> dict[str, Any]:
        """Run Level 1 MILP day-ahead pre-scheduling."""
        return self.optimizer.optimize_schedule(patients, resources)

    def handle_emergency(
        self,
        emergency_patient: PatientState,
        active_assignments: list[Assignment],
        waiting_queue: list[PatientState],
        resources: list[Resource],
        state: DepartmentState,
    ) -> tuple[Assignment | None, PatientState | None]:
        """Trigger Level 3 emergency preemption and cascade rescheduling."""
        return self.rescheduler.handle_emergency_preemption(
            emergency_patient, active_assignments, waiting_queue, resources, state
        )

    def handle_noshow(
        self,
        scheduled_patient: PatientState,
        waiting_queue: list[PatientState],
        available_resources: list[Resource],
        state: DepartmentState,
    ) -> Assignment | None:
        """Trigger Level 3 no-show detection and pull-forward dispatch."""
        return self.rescheduler.handle_noshow_pullforward(
            scheduled_patient, waiting_queue, available_resources, state
        )

    def handle_equipment_failure(
        self,
        failed_resource_id: str,
        resources: list[Resource],
        active_assignments: list[Assignment],
        waiting_queue: list[PatientState],
        state: DepartmentState,
    ) -> list[Assignment]:
        """Trigger Level 3 equipment failure re-routing."""
        return self.rescheduler.handle_equipment_failure(
            failed_resource_id, resources, active_assignments, waiting_queue, state
        )

    def detect_surge(
        self, current_arrival_rate: float, mean_rate: float, std_rate: float
    ) -> dict[str, Any]:
        """Evaluate Level 3 surge arrival rate and overflow trigger."""
        return self.rescheduler.detect_surge(current_arrival_rate, mean_rate, std_rate)

    def order_by_policy(
        self,
        policy_name: str,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        """Order queue according to any registered baseline or comparative policy."""
        policy = get_policy(policy_name)
        return policy.order_queue(queue, state, modality)

    def compute_equity_metrics(
        self, waiting_times: Sequence[float]
    ) -> dict[str, float]:
        """Calculate Rawlsian equity, dispersion, and Gini coefficient."""
        return self.fairness.compute_equity_metrics(waiting_times)


__all__ = [
    "RadiologyScheduler",
    "DayAheadOptimizer",
    "RealTimeDispatcher",
    "ReschedulingEngine",
    "FairnessEngine",
    "PatientState",
    "Resource",
    "Assignment",
    "DepartmentState",
    "BaseSchedulingPolicy",
    "FCFSPolicy",
    "PriorityPolicy",
    "SJFPolicy",
    "WavePolicy",
    "get_policy",
    "load_scheduler_config",
    "SchedulerConfig",
]
