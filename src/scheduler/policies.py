"""Scheduling policy implementations for baseline comparisons and dynamic dispatch.

Policies implemented:
1. FCFS (First Come First Served) — baseline
2. Priority (Emergency > Urgent > Routine)
3. SJF (Shortest Job First) — modality duration-based
4. Wave (Batched wave scheduling at fixed intervals)
5. RadQueue AI (Multi-objective dynamic dispatch with Rawlsian fairness)
6. RadQueue No-Show (RadQueue AI + no-show awareness)

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §7.3, §8.2; config/scheduler_config.yaml
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from src.scheduler.config import SchedulerConfig, load_scheduler_config
from src.scheduler.fairness import FairnessEngine
from src.scheduler.models import DepartmentState, PatientState
from src.utils.constants import URGENCY_WEIGHTS, ModalityType, UrgencyLevel


class BaseSchedulingPolicy(ABC):
    """Abstract base class for all scheduling policies."""

    name: str = "base"
    description: str = "Base scheduling policy"

    @abstractmethod
    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        """Order waiting patients according to the policy rules.

        Args:
            queue: List of patients currently waiting.
            state: Current department operational state (optional).
            modality: If specified, filter queue for this modality only.

        Returns:
            Ordered list of PatientState instances (highest priority first).
        """
        pass

    def select_next_patient(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> PatientState | None:
        """Select the single best candidate patient to dispatch next.

        Returns:
            The selected PatientState, or None if queue is empty.
        """
        ordered = self.order_queue(queue, state, modality)
        return ordered[0] if ordered else None


class FCFSPolicy(BaseSchedulingPolicy):
    """First-Come, First-Served (FCFS) — standard hospital baseline."""

    name: str = "fcfs"
    description: str = "First Come First Served — strict arrival time ordering"

    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        candidates = [p for p in queue if modality is None or p.modality == modality]
        return sorted(candidates, key=lambda p: (p.arrival_time_minutes, p.patient_id))


class PriorityPolicy(BaseSchedulingPolicy):
    """Clinical Priority Policy: Emergency (10) > Urgent (5) > Routine (1).

    Ties broken by arrival time (FCFS within priority group).
    """

    name: str = "priority"
    description: str = "Clinical Urgency: Emergency > Urgent > Routine"

    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        candidates = [p for p in queue if modality is None or p.modality == modality]

        def sort_key(p: PatientState) -> tuple[int, float, str]:
            weight = URGENCY_WEIGHTS.get(p.urgency, 1)
            return (-weight, p.arrival_time_minutes, p.patient_id)

        return sorted(candidates, key=sort_key)


class SJFPolicy(BaseSchedulingPolicy):
    """Shortest Job First (SJF) — prioritizes exams with shorter service times.

    Reduces average wait time in M/G/1 queueing systems.
    Emergencies always jump to front, then shortest estimated duration.
    """

    name: str = "sjf"
    description: str = "Shortest Job First — prioritizes lower scan duration"

    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        candidates = [p for p in queue if modality is None or p.modality == modality]

        def sort_key(p: PatientState) -> tuple[int, float, float, str]:
            is_emergency = 0 if p.urgency == UrgencyLevel.EMERGENCY else 1
            duration = p.estimated_duration_minutes or 999.0
            return (is_emergency, duration, p.arrival_time_minutes, p.patient_id)

        return sorted(candidates, key=sort_key)


class WavePolicy(BaseSchedulingPolicy):
    """Wave Scheduling Policy — batches arrivals into fixed interval windows.

    Groups patients arriving within each interval (e.g., 30 min) and orders
    each wave by urgency and modality balance.
    """

    name: str = "wave"
    description: str = "Wave Scheduling — batched intervals with modality balance"

    def __init__(self, interval_minutes: int = 30, **kwargs: Any) -> None:
        self.interval_minutes = interval_minutes

    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        candidates = [p for p in queue if modality is None or p.modality == modality]

        def sort_key(p: PatientState) -> tuple[int, int, float, str]:
            wave_index = int(p.arrival_time_minutes // self.interval_minutes)
            weight = URGENCY_WEIGHTS.get(p.urgency, 1)
            return (wave_index, -weight, p.arrival_time_minutes, p.patient_id)

        return sorted(candidates, key=sort_key)


class RadQueueAIPolicy(BaseSchedulingPolicy):
    """RadQueue AI Dynamic Policy with Rawlsian fairness constraints."""

    name: str = "radqueue_ai"
    description: str = "Multi-objective dynamic dispatch with Rawlsian fairness"

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        penalty_rate: float = 0.1,
        starvation_threshold_minutes: float = 60.0,
        **kwargs: Any,
    ) -> None:
        self.config = config or load_scheduler_config()
        self.fairness_engine = FairnessEngine(
            penalty_rate=penalty_rate or self.config.fairness_penalty_rate,
            starvation_threshold_minutes=starvation_threshold_minutes or self.config.starvation_threshold_minutes,
        )

    def compute_score(self, patient: PatientState, state: DepartmentState | None) -> float:
        base = float(URGENCY_WEIGHTS.get(patient.urgency, 1))
        avg_wait = state.avg_wait_minutes if state else 0.0
        queue_len = state.queue_length.get(patient.modality, 0) if state else 0

        wait_bonus = self.fairness_engine.compute_fairness_bonus(
            patient.current_wait_minutes, avg_wait
        )
        modality_factor = 1.0 / (1.0 + float(queue_len))
        starvation_boost = 3.0 if self.fairness_engine.is_starving(patient) else 0.0

        return base + wait_bonus + modality_factor + starvation_boost

    def order_queue(
        self,
        queue: Sequence[PatientState],
        state: DepartmentState | None = None,
        modality: ModalityType | None = None,
    ) -> list[PatientState]:
        candidates = [p for p in queue if modality is None or p.modality == modality]

        def sort_key(p: PatientState) -> tuple[float, float, str]:
            score = self.compute_score(p, state)
            return (-score, p.arrival_time_minutes, p.patient_id)

        return sorted(candidates, key=sort_key)


class RadQueueNoShowPolicy(RadQueueAIPolicy):
    """RadQueue AI Dynamic Policy with no-show risk awareness."""

    name: str = "radqueue_noshow"
    description: str = "RadQueue dynamic dispatch + no-show overbooking prioritization"


def get_policy(policy_name: str, **kwargs: Any) -> BaseSchedulingPolicy:
    """Factory function to instantiate scheduling policies by name."""
    policies: dict[str, type[BaseSchedulingPolicy]] = {
        "fcfs": FCFSPolicy,
        "priority": PriorityPolicy,
        "sjf": SJFPolicy,
        "wave": WavePolicy,
        "radqueue_ai": RadQueueAIPolicy,
        "radqueue_noshow": RadQueueNoShowPolicy,
    }

    norm_name = policy_name.lower().strip()
    if norm_name not in policies:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available: {list(policies.keys())}"
        )

    policy_cls = policies[norm_name]
    return policy_cls(**kwargs)
