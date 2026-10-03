"""Scheduling policy implementations for baseline comparisons and dynamic dispatch.

Policies implemented:
1. FCFS (First Come First Served) — baseline
2. Priority (Emergency > Urgent > Routine)
3. SJF (Shortest Job First) — modality duration-based
4. Wave (Batched wave scheduling at fixed intervals)

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §7.3, §8.2; config/scheduler_config.yaml
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.scheduler.models import DepartmentState, PatientState
from src.utils.constants import ModalityType, UrgencyLevel, URGENCY_WEIGHTS


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
        # Sort strictly by arrival time
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
            # Higher weight first (so -weight), then arrival time
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
            # Emergencies first
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

    def __init__(self, interval_minutes: int = 30) -> None:
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
            # Waves processed in chronological order; within wave, higher urgency first
            return (wave_index, -weight, p.arrival_time_minutes, p.patient_id)

        return sorted(candidates, key=sort_key)


def get_policy(policy_name: str, **kwargs) -> BaseSchedulingPolicy:
    """Factory function to instantiate scheduling policies by name."""
    policies: dict[str, type[BaseSchedulingPolicy]] = {
        "fcfs": FCFSPolicy,
        "priority": PriorityPolicy,
        "sjf": SJFPolicy,
        "wave": WavePolicy,
    }

    norm_name = policy_name.lower().strip()
    if norm_name not in policies:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available: {list(policies.keys())}"
        )

    policy_cls = policies[norm_name]
    return policy_cls(**kwargs)
