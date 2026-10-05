"""Core SimPy digital-twin engine.

The engine owns machines and technologists and calls the scheduling policy
whenever capacity frees up or a patient joins a queue, so the policy decides
who is scanned next. Department state (average wait, per-modality queue
lengths) is refreshed before every decision.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §8.1, §8.2
"""

from __future__ import annotations

import numpy as np
import simpy

from src.scheduler.models import DepartmentState
from src.scheduler.policies import BaseSchedulingPolicy
from src.simulation.arrival_patterns import generate_day_patients
from src.simulation.entities import SimPatient, SimulationSettings
from src.simulation.kpi_collector import KPICollector
from src.simulation.processes import arrival_process, make_radiologist_pool, queue_monitor
from src.utils.constants import ModalityType, UrgencyLevel

MINUTES_PER_HOUR: int = 60


class DayRun:
    """State of one simulated operating day."""

    def __init__(
        self,
        settings: SimulationSettings,
        policy: BaseSchedulingPolicy,
        kpi: KPICollector,
        day: int,
    ) -> None:
        self.env = simpy.Environment()
        self.settings = settings
        self.policy = policy
        self.kpi = kpi
        self.day = day
        self.open_minute = settings.open_hour * MINUTES_PER_HOUR
        self.close_minute = settings.close_hour * MINUTES_PER_HOUR
        self.overbooking_enabled = bool(getattr(policy, "overbooking_enabled", False))

        staffing = settings.staffing_at(self.open_minute)
        self.registration = simpy.Resource(self.env, capacity=max(1, settings.registration_desks))
        self.radiologists = make_radiologist_pool(self.env, staffing.radiologists)
        self.free_machines: dict[ModalityType, int] = dict(settings.machines)
        self.busy_technologists = 0
        self.queues: dict[ModalityType, list[SimPatient]] = {m: [] for m in ModalityType}
        self.dispatch_events: dict[str, simpy.Event] = {}

    # -- capacity -------------------------------------------------------------

    def _technologist_available(self) -> bool:
        return self.busy_technologists < self.settings.staffing_at(self.env.now).technologists

    def _reserved_for_emergencies(self, modality: ModalityType) -> int:
        """Machines held back for emergencies/urgent cases (emergency buffer).

        Rounded to the nearest machine (20% of 3 X-ray rooms reserves 1), but the
        last machine of a modality is never reserved.
        """
        machines = self.settings.machines.get(modality, 0)
        return min(machines - 1, round(self.settings.emergency_buffer_fraction * machines)) if machines > 1 else 0

    def _department_state(self) -> DepartmentState:
        waiting = [p for q in self.queues.values() for p in q]
        waits = [self.env.now - p.queue_entry_time for p in waiting if p.queue_entry_time is not None]
        return DepartmentState(
            current_time_minutes=self.env.now,
            avg_wait_minutes=float(np.mean(waits)) if waits else 0.0,
            queue_length={m: len(q) for m, q in self.queues.items()},
        )

    # -- dispatch -------------------------------------------------------------

    def _select_next(self, modality: ModalityType) -> SimPatient | None:
        queue = self.queues[modality]
        state = self._department_state()
        ordered = self.policy.order_queue([p.to_patient_state(self.env.now) for p in queue], state, modality)
        by_id = {p.id: p for p in queue}
        if self.free_machines[modality] > self._reserved_for_emergencies(modality):
            return by_id[ordered[0].patient_id] if ordered else None
        # Only buffer capacity left: serve non-routine patients only
        for ps in ordered:
            if ps.urgency != UrgencyLevel.ROUTINE:
                return by_id[ps.patient_id]
        return None

    def dispatch(self, modality: ModalityType) -> None:
        """Start as many queued patients of ``modality`` as capacity allows."""
        while self.queues[modality] and self.free_machines[modality] > 0 and self._technologist_available():
            patient = self._select_next(modality)
            if patient is None:
                return
            self.queues[modality].remove(patient)
            self.free_machines[modality] -= 1
            self.busy_technologists += 1
            patient.scan_start_time = self.env.now
            self.dispatch_events.pop(patient.id).succeed()

    def dispatch_all(self) -> None:
        """Re-run dispatch for every modality (a freed technologist can serve any)."""
        for modality in ModalityType:
            self.dispatch(modality)

    def release_machine(self, patient: SimPatient) -> None:
        """Free the machine + technologist after imaging and immediately re-dispatch."""
        self.free_machines[patient.modality] += 1
        self.busy_technologists -= 1
        self.kpi.record_busy_interval(
            patient.modality, patient.scan_start_time or self.env.now, self.env.now,
            self.open_minute, self.close_minute,
        )
        self.dispatch(patient.modality)
        self.dispatch_all()


class RadiologySimulation:
    """Multi-day radiology department simulation under one scheduling policy."""

    def __init__(
        self,
        policy: BaseSchedulingPolicy,
        settings: SimulationSettings | None = None,
        seed: int = 42,
    ) -> None:
        """Create a simulation.

        Args:
            policy: Scheduling policy deciding queue order.
            settings: Department settings (defaults to Tier 1, 1 day).
            seed: Replication seed; equal seeds give identical patient streams.
        """
        self.policy = policy
        self.settings = settings or SimulationSettings.from_config("tier_1")
        self.seed = seed
        self.kpi = KPICollector()

    def run(self) -> KPICollector:
        """Simulate all configured days and return the collected KPIs."""
        for day in range(self.settings.days):
            self._run_day(day)
        self.kpi.days_simulated = self.settings.days
        return self.kpi

    def _run_day(self, day: int) -> None:
        run = DayRun(self.settings, self.policy, self.kpi, day)
        patients = generate_day_patients(self.settings, day, self.seed)
        run.env.process(arrival_process(run, patients))
        run.env.process(queue_monitor(run))
        run.env.run()  # until every admitted patient has departed (overtime allowed)
        operating_minutes = run.close_minute - run.open_minute
        for modality, count in self.settings.machines.items():
            self.kpi.record_capacity(modality, count * operating_minutes)
