"""Live (in-memory) radiology department driven by the RadQueue scheduler.

Supports the real-time workflow behind the API's /schedule endpoints and the
dashboard's Smart Scheduler page: register patients, dispatch them to
machines with the Level-2 dispatcher, handle emergencies (Level-3 preemption),
equipment failures, completions, and give each waiting patient an ETA.

Owner: P5 (Dashboard & API) on top of P3's scheduler
Reference: MASTER_PROMPT §5.2, §7.3
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from src.scheduler import RadiologyScheduler
from src.scheduler.dispatcher import build_scheduler_context
from src.scheduler.models import Assignment, DepartmentState, PatientState, Resource
from src.simulation.entities import SimulationSettings
from src.utils.constants import SERVICE_TIME_PARAMS, ModalityType, UrgencyLevel
from src.utils.exceptions import InvalidInputError, ResourceNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiveDepartment:
    """Thread-safe in-memory department state for one hospital tier.

    Times are minutes since the department opened (``now``); callers advance
    the clock explicitly so behaviour is deterministic and testable.
    """

    def __init__(self, tier: str = "tier_1", wait_time_predictor: Any | None = None) -> None:
        settings = SimulationSettings.from_config(tier)
        self.tier = tier
        self.now = 0.0
        self.predictor = wait_time_predictor
        self.scheduler = RadiologyScheduler(wait_time_predictor=wait_time_predictor)
        self.resources: list[Resource] = [
            Resource(resource_id=f"{m.value}_{i + 1}", modality=m, name=f"{m.value.upper()} room {i + 1}")
            for m, count in settings.machines.items()
            for i in range(count)
        ]
        self.waiting: list[PatientState] = []
        self.active: list[PatientState] = []
        self.assignments: list[Assignment] = []
        self.completed: list[dict[str, Any]] = []
        self.events: list[str] = []
        self._lock = threading.RLock()

    # -- state ------------------------------------------------------------------

    def _state(self) -> DepartmentState:
        for p in self.waiting:
            p.current_wait_minutes = max(0.0, self.now - p.arrival_time_minutes)
        waits = [p.current_wait_minutes for p in self.waiting]
        return DepartmentState(
            current_time_minutes=self.now,
            avg_wait_minutes=float(np.mean(waits)) if waits else 0.0,
            queue_length={m: sum(1 for p in self.waiting if p.modality == m) for m in ModalityType},
            waiting_patients=list(self.waiting),
            active_patients=list(self.active),
            resources=list(self.resources),
            active_assignments=list(self.assignments),
        )

    def _resource(self, resource_id: str) -> Resource:
        for r in self.resources:
            if r.resource_id == resource_id:
                return r
        raise ResourceNotFoundError(f"Unknown resource_id '{resource_id}'")

    def _start(self, assignment: Assignment) -> None:
        patient = next(p for p in self.waiting if p.patient_id == assignment.patient_id)
        resource = self._resource(assignment.resource_id)
        self.waiting.remove(patient)
        patient.status = "in_scan"
        self.active.append(patient)
        resource.is_available = False
        resource.current_patient_id = patient.patient_id
        resource.busy_until_minute = self.now + assignment.estimated_duration_minutes
        assignment.start_time_minutes = self.now
        assignment.predicted_wait_minutes = round(self.now - patient.arrival_time_minutes, 1)
        self.assignments.append(assignment)
        self.events.append(f"t={self.now:.0f}: {patient.patient_id} -> {resource.resource_id}")

    def _dispatch_all(self) -> list[Assignment]:
        started: list[Assignment] = []
        for modality in ModalityType:
            while True:
                free = [r for r in self.resources if r.modality == modality and r.is_available]
                queue = [p for p in self.waiting if p.modality == modality]
                if not free or not queue:
                    break
                assignment = self.scheduler.dispatch_next(queue, free, self._state(), modality)
                if assignment is None:
                    break
                self._start(assignment)
                started.append(assignment)
        return started

    # -- commands ---------------------------------------------------------------

    def register_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        """Add a patient to the queue and dispatch if capacity allows.

        Emergencies with no free machine trigger Level-3 preemption of a routine scan.
        """
        with self._lock:
            pid = patient["patient_id"]
            if any(p.patient_id == pid for p in self.waiting + self.active):
                raise InvalidInputError(f"patient_id '{pid}' is already in the department")
            state = PatientState(
                patient_id=pid,
                modality=ModalityType(patient["modality"]),
                urgency=UrgencyLevel(patient.get("urgency", "routine")),
                visit_type=patient.get("visit_type", "walk_in"),
                arrival_time_minutes=self.now,
                estimated_duration_minutes=patient.get("estimated_duration_minutes"),
                features={k: v for k, v in patient.items() if k not in {"patient_id"}},
            )
            self.waiting.append(state)
            preempted = self._maybe_preempt(state)
            started = self._dispatch_all()
            mine = next((a for a in started if a.patient_id == pid), None)
            return {
                "patient_id": pid,
                "status": "in_scan" if mine else "waiting",
                "assignment": mine.model_dump() if mine else None,
                "preempted_patient_id": preempted,
                "eta": None if mine else self.estimate_wait(pid),
            }

    def _maybe_preempt(self, patient: PatientState) -> str | None:
        if patient.urgency != UrgencyLevel.EMERGENCY:
            return None
        if any(r.is_available for r in self.resources if r.modality == patient.modality):
            return None
        emergency_asgn, displaced = self.scheduler.handle_emergency(
            patient, self.assignments, self.waiting, self.resources, self._state(), self.active,
        )
        if emergency_asgn is None or displaced is None:
            return None
        # Displaced routine patient goes back to the front of the waiting list with their original arrival.
        self.active = [p for p in self.active if p.patient_id != displaced.patient_id]
        displaced.status = "waiting"
        self.waiting.insert(0, displaced)  # displaced patient keeps their place at the front
        self.assignments.remove(emergency_asgn)
        resource = self._resource(emergency_asgn.resource_id)
        resource.is_available = True
        self.events.append(f"t={self.now:.0f}: EMERGENCY {patient.patient_id} preempted {displaced.patient_id}")
        return displaced.patient_id

    def complete_scan(self, resource_id: str) -> dict[str, Any]:
        """Mark the scan on ``resource_id`` finished and dispatch the next patient."""
        with self._lock:
            resource = self._resource(resource_id)
            if resource.is_available or resource.current_patient_id is None:
                raise InvalidInputError(f"Resource '{resource_id}' is not currently scanning")
            pid = resource.current_patient_id
            patient = next(p for p in self.active if p.patient_id == pid)
            self.active.remove(patient)
            asgn = next(a for a in self.assignments if a.patient_id == pid)
            self.assignments.remove(asgn)
            resource.total_busy_minutes += self.now - asgn.start_time_minutes
            resource.is_available, resource.current_patient_id = True, None
            self.completed.append({"patient_id": pid, "resource_id": resource_id,
                                   "wait_minutes": asgn.predicted_wait_minutes, "completed_at": self.now})
            started = self._dispatch_all()
            return {"completed_patient_id": pid, "started": [a.model_dump() for a in started]}

    def advance_clock(self, minutes: float) -> dict[str, Any]:
        """Advance time, auto-completing scans whose estimated end has passed."""
        if minutes < 0:
            raise InvalidInputError("minutes must be non-negative")
        with self._lock:
            target = self.now + minutes
            done: list[str] = []
            while True:
                busy = [r for r in self.resources if not r.is_available and r.current_patient_id]
                due = [r for r in busy if r.busy_until_minute <= target]
                if not due:
                    break
                nxt = min(due, key=lambda r: r.busy_until_minute)
                self.now = max(self.now, nxt.busy_until_minute)
                done.append(self.complete_scan(nxt.resource_id)["completed_patient_id"])
            self.now = target
            self._dispatch_all()
            return {"now": self.now, "completed": done}

    def fail_equipment(self, resource_id: str) -> dict[str, Any]:
        """Take a machine out of service and re-route its patient (Level 3)."""
        with self._lock:
            resource = self._resource(resource_id)
            displaced = resource.current_patient_id
            if displaced:
                patient = next(p for p in self.active if p.patient_id == displaced)
                self.active.remove(patient)
                self.assignments = [a for a in self.assignments if a.patient_id != displaced]
                patient.status = "waiting"
                self.waiting.append(patient)
            resource.is_available, resource.current_patient_id = False, None
            resource.busy_until_minute = float("inf")
            self.events.append(f"t={self.now:.0f}: equipment failure on {resource_id}")
            started = self._dispatch_all()
            return {"failed_resource_id": resource_id, "requeued_patient_id": displaced,
                    "started": [a.model_dump() for a in started]}

    # -- queries ----------------------------------------------------------------

    def estimate_wait(self, patient_id: str) -> dict[str, Any]:
        """Patient-facing ETA (USP 10) for someone already in the queue.

        The primary ETA uses the patient's queue position (under the live
        scheduling policy) and when each machine frees up. The ML model's
        registration-time estimate is returned alongside as ``ml_estimate_minutes``;
        it does not know the queue position, so it is a second opinion only.
        """
        state = self._state()
        patient = next((p for p in self.waiting if p.patient_id == patient_id), None)
        if patient is None:
            raise ResourceNotFoundError(f"patient_id '{patient_id}' is not waiting")
        machines = [r for r in self.resources if r.modality == patient.modality and r.busy_until_minute != float("inf")]
        if not machines:
            return {"eta_minutes": None, "method": "no_working_machine", "ml_estimate_minutes": None}
        ranked = [p for p, _ in self.scheduler.dispatcher.rank_queue(self.waiting, state, patient.modality)]
        ahead = [p.patient_id for p in ranked].index(patient_id)
        # Simulate machines picking up the patients ahead of us in order
        free_at = sorted(max(self.now, r.busy_until_minute) for r in machines)
        for p in ranked[:ahead]:
            free_at[0] += p.estimated_duration_minutes or SERVICE_TIME_PARAMS[p.modality]["median_minutes"]
            free_at.sort()
        result: dict[str, Any] = {
            "eta_minutes": round(free_at[0] - self.now, 1),
            "method": "queue_position",
            "queue_position": ahead + 1,
            "ml_estimate_minutes": None,
        }
        if self.predictor is not None:
            try:
                raw = dict(patient.features)
                raw.update(build_scheduler_context(patient, machines[0], state))
                result["ml_estimate_minutes"] = round(self.predictor.predict(raw), 1)
            except Exception as exc:
                logger.warning(f"ML estimate failed for {patient_id}: {exc}")
        return result

    def snapshot(self) -> dict[str, Any]:
        """Serializable view of the department."""
        with self._lock:
            state = self._state()
            return {
                "tier": self.tier,
                "now": self.now,
                "avg_wait_minutes": round(state.avg_wait_minutes, 1),
                "queue_length": {m.value: n for m, n in state.queue_length.items()},
                "waiting": [
                    {"patient_id": p.patient_id, "modality": p.modality.value, "urgency": p.urgency.value,
                     "waited_minutes": round(p.current_wait_minutes, 1)}
                    for p in self.waiting
                ],
                "resources": [
                    {"resource_id": r.resource_id, "modality": r.modality.value,
                     "status": "busy" if r.current_patient_id else ("down" if not r.is_available else "free"),
                     "current_patient_id": r.current_patient_id,
                     "busy_until": None if r.busy_until_minute in (0.0, float("inf")) else round(r.busy_until_minute, 1)}
                    for r in self.resources
                ],
                "completed": list(self.completed),
                "events": self.events[-20:],
            }
