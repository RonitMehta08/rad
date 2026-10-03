"""Level 1: Day-ahead slot-level Mixed-Integer Linear Programming (MILP) scheduler.

Uses PuLP with CBC solver to pre-schedule appointments across machines and time slots,
incorporating:
- Machine and modality compatibility
- Non-overlapping machine capacity
- Emergency capacity reservation buffers (10-15%)
- Clinical urgency weighting and maximum waiting ceilings
- Multi-objective minimization of total wait time and idle machine gaps

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

import collections
import math
from typing import Any, Sequence
import pulp

from src.scheduler.config import SchedulerConfig, load_scheduler_config
from src.scheduler.models import Assignment, PatientState, Resource
from src.utils.constants import (
    ModalityType,
    UrgencyLevel,
    URGENCY_WEIGHTS,
    SERVICE_TIME_PARAMS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DayAheadOptimizer:
    """MILP-based day-ahead slot scheduling optimizer."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or load_scheduler_config()
        self.milp_cfg = self.config.milp
        self.slot_duration = self.milp_cfg.slot_duration_minutes
        self.total_slots = int(
            (self.milp_cfg.planning_horizon_hours * 60) // self.slot_duration
        )
        self.buffer_fraction = self.milp_cfg.emergency_buffer_fraction
        self.max_routine_wait = self.config.constraints.max_wait_routine_minutes
        self.max_urgent_wait = self.config.constraints.max_wait_urgent_minutes
        self.max_emergency_wait = self.config.constraints.max_wait_emergency_minutes

    def _slots_for_patient(self, patient: PatientState) -> int:
        """Calculate required number of discrete time slots for a patient's scan."""
        dur = (
            patient.estimated_duration_minutes
            or SERVICE_TIME_PARAMS[patient.modality]["median_minutes"]
        )
        return max(1, math.ceil(dur / self.slot_duration))

    def _max_allowed_wait(self, urgency: UrgencyLevel) -> float:
        if urgency == UrgencyLevel.EMERGENCY:
            return float(self.max_emergency_wait)
        if urgency == UrgencyLevel.URGENT:
            return float(self.max_urgent_wait)
        return float(self.max_routine_wait)

    def optimize_schedule(
        self,
        patients: Sequence[PatientState],
        resources: Sequence[Resource],
        day_start_hour: int = 8,
    ) -> dict[str, Any]:
        """Formulate and solve MILP schedule for the planning horizon.

        Args:
            patients: Pre-booked or anticipated scheduled patients.
            resources: Available imaging machines.
            day_start_hour: Hospital opening hour (default: 8 AM).

        Returns:
            Dictionary containing:
            - 'assignments': list of Assignment objects
            - 'status': Solver status string ("Optimal", "Feasible", "Infeasible")
            - 'objective_value': float
            - 'unassigned_patients': list of patient IDs not scheduled
            - 'slot_schedule': dict mapping resource_id to list of patient IDs per slot
        """
        prob = pulp.LpProblem("RadiologyDayAheadSchedule", pulp.LpMinimize)

        # O(1) patient lookup
        patients_by_id: dict[str, PatientState] = {p.patient_id: p for p in patients}

        # Precompute patient slot durations and arrival slots
        patient_slots: dict[str, int] = {}
        arrival_slots: dict[str, int] = {}
        for p in patients:
            patient_slots[p.patient_id] = self._slots_for_patient(p)
            arr_slot = int(p.arrival_time_minutes // self.slot_duration)
            arrival_slots[p.patient_id] = max(0, min(self.total_slots - 1, arr_slot))

        # Binary decision variable: x[p, r, s] = 1 if patient p starts on machine r at slot s
        x: dict[tuple[str, str, int], pulp.LpVariable] = {}
        valid_triplets: list[tuple[str, str, int]] = []
        occupancy_by_resource_slot: dict[tuple[str, int], list[pulp.LpVariable]] = collections.defaultdict(list)

        for p in patients:
            req_slots = patient_slots[p.patient_id]
            arr_slot = arrival_slots[p.patient_id]
            max_wait_slots = int(self._max_allowed_wait(p.urgency) // self.slot_duration)
            latest_start_slot = min(self.total_slots - req_slots, arr_slot + max_wait_slots)

            compatible_resources = [r for r in resources if r.modality == p.modality]
            for r in compatible_resources:
                for s in range(arr_slot, latest_start_slot + 1):
                    triplet = (p.patient_id, r.resource_id, s)
                    var = pulp.LpVariable(
                        f"x_{p.patient_id}_{r.resource_id}_{s}", cat=pulp.LpBinary
                    )
                    x[triplet] = var
                    valid_triplets.append(triplet)

                    # Map to resource and time slot occupancy (s <= t < s + req_slots)
                    for t in range(s, min(self.total_slots, s + req_slots)):
                        occupancy_by_resource_slot[(r.resource_id, t)].append(var)

        # Unassigned penalty variable per patient to guarantee feasibility
        u: dict[str, pulp.LpVariable] = {
            p.patient_id: pulp.LpVariable(f"u_{p.patient_id}", cat=pulp.LpBinary)
            for p in patients
        }

        # Constraint 1: Each patient is either assigned exactly once or flagged unassigned
        for p in patients:
            assigned_expr = pulp.lpSum(
                x[(p.patient_id, r, s)]
                for (pid, r, s) in valid_triplets
                if pid == p.patient_id
            )
            prob += assigned_expr + u[p.patient_id] == 1, f"AssignOnce_{p.patient_id}"

        # Constraint 2: Machine capacity & no overlap (O(R × S) vectorized via indexed occupancy)
        for r in resources:
            for t in range(self.total_slots):
                vars_at_slot = occupancy_by_resource_slot.get((r.resource_id, t), [])
                if vars_at_slot:
                    prob += pulp.lpSum(vars_at_slot) <= 1, f"Capacity_{r.resource_id}_slot_{t}"

        # Constraint 3: Emergency buffer reservation per modality
        for mod in ModalityType:
            mod_resources = [r for r in resources if r.modality == mod]
            if not mod_resources:
                continue
            total_mod_slots = len(mod_resources) * self.total_slots
            max_schedulable_slots = int(total_mod_slots * (1.0 - self.buffer_fraction))

            mod_occupancy_expr = []
            for (pid, rid, s) in valid_triplets:
                if any(r.resource_id == rid for r in mod_resources):
                    dur = patient_slots[pid]
                    mod_occupancy_expr.append(dur * x[(pid, rid, s)])

            if mod_occupancy_expr:
                prob += (
                    pulp.lpSum(mod_occupancy_expr) <= max_schedulable_slots,
                    f"EmergencyBuffer_{mod.value}",
                )

        # Objective function: weighted wait time + unassigned penalty
        wait_terms = []
        for (pid, rid, s) in valid_triplets:
            p_obj = patients_by_id[pid]
            arr_slot = arrival_slots[pid]
            wait_minutes = (s - arr_slot) * self.slot_duration
            urgency_wt = URGENCY_WEIGHTS.get(p_obj.urgency, 1)
            wait_terms.append(wait_minutes * urgency_wt * x[(pid, rid, s)])

        penalty_terms = []
        for p in patients:
            penalty = 10_000.0 * URGENCY_WEIGHTS.get(p.urgency, 1)
            penalty_terms.append(penalty * u[p.patient_id])

        prob += pulp.lpSum(wait_terms) + pulp.lpSum(penalty_terms)

        # Solve with PuLP CBC solver
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=self.milp_cfg.time_limit_seconds,
            gapRel=self.milp_cfg.mip_gap,
        )
        prob.solve(solver)

        status_str = pulp.LpStatus.get(prob.status, "Unknown")
        logger.info(f"MILP solve completed with status: {status_str}")

        # Extract assignments
        assignments: list[Assignment] = []
        unassigned: list[str] = []
        slot_schedule: dict[str, list[str | None]] = {
            r.resource_id: [None] * self.total_slots for r in resources
        }

        for p in patients:
            if pulp.value(u[p.patient_id]) and pulp.value(u[p.patient_id]) > 0.5:
                unassigned.append(p.patient_id)

        for (pid, rid, s) in valid_triplets:
            val = pulp.value(x[(pid, rid, s)])
            if val is not None and val > 0.5:
                p_obj = patients_by_id[pid]
                start_min = s * self.slot_duration
                arr_slot = arrival_slots[pid]
                wait_min = (s - arr_slot) * self.slot_duration
                dur_min = (
                    p_obj.estimated_duration_minutes
                    or SERVICE_TIME_PARAMS[p_obj.modality]["median_minutes"]
                )
                dur_slots = patient_slots[pid]

                assignments.append(
                    Assignment(
                        patient_id=pid,
                        resource_id=rid,
                        slot_index=s,
                        start_time_minutes=float(start_min),
                        estimated_duration_minutes=float(dur_min),
                        predicted_wait_minutes=float(wait_min),
                        urgency=p_obj.urgency,
                    )
                )

                for t in range(s, min(self.total_slots, s + dur_slots)):
                    slot_schedule[rid][t] = pid

        return {
            "status": status_str,
            "objective_value": float(pulp.value(prob.objective)) if prob.objective else 0.0,
            "assignments": assignments,
            "unassigned_patients": unassigned,
            "slot_schedule": slot_schedule,
            "slot_duration_minutes": self.slot_duration,
            "total_slots": self.total_slots,
        }
