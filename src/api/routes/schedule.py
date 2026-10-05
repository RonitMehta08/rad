"""Scheduling endpoints: live dispatch, emergencies, equipment failure, day-ahead MILP.

Owner: P5 (Dashboard & API) on top of P3's scheduler
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_department
from src.api.schemas import AdvanceClockRequest, DayAheadRequest, RegisterPatientRequest
from src.scheduler import DayAheadOptimizer, PatientState, Resource
from src.services.live_department import LiveDepartment
from src.simulation.entities import SimulationSettings

router = APIRouter(prefix="/schedule", tags=["Scheduling"])


@router.get("/state")
async def get_state(department: LiveDepartment = Depends(get_department)) -> dict[str, Any]:
    """Current queues, machine status, recent events and completed scans."""
    return department.snapshot()


@router.post("/patients")
async def register_patient(
    request: RegisterPatientRequest,
    department: LiveDepartment = Depends(get_department),
) -> dict[str, Any]:
    """Register a patient; dispatches immediately if a machine is free.

    Emergencies with no free machine preempt a routine scan (Level-3 rescheduling).
    """
    return await run_in_threadpool(department.register_patient, request.model_dump(mode="json", exclude_none=True))


@router.get("/patients/{patient_id}/eta")
async def patient_eta(patient_id: str, department: LiveDepartment = Depends(get_department)) -> dict[str, Any]:
    """Patient-facing estimated wait for someone still in the queue."""
    return await run_in_threadpool(department.estimate_wait, patient_id)


@router.post("/resources/{resource_id}/complete")
async def complete_scan(resource_id: str, department: LiveDepartment = Depends(get_department)) -> dict[str, Any]:
    """Mark a scan finished and dispatch the next patient for that machine."""
    return department.complete_scan(resource_id)


@router.post("/resources/{resource_id}/failure")
async def equipment_failure(resource_id: str, department: LiveDepartment = Depends(get_department)) -> dict[str, Any]:
    """Take a machine out of service and re-route its patient."""
    return department.fail_equipment(resource_id)


@router.post("/clock/advance")
async def advance_clock(
    request: AdvanceClockRequest,
    department: LiveDepartment = Depends(get_department),
) -> dict[str, Any]:
    """Advance department time; scans past their estimated end are completed."""
    return department.advance_clock(request.minutes)


def _solve_day_ahead(request: DayAheadRequest) -> dict[str, Any]:
    settings = SimulationSettings.from_config(request.tier)
    resources = [
        Resource(resource_id=f"{m.value}_{i + 1}", modality=m)
        for m, count in settings.machines.items()
        for i in range(count)
    ]
    patients = [PatientState(**p.model_dump(exclude_none=True)) for p in request.patients]
    result = DayAheadOptimizer().optimize_schedule(patients, resources)
    result["assignments"] = [a.model_dump(mode="json") for a in result["assignments"]]
    return result


@router.post("/day-ahead")
async def day_ahead(request: DayAheadRequest) -> dict[str, Any]:
    """Level-1 MILP pre-scheduling of booked appointments onto machines and 15-min slots."""
    return await run_in_threadpool(_solve_day_ahead, request)
