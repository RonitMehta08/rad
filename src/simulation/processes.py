"""SimPy processes for patient flow through the radiology department.

Flow: arrival -> registration -> modality queue -> (policy dispatch) ->
technologist prep + imaging -> radiologist reporting -> departure.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §8.1
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import simpy

from src.simulation.entities import SimPatient
from src.utils.constants import ModalityType

if TYPE_CHECKING:
    from src.simulation.engine import DayRun

QUEUE_SAMPLE_INTERVAL_MINUTES: float = 15.0


def arrival_process(run: DayRun, patients: list[SimPatient]) -> Generator[Any, Any, None]:
    """Release pre-sampled patients into the department at their arrival times."""
    for patient in patients:
        yield run.env.timeout(max(0.0, patient.arrival_time - run.env.now))
        if patient.is_overbooked and not run.overbooking_enabled:
            continue  # stand-by patients only exist under an overbooking policy
        if not patient.will_show:
            run.kpi.record_noshow()
            continue
        run.env.process(patient_flow(run, patient))


def patient_flow(run: DayRun, patient: SimPatient) -> Generator[Any, Any, None]:
    """Registration, then wait in the modality queue until the policy dispatches us."""
    with run.registration.request() as req:
        yield req
        yield run.env.timeout(patient.registration_minutes)

    patient.queue_entry_time = run.env.now
    dispatched = run.env.event()
    run.dispatch_events[patient.id] = dispatched
    run.queues[patient.modality].append(patient)
    run.dispatch(patient.modality)
    yield dispatched

    # Prep + imaging on the machine/technologist reserved by the dispatcher
    yield run.env.timeout(patient.prep_minutes + patient.scan_minutes)
    patient.imaging_end_time = run.env.now
    run.release_machine(patient)

    with run.radiologists.request() as req:
        yield req
        yield run.env.timeout(patient.report_minutes)
    patient.departure_time = run.env.now
    run.kpi.record_patient(patient)


def queue_monitor(run: DayRun) -> Generator[Any, Any, None]:
    """Sample queue lengths at fixed intervals during operating hours."""
    yield run.env.timeout(run.open_minute)
    while run.env.now <= run.close_minute:
        for modality in ModalityType:
            run.kpi.record_queue_length(run.day, run.env.now, modality, len(run.queues[modality]))
        yield run.env.timeout(QUEUE_SAMPLE_INTERVAL_MINUTES)


def make_radiologist_pool(env: simpy.Environment, capacity: int) -> simpy.Resource:
    """Radiologist reporting pool (FIFO)."""
    return simpy.Resource(env, capacity=max(1, capacity))
