"""SimPy processes for patient flow.

Owner: P4
"""
import simpy
from typing import Any
from src.simulation.entities import SimPatient
from src.utils.constants import SERVICE_TIME_PARAMS

def patient_process(env: simpy.Environment, patient: SimPatient, resources: dict[str, Any], kpi: Any):
    """Core process simulating patient going through the department."""
    # 1. Registration
    patient.registration_start = env.now
    with resources["registration"].request() as req:
        yield req
        yield env.timeout(5.0)  # avg registration time
    patient.registration_end = env.now
    
    # Wait for scheduling / routing (handled externally by engine via events, or simplified here)
    # Actually, if we're evaluating scheduling policies, we need to interact with the policy.
    # We yield an event that the engine will trigger when this patient is assigned to a resource.
    
    if not hasattr(patient, 'assigned_event'):
        patient.assigned_event = env.event()
    
    yield patient.assigned_event
    
    # 2. Imaging
    patient.imaging_start = env.now
    modality_key = f"{patient.modality.value}_rooms"
    
    # Assume the engine already requested the specific machine.
    yield patient.req
    
    duration = SERVICE_TIME_PARAMS[patient.modality]["median_minutes"]
    yield env.timeout(duration)
    
    patient.req.resource.release(patient.req)
    patient.imaging_end = env.now
    
    # 3. Reporting
    patient.reporting_start = env.now
    with resources["radiologists"].request() as req:
        yield req
        yield env.timeout(12.0)
    patient.reporting_end = env.now
    patient.departure_time = env.now
    
    kpi.record_patient(patient)
    
    # Notify engine that machine is free
    if hasattr(patient, 'finished_event'):
        patient.finished_event.succeed()
