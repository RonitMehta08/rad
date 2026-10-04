"""Time-varying arrival generators for SimPy simulation.

Owner: P4
"""
import random
import numpy as np
import simpy
from typing import Any, Callable
from src.simulation.entities import SimPatient
from src.utils.constants import ModalityType, UrgencyLevel, VisitType

def generate_arrivals(
    env: simpy.Environment, 
    rate: float, 
    patient_handler: Callable[[SimPatient], None],
    n_patients: int = 100
):
    """Generate arrivals according to a Poisson process."""
    count = 0
    while count < n_patients:
        dt = random.expovariate(rate)
        yield env.timeout(dt)
        count += 1
        
        modality = random.choice(list(ModalityType))
        urgency = random.choices(list(UrgencyLevel), weights=[0.1, 0.2, 0.7])[0]
        visit = random.choices(list(VisitType), weights=[0.2, 0.6, 0.2])[0]
        
        patient = SimPatient(
            id=f"P{count}",
            arrival_time=env.now,
            modality=modality,
            urgency=urgency,
            visit_type=visit,
            attributes={}
        )
        patient_handler(patient)
