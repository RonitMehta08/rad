"""SimPy entities for the simulation.

Owner: P4 (Simulation Engineer)
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import simpy
from src.utils.constants import ModalityType, UrgencyLevel, VisitType
from src.scheduler.models import PatientState

@dataclass
class SimPatient:
    id: str
    arrival_time: float
    modality: ModalityType
    urgency: UrgencyLevel
    visit_type: VisitType
    attributes: dict[str, Any]
    
    # Timestamps
    registration_start: Optional[float] = None
    registration_end: Optional[float] = None
    imaging_start: Optional[float] = None
    imaging_end: Optional[float] = None
    reporting_start: Optional[float] = None
    reporting_end: Optional[float] = None
    departure_time: Optional[float] = None
    
    def to_patient_state(self, current_time: float) -> PatientState:
        return PatientState(
            patient_id=self.id,
            modality=self.modality,
            urgency=self.urgency,
            visit_type=self.visit_type,
            arrival_time_minutes=self.arrival_time,
            current_wait_minutes=current_time - self.arrival_time,
            status="waiting",
            features=self.attributes
        )
