"""Pydantic schemas for FastAPI endpoints.

Owner: P5
"""
from pydantic import BaseModel, Field
from typing import Any, Optional
from src.utils.constants import ModalityType, UrgencyLevel, VisitType

class PredictWaitTimeRequest(BaseModel):
    patient_id: str
    age_group: str = "18-40"
    gender: str = "M"
    visit_type: VisitType = VisitType.WALK_IN
    insurance_type: str = "government"
    modality: ModalityType
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    requires_contrast: bool = False
    requires_prep: bool = False
    
    # Context features that might be overridden for what-if
    current_queue_length: Optional[int] = None
    num_machines_available: Optional[int] = None

class PredictWaitTimeResponse(BaseModel):
    patient_id: str
    predicted_wait_minutes: float
    confidence_interval: tuple[float, float]
    shap_summary: str

class SchedulePatientRequest(BaseModel):
    patient_id: str
    modality: ModalityType
    urgency: UrgencyLevel
    
class SchedulePatientResponse(BaseModel):
    patient_id: str
    assigned_machine_id: str
    scheduled_time: float
    predicted_wait_minutes: float

class RunSimulationRequest(BaseModel):
    policy: str
    days: int = 1
    
class RunSimulationResponse(BaseModel):
    policy: str
    avg_wait_minutes: float
    max_wait_minutes: float
    patients_served: int
