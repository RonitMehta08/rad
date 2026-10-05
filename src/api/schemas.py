"""Pydantic request/response schemas for the FastAPI backend.

Owner: P5 (Dashboard & API)
Reference: MASTER_PROMPT §5.3, §19.3, §19.5 (validate all external input)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.utils.constants import ModalityType, UrgencyLevel, VisitType

MAX_QUEUE_LENGTH: int = 500
MAX_LEAD_TIME_DAYS: int = 365
MAX_SIM_DAYS: int = 30
MAX_REPLICATIONS: int = 30

PolicyName = Literal["fcfs", "priority", "sjf", "wave", "radqueue_ai", "radqueue_noshow"]
TierName = Literal["tier_1", "tier_2", "tier_3"]


class PatientInput(BaseModel):
    """Raw patient description shared by prediction and scheduling endpoints."""

    patient_id: str = Field(..., min_length=1, max_length=64)
    modality: ModalityType
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    visit_type: VisitType = VisitType.WALK_IN
    age_group: Literal["0-18", "18-40", "40-60", "60+"] = "18-40"
    gender: Literal["M", "F", "Other"] = "M"
    insurance_type: Literal["government", "private", "self_pay"] = "government"
    exam_complexity: Literal["simple", "moderate", "complex"] = "simple"
    distance_category: Literal["local", "city", "outstation"] = "local"
    requires_contrast: bool = False
    requires_prep: bool = False
    previous_no_show_count: int = Field(0, ge=0, le=50)
    previous_appointment_count: int = Field(0, ge=0, le=500)
    appointment_lead_time_days: int = Field(0, ge=0, le=MAX_LEAD_TIME_DAYS)


class DepartmentContext(BaseModel):
    """Optional live department context; omitted fields use typical (training-median) values."""

    timestamp: datetime | None = None
    current_queue_length_same_modality: int | None = Field(None, ge=0, le=MAX_QUEUE_LENGTH)
    current_queue_length_total: int | None = Field(None, ge=0, le=MAX_QUEUE_LENGTH)
    emergency_patients_in_queue: int | None = Field(None, ge=0, le=MAX_QUEUE_LENGTH)
    patients_in_service_count: int | None = Field(None, ge=0, le=100)
    num_machines_available: int | None = Field(None, ge=0, le=20)
    num_technologists_on_duty: int | None = Field(None, ge=0, le=100)
    num_radiologists_on_duty: int | None = Field(None, ge=0, le=100)
    rolling_avg_wait_30min: float | None = Field(None, ge=0)
    rolling_avg_wait_1hr: float | None = Field(None, ge=0)
    arrival_rate_last_15min: float | None = Field(None, ge=0)


class PredictWaitTimeRequest(BaseModel):
    """Wait-time prediction request."""

    patient: PatientInput
    context: DepartmentContext = Field(default_factory=DepartmentContext)
    explain: bool = True


class ShapContribution(BaseModel):
    feature: str
    display_name: str
    feature_value: float | str
    shap_value: float
    direction: str


class Explanation(BaseModel):
    base_value: float
    contributions: list[ShapContribution]
    clinician_text: str
    explained_model: str


class PredictWaitTimeResponse(BaseModel):
    patient_id: str
    predicted_wait_minutes: float
    lower_bound_minutes: float
    upper_bound_minutes: float
    confidence_level: float
    interval_method: str
    model_name: str
    explanation: Explanation | None = None


class PredictNoShowRequest(BaseModel):
    patient: PatientInput
    timestamp: datetime | None = None


class PredictNoShowResponse(BaseModel):
    patient_id: str
    no_show_probability: float
    should_overbook: bool
    overbooking_blocked_by_safety: bool
    reason: str
    decision_threshold: float


class RegisterPatientRequest(PatientInput):
    estimated_duration_minutes: float | None = Field(None, gt=0, le=240)


class AdvanceClockRequest(BaseModel):
    minutes: float = Field(..., ge=0, le=24 * 60)


class DayAheadPatient(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=64)
    modality: ModalityType
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    arrival_time_minutes: float = Field(0.0, ge=0, le=12 * 60, description="Requested time, minutes after opening")
    estimated_duration_minutes: float | None = Field(None, gt=0, le=240)


class DayAheadRequest(BaseModel):
    patients: list[DayAheadPatient] = Field(..., min_length=1, max_length=200)
    tier: TierName = "tier_1"


class RunSimulationRequest(BaseModel):
    policy: PolicyName = "radqueue_ai"
    tier: TierName = "tier_1"
    days: int = Field(3, ge=1, le=MAX_SIM_DAYS)
    seed: int = 42


class CompareSimulationRequest(BaseModel):
    policies: list[PolicyName] = Field(
        default_factory=lambda: ["fcfs", "priority", "sjf", "wave", "radqueue_ai", "radqueue_noshow"],
    )
    tier: TierName = "tier_1"
    days: int = Field(3, ge=1, le=MAX_SIM_DAYS)
    replications: int = Field(5, ge=1, le=MAX_REPLICATIONS)
    seed: int = 42


class WhatIfRequest(BaseModel):
    scenario_name: str | None = Field(None, description="Preset from simulation_config.yaml")
    machine_adjustments: dict[ModalityType, int] = Field(default_factory=dict)
    staff_adjustments: dict[Literal["technologists", "radiologists"], int] = Field(default_factory=dict)
    arrival_rate_multiplier: float = Field(1.0, gt=0, le=5)
    noshow_rate_multiplier: float = Field(1.0, ge=0, le=5)
    emergency_buffer_fraction: float | None = Field(None, ge=0, le=0.5)
    policy: PolicyName = "radqueue_ai"
    tier: TierName = "tier_1"
    days: int = Field(3, ge=1, le=MAX_SIM_DAYS)
    replications: int = Field(5, ge=1, le=MAX_REPLICATIONS)
    seed: int = 42


class GenericResponse(BaseModel):
    """Envelope for endpoints returning tabular/structured results."""

    data: Any
