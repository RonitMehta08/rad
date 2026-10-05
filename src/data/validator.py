"""Pydantic schemas for data validation across the entire project.

These schemas define the shared data contracts. Every team member must
conform to these when producing or consuming data.

Owner: P1 (Data & Config Lead)
Consumers: P2 (training data), P3 (scheduler input), P4 (simulation entities), P5 (API schemas)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils.constants import (
    AgeGroup,
    DistanceCategory,
    ExamComplexity,
    HospitalTier,
    InsuranceType,
    ModalityType,
    ShiftType,
    UrgencyLevel,
    VisitType,
    WeatherCategory,
)

# ---------------------------------------------------------------------------
# Core Patient Record — the main data contract
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    """A single patient's complete record in the radiology department.

    This is the primary data structure flowing through the entire system:
    generator → preprocessor → model → scheduler → dashboard.

    All 35+ features from MASTER_PROMPT §5.3 are captured here.
    """

    # -- Identity --
    patient_id: str = Field(..., description="Unique patient identifier (e.g., 'P-00001')")

    # -- Patient Features --
    age_group: AgeGroup
    gender: str = Field(..., pattern=r"^(M|F|Other)$")
    visit_type: VisitType
    insurance_type: InsuranceType
    previous_no_show_count: int = Field(ge=0)

    # -- Appointment Features --
    modality: ModalityType
    exam_complexity: ExamComplexity
    requires_contrast: bool = Field(default=False, description="CT/MRI contrast injection")
    requires_prep: bool = Field(default=False, description="Fasting for US, bowel prep, etc.")
    appointment_lead_time_days: int = Field(ge=0, description="Days between booking and visit; 0 for walk-ins")

    # -- Urgency --
    urgency: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)

    # -- Timestamps --
    registration_time: datetime
    queue_entry_time: datetime | None = None
    prep_start_time: datetime | None = None
    imaging_start_time: datetime | None = None
    imaging_end_time: datetime | None = None
    reporting_start_time: datetime | None = None
    reporting_end_time: datetime | None = None
    departure_time: datetime | None = None

    # -- Queue State Features (snapshot at registration) --
    current_queue_length_total: int = Field(ge=0)
    current_queue_length_same_modality: int = Field(ge=0)
    patients_in_service_count: int = Field(ge=0)
    avg_service_time_last_5_patients: float = Field(ge=0)
    time_since_last_patient_served_minutes: float = Field(ge=0)
    emergency_patients_in_queue: int = Field(ge=0)

    # -- Resource Features (snapshot at registration) --
    num_machines_available: int = Field(ge=0, description="Available machines for this modality")
    num_technologists_on_duty: int = Field(ge=1)
    num_radiologists_on_duty: int = Field(ge=1)
    equipment_under_maintenance: bool = False
    shift_type: ShiftType

    # -- Temporal Features --
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="0=Monday, 6=Sunday")
    is_weekend: bool
    is_holiday: bool
    is_monday: bool
    minutes_since_department_opened: float = Field(ge=0)

    # -- Indian Context --
    weather_category: WeatherCategory = WeatherCategory.SUMMER
    distance_category: DistanceCategory = DistanceCategory.LOCAL
    is_repeat_patient: bool = False
    previous_appointment_count: int = Field(ge=0, default=0)

    # -- Computed Target (populated after the fact) --
    actual_wait_time_minutes: float | None = Field(
        default=None, ge=0,
        description="Ground truth wait time from registration to imaging start"
    )
    showed_up: bool | None = Field(
        default=None,
        description="Whether the patient actually showed up (for no-show model)"
    )

    @field_validator("actual_wait_time_minutes", mode="before")
    @classmethod
    def round_wait_time(cls, v: float | None) -> float | None:
        """Round wait time to 1 decimal place for consistency."""
        if v is not None:
            return round(v, 1)
        return v


# ---------------------------------------------------------------------------
# Derived / Engineered Features (output of preprocessor)
# ---------------------------------------------------------------------------

class EngineeredFeatures(BaseModel):
    """Feature-engineered representation ready for model input.

    Produced by the preprocessor (P1), consumed by the trainer/predictor (P2).
    """

    # Cyclical time encoding
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float

    # Rolling statistics
    rolling_avg_wait_1hr: float = Field(ge=0)
    rolling_avg_wait_30min: float = Field(ge=0)
    arrival_rate_last_15min: float = Field(ge=0, description="patients per minute")
    utilization_rate: float = Field(ge=0, le=1, description="patients_in_service / machines")

    # Lag features
    wait_time_last_served_patient: float = Field(ge=0)
    wait_time_last_3_avg: float = Field(ge=0)
    queue_length_change_last_15min: int

    # Interaction features
    congestion_indicator: float  # queue_length * utilization_rate
    modality_hour_interaction: str  # e.g., "mri_10" for MRI at 10 AM

    # Estimated remaining service for in-progress patients
    estimated_remaining_service_minutes: float = Field(ge=0)


# ---------------------------------------------------------------------------
# Scheduler Data Contracts
# ---------------------------------------------------------------------------

class Assignment(BaseModel):
    """Output of the scheduler — a patient assigned to a resource."""

    patient_id: str
    resource_id: str
    modality: ModalityType
    assigned_time: datetime
    predicted_wait_minutes: float = Field(ge=0)
    priority_score: float
    explanation: str = Field(default="", description="Human-readable scheduling rationale")


class DepartmentState(BaseModel):
    """Snapshot of the department's current operational state.

    Used by the scheduler to make assignment decisions.
    """

    timestamp: datetime
    queue_lengths: dict[ModalityType, int]
    machines_available: dict[ModalityType, int]
    machines_total: dict[ModalityType, int]
    technologists_on_duty: int
    radiologists_on_duty: int
    avg_wait_time_minutes: float = Field(ge=0)
    emergency_count_in_queue: int = Field(ge=0)
    patients_in_service: dict[ModalityType, int]


# ---------------------------------------------------------------------------
# Simulation Contracts
# ---------------------------------------------------------------------------

class SimulationKPIs(BaseModel):
    """KPI results from a single simulation run."""

    policy_name: str
    avg_wait_time_minutes: float
    max_wait_time_minutes: float
    median_wait_time_minutes: float
    std_wait_time_minutes: float
    p90_wait_time_minutes: float
    avg_utilization_rate: float
    throughput_patients_per_day: float
    emergency_response_time_minutes: float
    starvation_count: int = Field(ge=0, description="Patients who waited > 90 min")
    total_patients_served: int
    per_modality_avg_wait: dict[str, float] = Field(default_factory=dict)


class WhatIfScenario(BaseModel):
    """Configuration for a what-if simulation scenario."""

    scenario_name: str
    machine_adjustments: dict[ModalityType, int] = Field(
        default_factory=dict,
        description="Delta in machine count per modality (e.g., {MRI: +1})"
    )
    staff_adjustments: dict[str, int] = Field(
        default_factory=dict,
        description="Delta in staff count (e.g., {'technologists': +2})"
    )
    scheduling_policy: str = "radqueue_ai"
    arrival_rate_multiplier: float = Field(default=1.0, ge=0.1, le=3.0)
    emergency_buffer_fraction: float = Field(default=0.10, ge=0.0, le=0.5)


class ScenarioComparisonReport(BaseModel):
    """Side-by-side comparison of baseline vs. scenario KPIs."""

    baseline: SimulationKPIs
    scenario: SimulationKPIs
    scenario_config: WhatIfScenario
    improvements: dict[str, float] = Field(
        default_factory=dict,
        description="Percentage improvement per KPI (positive = better)"
    )


# ---------------------------------------------------------------------------
# SHAP / Explainability Contracts
# ---------------------------------------------------------------------------

class ShapExplanation(BaseModel):
    """SHAP explanation for a single prediction."""

    patient_id: str
    predicted_wait_minutes: float
    base_value: float
    feature_contributions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {feature, value, shap_value, direction} dicts"
    )
    clinician_text: str = Field(
        default="",
        description="Human-readable explanation for clinicians"
    )


# ---------------------------------------------------------------------------
# Hospital Configuration Contract
# ---------------------------------------------------------------------------

class HospitalConfig(BaseModel):
    """Hospital profile configuration (loaded from YAML)."""

    tier: HospitalTier
    name: str = "Default Hospital"
    daily_patient_volume: int = Field(ge=10)
    walk_in_ratio: float = Field(ge=0.0, le=1.0)

    # Resources per modality
    xray_rooms: int = Field(ge=1)
    ct_scanners: int = Field(ge=1)
    mri_machines: int = Field(ge=1)
    us_rooms: int = Field(ge=1)

    # Staff
    technologists: int = Field(ge=1)
    radiologists: int = Field(ge=1)
    registration_desks: int = Field(ge=1, default=2)

    # Operational
    emergency_ratio: float = Field(ge=0.0, le=0.3, default=0.10)
    noshow_rate: float = Field(ge=0.0, le=0.5, default=0.20)
