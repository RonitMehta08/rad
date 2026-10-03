"""Domain models and entity definitions for the radiology scheduling system.

Owner: P3 (Scheduling Engineer)
Reference: MASTER_PROMPT §5.3, §7.3; config/scheduler_config.yaml
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from src.utils.constants import (
    ModalityType,
    UrgencyLevel,
    VisitType,
    SERVICE_TIME_PARAMS,
)


class PatientState(BaseModel):
    """Represents a patient's state in the radiology OPD pipeline."""

    patient_id: str
    modality: ModalityType
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    visit_type: VisitType = VisitType.WALK_IN
    arrival_time_minutes: float = 0.0
    registration_time_minutes: float | None = None
    current_wait_minutes: float = 0.0
    scheduled_slot: int | None = None
    is_preemptable: bool | None = None
    status: str = "waiting"  # "waiting", "in_prep", "in_scan", "completed", "noshow", "preempted"
    estimated_duration_minutes: float | None = None
    features: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Set default estimated duration and preemption eligibility if not provided."""
        if self.estimated_duration_minutes is None:
            self.estimated_duration_minutes = SERVICE_TIME_PARAMS[self.modality]["median_minutes"]
        if self.is_preemptable is None:
            # Emergencies are non-preemptable by default, routine/urgent are preemptable
            self.is_preemptable = (self.urgency != UrgencyLevel.EMERGENCY)


class Resource(BaseModel):
    """Represents an imaging machine or clinical unit."""

    resource_id: str
    modality: ModalityType
    name: str = ""
    is_available: bool = True
    utilization_rate: float = 0.0
    current_patient_id: str | None = None
    busy_until_minute: float = 0.0
    total_busy_minutes: float = 0.0


class Assignment(BaseModel):
    """Result of assigning a patient to a resource/time slot."""

    patient_id: str
    resource_id: str
    slot_index: int | None = None
    start_time_minutes: float = 0.0
    estimated_duration_minutes: float = 0.0
    predicted_wait_minutes: float = 0.0
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    score: float = 0.0


class DepartmentState(BaseModel):
    """Current operational state of the radiology department."""

    current_time_minutes: float = 0.0
    avg_wait_minutes: float = 0.0
    queue_length: dict[ModalityType, int] = Field(
        default_factory=lambda: {m: 0 for m in ModalityType}
    )
    waiting_patients: list[PatientState] = Field(default_factory=list)
    active_patients: list[PatientState] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)
    active_assignments: list[Assignment] = Field(default_factory=list)
