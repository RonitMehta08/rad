"""Scheduling endpoints.

Owner: P5
"""
from fastapi import APIRouter
from src.api.schemas import SchedulePatientRequest, SchedulePatientResponse

router = APIRouter(prefix="/schedule", tags=["Scheduling"])

@router.post("/assign", response_model=SchedulePatientResponse)
async def assign_patient(request: SchedulePatientRequest):
    # Dummy implementation
    return SchedulePatientResponse(
        patient_id=request.patient_id,
        assigned_machine_id=f"{request.modality.value}_room_1",
        scheduled_time=0.0,
        predicted_wait_minutes=15.0
    )
