"""Prediction endpoints.

Owner: P5
"""
from fastapi import APIRouter
from src.api.schemas import PredictWaitTimeRequest, PredictWaitTimeResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/wait-time", response_model=PredictWaitTimeResponse)
async def predict_wait_time(request: PredictWaitTimeRequest):
    # Dummy implementation since we don't have the full ML models loaded here yet
    # In reality, this would load models/wait_time/ensemble_best.pkl and run inference
    
    # Simple dummy logic
    wait_time = 45.0
    if request.urgency.value == "emergency":
        wait_time = 5.0
    elif request.modality.value == "mri":
        wait_time = 90.0
        
    return PredictWaitTimeResponse(
        patient_id=request.patient_id,
        predicted_wait_minutes=wait_time,
        confidence_interval=(wait_time * 0.8, wait_time * 1.2),
        shap_summary=f"Key drivers: Modality {request.modality.value} (+{wait_time/2} min), Urgency {request.urgency.value}."
    )
