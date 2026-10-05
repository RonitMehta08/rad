"""Prediction endpoints: wait time (with SHAP) and no-show risk.

Owner: P5 (Dashboard & API)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_registry
from src.api.schemas import (
    PredictNoShowRequest,
    PredictNoShowResponse,
    PredictWaitTimeRequest,
    PredictWaitTimeResponse,
)
from src.services.model_registry import ModelRegistry
from src.services.prediction_service import predict_noshow, predict_wait_time

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/wait-time", response_model=PredictWaitTimeResponse)
async def predict_wait_time_endpoint(
    request: PredictWaitTimeRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictWaitTimeResponse:
    """Predict a patient's wait (minutes) with an empirical 90% interval and SHAP explanation."""
    raw = {
        **request.patient.model_dump(mode="json", exclude={"patient_id"}),
        **request.context.model_dump(mode="json", exclude_none=True),
    }
    result = await run_in_threadpool(predict_wait_time, registry, raw, request.explain)
    return PredictWaitTimeResponse(patient_id=request.patient.patient_id, **result)


@router.post("/no-show", response_model=PredictNoShowResponse)
async def predict_noshow_endpoint(
    request: PredictNoShowRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictNoShowResponse:
    """No-show probability for a scheduled appointment and an overbooking recommendation."""
    raw = request.patient.model_dump(mode="json", exclude={"patient_id"})
    if request.timestamp is not None:
        raw["timestamp"] = request.timestamp.isoformat()
    result = await run_in_threadpool(predict_noshow, registry, raw)
    return PredictNoShowResponse(patient_id=request.patient.patient_id, **result)
