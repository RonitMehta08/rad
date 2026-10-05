"""Monitoring endpoints: health, model metadata, drift.

Owner: P5 (Dashboard & API)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from src.api.dependencies import get_registry
from src.services.model_registry import ModelRegistry
from src.services.monitoring_service import run_drift_report

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


@router.get("/health")
async def health_check(registry: ModelRegistry = Depends(get_registry)) -> dict[str, Any]:
    """Liveness plus which trained models are available."""
    models = registry.model_status()
    return {"status": "ok" if all(models.values()) else "degraded", "models": models}


@router.get("/models")
async def model_info(registry: ModelRegistry = Depends(get_registry)) -> dict[str, Any]:
    """Validation/test metrics and parameters of every trained model."""
    return registry.model_metadata()


@router.get("/drift")
async def drift(registry: ModelRegistry = Depends(get_registry)) -> dict[str, Any]:
    """PSI/KS drift of the latest window (test split) vs training, plus ADWIN/CUSUM on errors."""
    return await run_in_threadpool(run_drift_report, registry)
