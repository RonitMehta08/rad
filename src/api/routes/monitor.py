"""Monitoring endpoints.

Owner: P5
"""
from fastapi import APIRouter

router = APIRouter(prefix="/monitor", tags=["Monitoring"])

@router.get("/health")
async def health_check():
    return {"status": "ok", "drift_detected": False}
