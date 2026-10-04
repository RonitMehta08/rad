"""Simulation endpoints.

Owner: P5
"""
from fastapi import APIRouter
from src.api.schemas import RunSimulationRequest, RunSimulationResponse

router = APIRouter(prefix="/simulate", tags=["Simulation"])

@router.post("/run", response_model=RunSimulationResponse)
async def run_simulation(request: RunSimulationRequest):
    # Dummy implementation 
    # In reality, this would invoke src.simulation.scenarios
    return RunSimulationResponse(
        policy=request.policy,
        avg_wait_minutes=35.0 if request.policy != "RadQueue AI" else 20.0,
        max_wait_minutes=150.0 if request.policy != "RadQueue AI" else 45.0,
        patients_served=200
    )
