"""Simulation endpoints: single run, policy comparison, what-if scenarios.

Owner: P5 (Dashboard & API) on top of P4's digital twin
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import CompareSimulationRequest, RunSimulationRequest, WhatIfRequest
from src.simulation.entities import SimulationSettings
from src.simulation.scenarios import load_whatif_scenarios, run_comparison, run_single, run_whatif
from src.utils.exceptions import ResourceNotFoundError

router = APIRouter(prefix="/simulate", tags=["Simulation"])


@router.post("/run")
async def run_simulation(request: RunSimulationRequest) -> dict[str, Any]:
    """Run one seeded simulation and return its KPI summary."""
    settings = SimulationSettings.from_config(request.tier, days=request.days)
    return await run_in_threadpool(run_single, request.policy, settings, request.seed)


@router.post("/compare")
async def compare_policies(request: CompareSimulationRequest) -> dict[str, Any]:
    """Compare policies on identical patient streams (mean and 95% CI per KPI)."""
    settings = SimulationSettings.from_config(request.tier, days=request.days)
    _, summary = await run_in_threadpool(
        run_comparison, list(request.policies), settings, request.replications, request.seed,
    )
    return {"summary": summary.round(4).to_dict(orient="records")}


@router.get("/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    """What-if presets from simulation_config.yaml."""
    return load_whatif_scenarios()


def _resolve_scenario(request: WhatIfRequest) -> dict[str, Any]:
    if request.scenario_name:
        presets = {s["name"]: s for s in load_whatif_scenarios()}
        if request.scenario_name not in presets:
            raise ResourceNotFoundError(
                f"Unknown scenario '{request.scenario_name}'. Available: {sorted(presets)}"
            )
        return presets[request.scenario_name]
    scenario: dict[str, Any] = {
        "name": "custom",
        "machine_adjustments": {m.value: n for m, n in request.machine_adjustments.items()},
        "staff_adjustments": dict(request.staff_adjustments),
        "arrival_rate_multiplier": request.arrival_rate_multiplier,
        "noshow_rate_multiplier": request.noshow_rate_multiplier,
    }
    if request.emergency_buffer_fraction is not None:
        scenario["emergency_buffer_fraction"] = request.emergency_buffer_fraction
    return scenario


@router.post("/whatif")
async def whatif(request: WhatIfRequest) -> dict[str, Any]:
    """Baseline vs scenario KPIs (preset name or custom adjustments)."""
    scenario = _resolve_scenario(request)
    settings = SimulationSettings.from_config(request.tier, days=request.days)
    table = await run_in_threadpool(
        run_whatif, scenario, request.policy, settings, request.replications, request.seed,
    )
    return {"scenario": scenario, "comparison": table.round(4).to_dict(orient="records")}
