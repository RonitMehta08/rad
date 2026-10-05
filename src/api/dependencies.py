"""Shared FastAPI dependencies (model registry, live department).

Owner: P5 (Dashboard & API)
"""

from __future__ import annotations

from fastapi import Request

from src.services.live_department import LiveDepartment
from src.services.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Model registry created at application startup."""
    return request.app.state.registry


def get_department(request: Request) -> LiveDepartment:
    """Live department created at application startup (lazily, so models load once)."""
    if getattr(request.app.state, "department", None) is None:
        registry: ModelRegistry = request.app.state.registry
        request.app.state.department = LiveDepartment(
            tier=request.app.state.tier,
            wait_time_predictor=registry.try_wait_time_predictor(),
        )
    return request.app.state.department
