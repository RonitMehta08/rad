"""FastAPI application entry point.

Owner: P5 (Dashboard & API)

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import monitor, predict, schedule, simulate
from src.services.model_registry import ModelRegistry
from src.utils.exceptions import (
    ConfigurationError,
    InvalidInputError,
    ModelNotAvailableError,
    ResourceNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_ORIGINS: list[str] = os.environ.get(
    "RADQUEUE_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501",
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the model registry and (lazily) the live department."""
    app.state.registry = ModelRegistry()
    app.state.tier = os.environ.get("RADQUEUE_TIER", "tier_1")
    app.state.department = None
    logger.info(f"RadQueue AI API starting; models available: {app.state.registry.model_status()}")
    yield


app = FastAPI(
    title="RadQueue AI API",
    description="Intelligent radiology OPD wait-time prediction and dynamic scheduling.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _error(status: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)})


@app.exception_handler(ModelNotAvailableError)
async def model_missing_handler(_: Request, exc: ModelNotAvailableError) -> JSONResponse:
    return _error(503, exc)


@app.exception_handler(ResourceNotFoundError)
async def not_found_handler(_: Request, exc: ResourceNotFoundError) -> JSONResponse:
    return _error(404, exc)


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(_: Request, exc: InvalidInputError) -> JSONResponse:
    return _error(422, exc)


@app.exception_handler(ConfigurationError)
async def config_error_handler(_: Request, exc: ConfigurationError) -> JSONResponse:
    logger.error(f"Configuration error: {exc}")
    return _error(500, exc)


app.include_router(predict.router)
app.include_router(schedule.router)
app.include_router(simulate.router)
app.include_router(monitor.router)


@app.get("/")
async def root() -> dict[str, str]:
    """API landing endpoint."""
    return {"message": "RadQueue AI API", "docs": "/docs"}
