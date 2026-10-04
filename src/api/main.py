"""FastAPI application entry point.

Owner: P5
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import predict, schedule, simulate, monitor

app = FastAPI(
    title="RadQueue AI API",
    description="API for intelligent radiology OPD scheduling and waiting-time prediction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(schedule.router)
app.include_router(simulate.router)
app.include_router(monitor.router)

@app.get("/")
async def root():
    return {"message": "Welcome to RadQueue AI API"}
