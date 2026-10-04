"""Configuration loader for the dynamic scheduler.

Owner: P3 (Scheduling Engineer)
Reference: config/scheduler_config.yaml; MASTER_PROMPT §7.3
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scheduler_config.yaml"


class PriorityWeightsConfig(BaseModel):
    emergency: int = 10
    urgent: int = 5
    routine: int = 1


class ObjectiveWeightsConfig(BaseModel):
    avg_wait_time: float = 0.40
    max_wait_time: float = 0.25
    wait_time_std_dev: float = 0.15
    utilization: float = 0.20


class MilpConfig(BaseModel):
    solver: str = "PULP_CBC_CMD"
    time_limit_seconds: int = 120
    mip_gap: float = 0.05
    slot_duration_minutes: int = 15
    planning_horizon_hours: int = 12
    emergency_buffer_fraction: float = 0.10
    objective_weights: ObjectiveWeightsConfig = Field(default_factory=ObjectiveWeightsConfig)


class ConstraintsConfig(BaseModel):
    max_wait_routine_minutes: int = 90
    max_wait_urgent_minutes: int = 30
    max_wait_emergency_minutes: int = 10
    min_utilization_target: float = 0.60
    max_overbooking_fraction: float = 0.15


class ReschedulingConfig(BaseModel):
    preemption_enabled: bool = True
    preempt_only_routine: bool = True
    noshow_detection_window_minutes: int = 15
    noshow_pullforward_enabled: bool = True
    equipment_failure_redistribution: bool = True
    surge_detection_sigma: float = 2.0
    surge_overflow_protocol: bool = True


class PolicyItemConfig(BaseModel):
    name: str
    description: str
    wave_interval_minutes: int | None = None


class WaveSchedulingConfig(BaseModel):
    interval_minutes: int = 30
    max_patients_per_wave: int = 10
    modality_balance: bool = True


class SchedulerConfig(BaseModel):
    """Complete typed scheduler configuration."""

    priority_weights: PriorityWeightsConfig = Field(default_factory=PriorityWeightsConfig)
    fairness_penalty_rate: float = 0.1
    starvation_threshold_minutes: int = 60
    milp: MilpConfig = Field(default_factory=MilpConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    rescheduling: ReschedulingConfig = Field(default_factory=ReschedulingConfig)
    policies: list[PolicyItemConfig] = Field(default_factory=list)
    wave_scheduling: WaveSchedulingConfig = Field(default_factory=WaveSchedulingConfig)


def load_scheduler_config(config_path: Path | str | None = None) -> SchedulerConfig:
    """Load and parse scheduler configuration YAML.

    Args:
        config_path: Path to scheduler_config.yaml. If None, uses default project path.

    Returns:
        Validated SchedulerConfig object.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        # Fallback check
        alt_path = Path("config/scheduler_config.yaml")
        if alt_path.is_file():
            path = alt_path
        else:
            logger.warning(f"Scheduler config not found at {path}, using defaults")
            return SchedulerConfig()

    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    logger.info(f"Loaded scheduler configuration from {path}")
    return SchedulerConfig(**raw_data)
