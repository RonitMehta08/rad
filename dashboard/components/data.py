"""Cached data access for dashboard pages (models, simulations, datasets).

All heavy work is delegated to framework-free services in ``src`` and cached
with Streamlit so pages stay responsive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components import PROJECT_ROOT
from src.scheduler.policies import get_policy
from src.services.model_registry import ModelRegistry
from src.simulation.engine import RadiologySimulation
from src.simulation.entities import SimulationSettings
from src.simulation.scenarios import run_comparison, run_whatif

GENERATED_PATIENTS: Path = PROJECT_ROOT / "data" / "generated" / "patients.csv"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
TRAIN_COMMANDS = "See MANUAL_COMMANDS.md (generate data → preprocess → train)."


@st.cache_resource(show_spinner="Loading models…")
def get_registry() -> ModelRegistry:
    """Process-wide model registry."""
    return ModelRegistry()


@st.cache_data(show_spinner="Simulating department…")
def simulate_day(policy: str, tier: str, days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run the digital twin; returns (patients, queue samples, KPI summary)."""
    sim = RadiologySimulation(get_policy(policy), SimulationSettings.from_config(tier, days=days), seed=seed)
    kpi = sim.run()
    return kpi.to_dataframe(), pd.DataFrame(kpi.queue_samples), kpi.summary()


@st.cache_data(show_spinner="Comparing policies on identical patient streams…")
def compare_policies(policies: tuple[str, ...], tier: str, days: int, replications: int, seed: int) -> pd.DataFrame:
    """Policy comparison summary (mean + 95% CI)."""
    _, summary = run_comparison(list(policies), SimulationSettings.from_config(tier, days=days), replications, seed)
    return summary


@st.cache_data(show_spinner="Running what-if scenario…")
def whatif(scenario_json: str, policy: str, tier: str, days: int, replications: int, seed: int) -> pd.DataFrame:
    """Baseline vs scenario KPI table (scenario passed as JSON for cache hashing)."""
    settings = SimulationSettings.from_config(tier, days=days)
    return run_whatif(json.loads(scenario_json), policy, settings, replications, seed)


@st.cache_data(show_spinner="Loading generated dataset…")
def load_generated_patients() -> pd.DataFrame | None:
    """The synthetic training dataset, or None if not generated yet."""
    if not GENERATED_PATIENTS.exists():
        return None
    df = pd.read_csv(GENERATED_PATIENTS, parse_dates=["registration_time"])
    df["date"] = df["registration_time"].dt.date
    return df


def load_json(path: Path) -> Any | None:
    """Read a JSON file if present."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def require_models(registry: ModelRegistry, *names: str) -> bool:
    """Show an actionable warning and return False if any named model is missing."""
    status = registry.model_status()
    missing = [n for n in names if not status.get(n)]
    if missing:
        st.warning(f"Trained model(s) missing: {', '.join(missing)}. {TRAIN_COMMANDS}")
        return False
    return True
