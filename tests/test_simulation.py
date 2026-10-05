"""Tests for the SimPy digital twin and scenario runner."""

from __future__ import annotations

import pytest

from src.scheduler.policies import get_policy
from src.simulation.arrival_patterns import generate_day_patients
from src.simulation.engine import RadiologySimulation
from src.simulation.entities import SimulationSettings
from src.simulation.scenarios import run_comparison, run_whatif
from src.utils.constants import ModalityType


@pytest.fixture(scope="module")
def settings() -> SimulationSettings:
    return SimulationSettings.from_config("tier_1", days=2)


def _summary(policy: str, settings: SimulationSettings, seed: int = 3) -> dict[str, float]:
    return RadiologySimulation(get_policy(policy), settings, seed=seed).run().summary()


def test_should_load_tier_resources_from_config(settings: SimulationSettings) -> None:
    assert settings.machines[ModalityType.MRI] == 1
    assert settings.machines[ModalityType.ULTRASOUND] == 2
    assert settings.staffing_at(9 * 60).technologists == 8


def test_should_reproduce_results_when_seed_is_fixed(settings: SimulationSettings) -> None:
    assert _summary("radqueue_ai", settings) == _summary("radqueue_ai", settings)


def test_should_give_every_policy_the_same_patient_stream(settings: SimulationSettings) -> None:
    base = [p.id for p in generate_day_patients(settings, 0, seed=3)]
    assert base == [p.id for p in generate_day_patients(settings, 0, seed=3)]
    assert _summary("fcfs", settings)["patients_served"] == _summary("radqueue_ai", settings)["patients_served"]


def test_should_serve_emergencies_faster_than_fcfs(settings: SimulationSettings) -> None:
    assert _summary("radqueue_ai", settings)["emergency_avg_wait"] < _summary("fcfs", settings)["emergency_avg_wait"]


def test_should_serve_more_patients_when_overbooking(settings: SimulationSettings) -> None:
    assert _summary("radqueue_noshow", settings)["patients_served"] > _summary("radqueue_ai", settings)["patients_served"]


def test_should_reduce_mri_wait_when_mri_machine_added(settings: SimulationSettings) -> None:
    table = run_whatif({"name": "add_1_mri", "machine_adjustments": {"mri": 1}}, "radqueue_ai", settings,
                       replications=2).set_index("kpi")
    assert table.loc["avg_wait", "scenario"] < table.loc["avg_wait", "baseline"]


def test_should_report_confidence_intervals_in_comparison(settings: SimulationSettings) -> None:
    _, summary = run_comparison(["fcfs", "priority"], settings, replications=2)
    assert {"avg_wait", "avg_wait_ci95"} <= set(summary.columns)
    assert len(summary) == 2
