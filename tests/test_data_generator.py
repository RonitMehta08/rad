"""Tests for the SimPy synthetic data generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.generator import RadiologyDepartmentGenerator, _load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "simulation_config.yaml"
N_DAYS = 14


@pytest.fixture(scope="module")
def generated() -> pd.DataFrame:
    return RadiologyDepartmentGenerator(_load_config(CONFIG), seed=7).generate(n_days=N_DAYS)


def test_should_be_deterministic_when_seed_is_fixed(generated: pd.DataFrame) -> None:
    again = RadiologyDepartmentGenerator(_load_config(CONFIG), seed=7).generate(n_days=N_DAYS)
    pd.testing.assert_frame_equal(generated, again)


def test_should_only_mark_scheduled_patients_as_noshows(generated: pd.DataFrame) -> None:
    noshows = generated[~generated["showed_up"].astype(bool)]
    assert len(noshows) > 0
    assert set(noshows["visit_type"]) == {"scheduled"}
    assert noshows["actual_wait_time_minutes"].isna().all()


def test_should_have_higher_noshow_rate_for_long_lead_times(generated: pd.DataFrame) -> None:
    scheduled = generated[generated["visit_type"] == "scheduled"]
    rate = 1 - scheduled.groupby(scheduled["appointment_lead_time_days"] >= 7)["showed_up"].mean()
    assert rate[True] > rate[False]


def test_should_record_imaging_after_registration(generated: pd.DataFrame) -> None:
    served = generated[generated["showed_up"].astype(bool)]
    assert (pd.to_datetime(served["imaging_start_time"]) >= pd.to_datetime(served["registration_time"])).all()
    assert (served["actual_wait_time_minutes"] >= 0).all()


def test_should_use_configured_machine_count_for_ultrasound(generated: pd.DataFrame) -> None:
    us = generated[(generated["modality"] == "ultrasound") & generated["showed_up"].astype(bool)]
    assert set(us["num_machines_available"]) == {2}  # tier_1 us_rooms
