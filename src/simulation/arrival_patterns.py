"""Time-varying arrival generator for the SimPy digital twin.

Generates one operating day of patients from a non-homogeneous Poisson process
(hourly multipliers x Indian calendar demand), with every stochastic quantity
pre-sampled from a seeded RNG. Re-using the same seed therefore gives every
scheduling policy the identical patient stream (common random numbers), which
is what makes policy comparisons fair.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §6.1, §8.1
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np

from src.data.generator import compute_noshow_probability
from src.data.indian_context import get_demand_multiplier, get_monsoon_emergency_multiplier
from src.simulation.entities import SimPatient, SimulationSettings
from src.utils.constants import OVERBOOKING_PROBABILITY_THRESHOLD, ModalityType, UrgencyLevel, VisitType

MINUTES_PER_HOUR: int = 60
WALK_IN_URGENT_PROBABILITY: float = 0.15
SCHEDULED_URGENT_PROBABILITY: float = 0.10
SCHEDULED_LEAD_TIMES_DAYS: list[int] = [1, 2, 3, 5, 7, 14, 21]
CONTRAST_TIME_MULTIPLIER: float = 1.3
PREP_TIME_MULTIPLIER: float = 1.5
PREVIOUS_NOSHOW_RATE: float = 1.5  # exponential rate for history counts (matches generator)
SEED_DAY_STRIDE: int = 10_007


def _sample_normal_positive(rng: np.random.Generator, params: dict[str, float], default_mean: float) -> float:
    """Sample N(mean, std) clamped to the configured minimum."""
    mean = params.get("mean_minutes", default_mean)
    std = params.get("std_minutes", mean / 2.5)
    return max(params.get("min_minutes", 1.0), float(rng.normal(mean, std)))


def _sample_scan_minutes(rng: np.random.Generator, params: dict[str, float]) -> float:
    """Sample a clamped log-normal scan duration."""
    value = float(rng.lognormal(params.get("mu", 2.0), params.get("sigma", 0.4)))
    return min(params.get("max_minutes", 120.0), max(params.get("min_minutes", 1.0), value))


def _choose(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Weighted categorical choice from a {label: weight} dict."""
    labels = list(weights)
    probs = np.array([weights[k] for k in labels], dtype=float)
    return labels[int(rng.choice(len(labels), p=probs / probs.sum()))]


def _visit_and_urgency(
    rng: np.random.Generator, settings: SimulationSettings, emergency_ratio: float,
) -> tuple[VisitType, UrgencyLevel]:
    """Sample visit type and urgency using the same rules as the data generator."""
    r = rng.random()
    if r < emergency_ratio:
        return VisitType.EMERGENCY, UrgencyLevel.EMERGENCY
    if r < emergency_ratio + settings.walk_in_ratio * (1 - emergency_ratio):
        urgent = rng.random() < WALK_IN_URGENT_PROBABILITY
        return VisitType.WALK_IN, UrgencyLevel.URGENT if urgent else UrgencyLevel.ROUTINE
    urgent = rng.random() < SCHEDULED_URGENT_PROBABILITY
    return VisitType.SCHEDULED, UrgencyLevel.URGENT if urgent else UrgencyLevel.ROUTINE


def _build_patient(
    rng: np.random.Generator,
    settings: SimulationSettings,
    patient_id: str,
    day: int,
    arrival: float,
    visit: VisitType,
    urgency: UrgencyLevel,
    modality: ModalityType | None = None,
) -> SimPatient:
    """Sample all attributes and service times for one patient."""
    dists = settings.patient_distributions
    modality = modality or ModalityType(_choose(rng, settings.modality_mix))
    mod = modality.value
    contrast = rng.random() < dists.get("contrast_probability", {}).get(mod, 0.0)
    prep = rng.random() < dists.get("prep_probability", {}).get(mod, 0.0)
    attrs: dict[str, Any] = {
        "modality": mod,
        "visit_type": visit.value,
        "urgency": urgency.value,
        "requires_contrast": contrast,
        "requires_prep": prep,
        "exam_complexity": _choose(rng, dists.get("exam_complexity_weights", {"simple": 1.0})),
        "distance_category": _choose(rng, dists.get("distance_category_weights", {"local": 1.0})),
        "age_group": _choose(rng, dists.get("age_group_weights", {"18-40": 1.0})),
        "previous_no_show_count": int(rng.exponential(1 / PREVIOUS_NOSHOW_RATE)),
        "appointment_lead_time_days": (
            int(rng.choice(SCHEDULED_LEAD_TIMES_DAYS)) if visit == VisitType.SCHEDULED else 0
        ),
    }
    scan_params = settings.service_times.get(mod, {})
    contrast_mult = CONTRAST_TIME_MULTIPLIER if contrast else 1.0
    prep_mult = PREP_TIME_MULTIPLIER if prep else 1.0
    prep_params = settings.process_times.get("technologist_prep", {})
    expected_prep = prep_params.get("mean_minutes", 4.0) * prep_mult
    return SimPatient(
        id=patient_id,
        day=day,
        arrival_time=arrival,
        modality=modality,
        urgency=urgency,
        visit_type=visit,
        registration_minutes=_sample_normal_positive(rng, settings.process_times.get("registration", {}), 5.0),
        prep_minutes=_sample_normal_positive(rng, prep_params, 4.0) * prep_mult,
        scan_minutes=_sample_scan_minutes(rng, scan_params) * contrast_mult,
        report_minutes=_sample_normal_positive(rng, settings.process_times.get("radiologist_reporting", {}), 12.0),
        estimated_duration=expected_prep + scan_params.get("median_minutes", 10.0) * contrast_mult,
        attributes=attrs,
    )


def generate_day_patients(
    settings: SimulationSettings,
    day: int,
    seed: int,
) -> list[SimPatient]:
    """Generate the full (pre-sampled) patient stream for one operating day.

    Scheduled patients get a no-show probability (shared generator model) and a
    pre-drawn ``will_show`` flag. For each high-risk appointment a wait-listed
    standby patient (``is_overbooked=True``) is also generated; the engine only
    admits them when the policy enables overbooking, so the base stream is
    identical across policies.

    Args:
        settings: Department settings (scenario already applied).
        day: Zero-based day index.
        seed: Replication seed.

    Returns:
        Patients sorted by arrival time (minutes since midnight).
    """
    rng = np.random.default_rng(seed * SEED_DAY_STRIDE + day)
    date = settings.start_date + timedelta(days=day)
    demand = get_demand_multiplier(date.date()) * settings.arrival_rate_multiplier
    emergency_ratio = settings.emergency_ratio * get_monsoon_emergency_multiplier(date.date())
    hours = max(1, settings.close_hour - settings.open_hour)
    base_rate = settings.daily_volume * demand / hours

    patients: list[SimPatient] = []
    for hour in range(settings.open_hour, settings.close_hour):
        lam = base_rate * settings.hourly_multipliers.get(hour, 1.0)
        n = int(rng.poisson(lam))
        minutes = np.sort(rng.uniform(0, MINUTES_PER_HOUR, size=n))
        for minute in minutes:
            arrival = hour * MINUTES_PER_HOUR + float(minute)
            visit, urgency = _visit_and_urgency(rng, settings, emergency_ratio)
            patient = _build_patient(rng, settings, f"D{day}-P{len(patients):04d}", day, arrival, visit, urgency)
            if visit == VisitType.SCHEDULED:
                p_noshow = compute_noshow_probability(
                    settings.noshow_rate, settings.noshow_config, patient.attributes,
                    date, patient.attributes["appointment_lead_time_days"],
                ) * settings.noshow_rate_multiplier
                patient.noshow_probability = p_noshow
                patient.will_show = bool(rng.random() >= p_noshow)
                patients.append(patient)
                if p_noshow >= OVERBOOKING_PROBABILITY_THRESHOLD:
                    standby = _build_patient(
                        rng, settings, f"{patient.id}-OB", day, arrival,
                        VisitType.SCHEDULED, UrgencyLevel.ROUTINE, modality=patient.modality,
                    )
                    standby.is_overbooked = True
                    patients.append(standby)
            else:
                patients.append(patient)
    return sorted(patients, key=lambda p: p.arrival_time)
