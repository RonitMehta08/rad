"""SimPy-based synthetic radiology OPD data generator.

Generates 50K+ patient records over 6 months, calibrated to Indian hospital
parameters from published studies.

Owner: P1 (Data & Config Lead)
Consumers: P2 (training data), P4 (simulation baseline)
Reference: MASTER_PROMPT §6.1, §6.2, §8.1

Usage:
    python -m src.data.generator --config config/simulation_config.yaml \\
        --output data/generated/ --days 180
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simpy
import yaml

from src.data.indian_context import (
    generate_holiday_features,
    get_demand_multiplier,
    get_monsoon_emergency_multiplier,
    get_weather_category,
)
from src.utils.constants import (
    ShiftType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# No-show model parameters not covered by simulation_config.yaml
NOSHOW_HISTORY_EFFECT_PER_MISS: float = 0.30  # +30% relative risk per previous no-show
NOSHOW_DISTANCE_EFFECT: dict[str, float] = {"local": 0.9, "city": 1.0, "outstation": 1.4}
MAX_NOSHOW_PROBABILITY: float = 0.90
MODALITY_RESOURCE_KEYS: dict[str, str] = {
    "xray": "xray_rooms", "ct": "ct_scanners", "mri": "mri_machines", "ultrasound": "us_rooms",
}

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load simulation + hospital profile configuration.

    Args:
        config_path: Path to simulation_config.yaml.

    Returns:
        Merged configuration dictionary.
    """
    with open(config_path, encoding="utf-8") as f:
        sim_config = yaml.safe_load(f)

    hospital_path = config_path.parent / "hospital_profiles.yaml"
    with open(hospital_path, encoding="utf-8") as f:
        hospital_config = yaml.safe_load(f)

    return {"simulation": sim_config, "hospital": hospital_config}


# ---------------------------------------------------------------------------
# Random Samplers
# ---------------------------------------------------------------------------


def _sample_lognormal(mu: float, sigma: float, min_val: float = 0.0, max_val: float = float("inf")) -> float:
    """Sample from a clamped log-normal distribution."""
    value = random.lognormvariate(mu, sigma)
    return max(min_val, min(value, max_val))


def _sample_normal_positive(mean: float, std: float, min_val: float = 0.5) -> float:
    """Sample from a normal distribution, clamped to be positive."""
    return max(min_val, random.gauss(mean, std))


def _weighted_choice(options: list[str], weights: list[float]) -> str:
    """Weighted random choice."""
    return random.choices(options, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# No-Show Model (shared with the SimPy digital twin)
# ---------------------------------------------------------------------------


def compute_noshow_probability(
    base_rate: float,
    noshow_cfg: dict[str, Any],
    attrs: dict[str, Any],
    day_dt: datetime,
    lead_time: int,
) -> float:
    """No-show probability for a scheduled appointment.

    p = base_rate x lead_time_effect x monsoon x monday x history x distance,
    clipped to MAX_NOSHOW_PROBABILITY. Multipliers come from the ``noshow``
    section of simulation_config.yaml.

    Args:
        base_rate: Hospital-tier no-show rate.
        noshow_cfg: ``noshow`` section of the simulation config.
        attrs: Patient attributes (previous_no_show_count, distance_category).
        day_dt: Appointment date.
        lead_time: Days between booking and appointment.

    Returns:
        Probability in [0, MAX_NOSHOW_PROBABILITY].
    """
    lead_effects = {int(k): float(v) for k, v in noshow_cfg.get("lead_time_effect", {}).items()}
    eligible = [k for k in lead_effects if k <= lead_time]
    mult = lead_effects[max(eligible)] if eligible else 1.0
    if get_weather_category(day_dt.date() if isinstance(day_dt, datetime) else day_dt).value == "monsoon":
        mult *= noshow_cfg.get("monsoon_multiplier", 1.0)
    if day_dt.weekday() == 0:
        mult *= noshow_cfg.get("monday_multiplier", 1.0)
    mult *= 1.0 + NOSHOW_HISTORY_EFFECT_PER_MISS * attrs.get("previous_no_show_count", 0)
    mult *= NOSHOW_DISTANCE_EFFECT.get(attrs.get("distance_category", "local"), 1.0)
    return min(MAX_NOSHOW_PROBABILITY, base_rate * mult)


# ---------------------------------------------------------------------------
# Arrival Rate Generator
# ---------------------------------------------------------------------------


def _hourly_arrival_rate(
    hour: int,
    base_rate_per_hour: float,
    hourly_multipliers: dict[str, float],
) -> float:
    """Compute Poisson arrival rate for a given hour.

    Args:
        hour: Hour of day (8-19).
        base_rate_per_hour: Average patients per hour.
        hourly_multipliers: Hour-to-multiplier mapping from config.

    Returns:
        Lambda for the Poisson process at this hour.
    """
    mult = hourly_multipliers.get(str(hour), 1.0)
    return base_rate_per_hour * mult


# ---------------------------------------------------------------------------
# Core Simulation
# ---------------------------------------------------------------------------


class RadiologyDepartmentGenerator:
    """SimPy-based generator that simulates a full radiology department.

    Produces per-patient records with all timestamps and features needed
    for ML model training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tier: str = "tier_1",
        seed: int = 42,
    ) -> None:
        self.sim_config = config["simulation"]
        self.hospital_config = config["hospital"][tier]
        self.patient_dists = self.sim_config.get("patient_distributions", {})
        self.service_times = self.sim_config.get("service_times", {})
        self.process_times = self.sim_config.get("process_times", {})
        self.arrival_config = self.sim_config.get("arrival_rates", {})
        self.tier = tier
        self.seed = seed

        # Records collected during simulation
        self.records: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._patient_counter = 0

    def _get_shift(self, hour: int) -> ShiftType:
        """Determine shift type from hour of day."""
        if hour < 14:
            return ShiftType.MORNING
        elif hour < 20:
            return ShiftType.AFTERNOON
        return ShiftType.EVENING

    def _sample_patient_attributes(self, current_date: datetime) -> dict[str, Any]:
        """Generate random patient demographic and clinical attributes."""
        d = self.patient_dists

        age_weights = d.get("age_group_weights", {"0-18": 0.1, "18-40": 0.35, "40-60": 0.3, "60+": 0.25})
        age_group = _weighted_choice(list(age_weights.keys()), list(age_weights.values()))

        gender_weights = d.get("gender_weights", {"M": 0.52, "F": 0.48})
        gender = _weighted_choice(list(gender_weights.keys()), list(gender_weights.values()))

        ins_weights = d.get("insurance_weights", {"government": 0.45, "private": 0.35, "self_pay": 0.20})
        insurance = _weighted_choice(list(ins_weights.keys()), list(ins_weights.values()))

        complexity_weights = d.get("exam_complexity_weights", {"simple": 0.50, "moderate": 0.35, "complex": 0.15})
        complexity = _weighted_choice(list(complexity_weights.keys()), list(complexity_weights.values()))

        dist_weights = d.get("distance_category_weights", {"local": 0.60, "city": 0.30, "outstation": 0.10})
        distance = _weighted_choice(list(dist_weights.keys()), list(dist_weights.values()))

        # Modality selection based on tier
        mod_dist = self.hospital_config.get("modality_distribution",
                    self.sim_config.get("hospital", {}).get("modality_distribution", {}).get(self.tier, {}))
        if not mod_dist:
            # fallback from outer hospital config
            mod_dist_outer = self._get_modality_distribution()
            modality = _weighted_choice(list(mod_dist_outer.keys()), list(mod_dist_outer.values()))
        else:
            modality = _weighted_choice(list(mod_dist.keys()), list(mod_dist.values()))

        # Contrast and prep probabilities
        contrast_probs = d.get("contrast_probability", {})
        prep_probs = d.get("prep_probability", {})
        requires_contrast = random.random() < contrast_probs.get(modality, 0.0)
        requires_prep = random.random() < prep_probs.get(modality, 0.0)

        # No-show history
        previous_no_show = max(0, int(random.expovariate(1.5)))
        previous_appointments = previous_no_show + max(0, int(random.expovariate(0.3)))
        is_repeat = previous_appointments > 0

        return {
            "age_group": age_group,
            "gender": gender,
            "insurance_type": insurance,
            "exam_complexity": complexity,
            "distance_category": distance,
            "modality": modality,
            "requires_contrast": requires_contrast,
            "requires_prep": requires_prep,
            "previous_no_show_count": previous_no_show,
            "previous_appointment_count": previous_appointments,
            "is_repeat_patient": is_repeat,
        }

    def _get_modality_distribution(self) -> dict[str, float]:
        """Get modality distribution from the hospital config hierarchy."""
        outer_hospital = self.sim_config.get("hospital", {})
        if isinstance(outer_hospital, dict):
            mod_dist = outer_hospital.get("modality_distribution", {}).get(self.tier, {})
            if mod_dist:
                return mod_dist
        # Hardcoded fallback matching §6.3
        return {"xray": 0.45, "ct": 0.20, "mri": 0.10, "ultrasound": 0.25}

    def _get_service_time(self, modality: str) -> float:
        """Sample service time for a modality from its log-normal distribution."""
        st = self.service_times.get(modality, {})
        mu = st.get("mu", 1.8)
        sigma = st.get("sigma", 0.4)
        min_val = st.get("min_minutes", 2.0)
        max_val = st.get("max_minutes", 120.0)
        return _sample_lognormal(mu, sigma, min_val, max_val)

    def _patient_process(
        self,
        env: simpy.Environment,
        patient_id: str,
        attrs: dict[str, Any],
        resources: dict[str, simpy.Resource | simpy.PriorityResource],
        sim_start: datetime,
        queue_state: dict[str, Any],
    ) -> Any:
        """SimPy process for a single patient flowing through the department.

        Flow: registration → queue → prep → imaging → reporting → departure
        """
        arrival_sim_time = env.now
        arrival_dt = sim_start + timedelta(minutes=arrival_sim_time)

        hour = arrival_dt.hour
        shift = self._get_shift(hour)

        # Snapshot queue state at arrival
        modality = attrs["modality"]
        queue_snapshot = {
            "current_queue_length_total": queue_state["total_in_queue"],
            "current_queue_length_same_modality": queue_state["modality_queues"].get(modality, 0),
            "patients_in_service_count": queue_state["in_service"],
            "avg_service_time_last_5_patients": queue_state["avg_service_last_5"],
            "time_since_last_patient_served_minutes": queue_state["time_since_last_served"],
            "emergency_patients_in_queue": queue_state["emergency_in_queue"],
            "num_machines_available": self.hospital_config.get("resources", {}).get(
                MODALITY_RESOURCE_KEYS[modality], 1),
            "num_technologists_on_duty": self.hospital_config.get("staff", {}).get("technologists", 5),
            "num_radiologists_on_duty": self.hospital_config.get("staff", {}).get("radiologists", 2),
        }

        record: dict[str, Any] = {
            "patient_id": patient_id,
            **attrs,
            **queue_snapshot,
            "equipment_under_maintenance": random.random() < 0.05,
            "shift_type": shift.value,
            "hour_of_day": hour,
            "day_of_week": arrival_dt.weekday(),
            "is_weekend": arrival_dt.weekday() >= 5,
            "is_holiday": generate_holiday_features(arrival_dt.date())["is_holiday"],
            "is_monday": arrival_dt.weekday() == 0,
            "minutes_since_department_opened": max(0, (hour - 8) * 60 + arrival_dt.minute),
            "weather_category": get_weather_category(arrival_dt.date()).value,
            "registration_time": arrival_dt,
        }

        # Update queue state
        queue_state["total_in_queue"] += 1
        queue_state["modality_queues"][modality] = queue_state["modality_queues"].get(modality, 0) + 1
        if attrs.get("urgency", "routine") == "emergency":
            queue_state["emergency_in_queue"] += 1

        # 1. Registration
        pt = self.process_times.get("registration", {})
        reg_time = _sample_normal_positive(pt.get("mean_minutes", 5.0), pt.get("std_minutes", 2.0),
                                           pt.get("min_minutes", 1.0))
        with resources["registration"].request() as req:
            yield req
            yield env.timeout(reg_time)

        record["queue_entry_time"] = sim_start + timedelta(minutes=env.now)

        # 2. Wait for modality resource (imaging room + technologist)
        urgency = attrs.get("urgency", "routine")
        priority = {"emergency": 0, "urgent": 1, "routine": 2}.get(urgency, 2)

        modality_resource_key = f"{modality}_rooms"
        if modality_resource_key not in resources:
            modality_resource_key = modality  # fallback key

        with resources[modality_resource_key].request(priority=priority) as mod_req:
            yield mod_req

            with resources["technologists"].request() as tech_req:
                yield tech_req

                # 3. Prep
                pt_prep = self.process_times.get("technologist_prep", {})
                prep_time = _sample_normal_positive(pt_prep.get("mean_minutes", 4.0),
                                                    pt_prep.get("std_minutes", 2.0),
                                                    pt_prep.get("min_minutes", 1.0))
                if attrs.get("requires_prep"):
                    prep_time *= 1.5  # extra prep time

                record["prep_start_time"] = sim_start + timedelta(minutes=env.now)
                yield env.timeout(prep_time)

                # 4. Imaging
                service_time = self._get_service_time(modality)
                if attrs.get("requires_contrast"):
                    service_time *= 1.3  # contrast adds ~30% time

                record["imaging_start_time"] = sim_start + timedelta(minutes=env.now)
                queue_state["in_service"] += 1
                queue_state["total_in_queue"] = max(0, queue_state["total_in_queue"] - 1)
                queue_state["modality_queues"][modality] = max(0, queue_state["modality_queues"].get(modality, 1) - 1)

                yield env.timeout(service_time)

                record["imaging_end_time"] = sim_start + timedelta(minutes=env.now)
                queue_state["in_service"] = max(0, queue_state["in_service"] - 1)
                queue_state["recent_service_times"].append(service_time)
                if len(queue_state["recent_service_times"]) > 5:
                    queue_state["recent_service_times"].pop(0)
                queue_state["avg_service_last_5"] = (
                    sum(queue_state["recent_service_times"]) / len(queue_state["recent_service_times"])
                )
                queue_state["time_since_last_served"] = 0.0

        # 5. Reporting (radiologist)
        with resources["radiologists"].request() as rad_req:
            yield rad_req
            pt_report = self.process_times.get("radiologist_reporting", {})
            report_time = _sample_normal_positive(pt_report.get("mean_minutes", 12.0),
                                                   pt_report.get("std_minutes", 5.0),
                                                   pt_report.get("min_minutes", 3.0))
            record["reporting_start_time"] = sim_start + timedelta(minutes=env.now)
            yield env.timeout(report_time)
            record["reporting_end_time"] = sim_start + timedelta(minutes=env.now)

        record["departure_time"] = sim_start + timedelta(minutes=env.now)

        # Compute actual wait time (registration to imaging start)
        if record.get("imaging_start_time") and record.get("registration_time"):
            wait_delta = record["imaging_start_time"] - record["registration_time"]
            record["actual_wait_time_minutes"] = round(wait_delta.total_seconds() / 60, 1)
        else:
            record["actual_wait_time_minutes"] = None

        # No-show flag (this patient showed up since they went through the process)
        record["showed_up"] = True

        if urgency == "emergency":
            queue_state["emergency_in_queue"] = max(0, queue_state["emergency_in_queue"] - 1)

        self.records.append(record)

    def _noshow_probability(self, attrs: dict[str, Any], day_dt: datetime, lead_time: int) -> float:
        """No-show probability for a scheduled appointment (see ``compute_noshow_probability``)."""
        cfg = self.sim_config.get("noshow", {})
        base = self.hospital_config.get("noshow_rate", cfg.get("base_rate", 0.20))
        return compute_noshow_probability(base, cfg, attrs, day_dt, lead_time)

    def _record_noshow(
        self,
        patient_id: str,
        attrs: dict[str, Any],
        appointment_dt: datetime,
        queue_state: dict[str, Any],
    ) -> None:
        """Record a scheduled patient who did not arrive, with the real queue state at that time."""
        modality = attrs["modality"]
        hour = appointment_dt.hour
        resources = self.hospital_config.get("resources", {})
        self.records.append({
            "patient_id": patient_id,
            **attrs,
            "current_queue_length_total": queue_state["total_in_queue"],
            "current_queue_length_same_modality": queue_state["modality_queues"].get(modality, 0),
            "patients_in_service_count": queue_state["in_service"],
            "avg_service_time_last_5_patients": queue_state["avg_service_last_5"],
            "time_since_last_patient_served_minutes": queue_state["time_since_last_served"],
            "emergency_patients_in_queue": queue_state["emergency_in_queue"],
            "num_machines_available": resources.get(MODALITY_RESOURCE_KEYS[modality], 1),
            "num_technologists_on_duty": self.hospital_config.get("staff", {}).get("technologists", 5),
            "num_radiologists_on_duty": self.hospital_config.get("staff", {}).get("radiologists", 2),
            "equipment_under_maintenance": False,
            "shift_type": self._get_shift(hour).value,
            "hour_of_day": hour,
            "day_of_week": appointment_dt.weekday(),
            "is_weekend": appointment_dt.weekday() >= 5,
            "is_holiday": generate_holiday_features(appointment_dt.date())["is_holiday"],
            "is_monday": appointment_dt.weekday() == 0,
            "minutes_since_department_opened": max(0, (hour - 8) * 60 + appointment_dt.minute),
            "weather_category": get_weather_category(appointment_dt.date()).value,
            "registration_time": appointment_dt,
            "showed_up": False,
            "actual_wait_time_minutes": None,
        })

    def generate(self, n_days: int = 180, start_date: datetime | None = None) -> pd.DataFrame:
        """Run the full simulation and return a DataFrame of patient records.

        Args:
            n_days: Number of days to simulate.
            start_date: Simulation start date (defaults to 2024-01-01).

        Returns:
            DataFrame with 50K+ patient records.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        start_date = start_date or datetime(2024, 1, 1, 8, 0, 0)
        self.records = []
        self._patient_counter = 0

        daily_volume = self.hospital_config.get("daily_patient_volume", 200)
        walk_in_ratio = self.hospital_config.get("walk_in_ratio", 0.70)
        emergency_ratio = self.hospital_config.get("emergency_ratio", 0.10)
        operating_hours = 12  # 8 AM to 8 PM

        hourly_mults = self.arrival_config.get("hourly_multipliers", {})
        resources_config = self.hospital_config.get("resources", {})

        logger.info(f"Starting simulation: {n_days} days, ~{daily_volume} patients/day, tier={self.tier}")

        for day in range(n_days):
            day_date = start_date + timedelta(days=day)
            demand_mult = get_demand_multiplier(day_date.date())
            emergency_mult = get_monsoon_emergency_multiplier(day_date.date())

            adj_volume = int(daily_volume * demand_mult)
            if adj_volume < 5:
                continue  # Skip near-zero days (major holidays)

            # Create SimPy environment for this day
            env = simpy.Environment()

            # Create resources
            resources: dict[str, Any] = {
                "registration": simpy.Resource(env, capacity=resources_config.get("registration_desks", 2)),
                "xray_rooms": simpy.PriorityResource(env, capacity=resources_config.get("xray_rooms", 3)),
                "ct_rooms": simpy.PriorityResource(env, capacity=resources_config.get("ct_scanners", 2)),
                "mri_rooms": simpy.PriorityResource(env, capacity=resources_config.get("mri_machines", 1)),
                "ultrasound_rooms": simpy.PriorityResource(env, capacity=resources_config.get("us_rooms", 2)),
                "technologists": simpy.Resource(env, capacity=resources_config.get("technologists",
                                                self.hospital_config.get("staff", {}).get("technologists", 8))),
                "radiologists": simpy.Resource(env, capacity=resources_config.get("radiologists",
                                               self.hospital_config.get("staff", {}).get("radiologists", 4))),
            }

            # Map modality names to resource keys
            modality_resource_map = {
                "xray": "xray_rooms",
                "ct": "ct_rooms",
                "mri": "mri_rooms",
                "ultrasound": "ultrasound_rooms",
            }
            for mod_name, res_key in modality_resource_map.items():
                resources[mod_name] = resources[res_key]

            queue_state: dict[str, Any] = {
                "total_in_queue": 0,
                "modality_queues": {},
                "in_service": 0,
                "avg_service_last_5": 10.0,
                "time_since_last_served": 0.0,
                "emergency_in_queue": 0,
                "recent_service_times": [10.0],
            }

            base_rate = adj_volume / operating_hours

            # Schedule arrivals throughout the day
            def _arrival_generator(
                env: simpy.Environment,
                day_dt: datetime,
                base_r: float,
            ) -> Any:
                for hour in range(8, 20):
                    rate = _hourly_arrival_rate(hour, base_r, hourly_mults)
                    n_patients_this_hour = max(0, np.random.poisson(rate))

                    for _ in range(n_patients_this_hour):
                        self._patient_counter += 1
                        patient_id = f"P-{self._patient_counter:05d}"
                        attrs = self._sample_patient_attributes(day_dt)

                        # Determine visit type
                        r = random.random()
                        adj_emergency_ratio = emergency_ratio * emergency_mult
                        if r < adj_emergency_ratio:
                            visit_type = "emergency"
                            urgency = "emergency"
                            lead_time = 0
                        elif r < adj_emergency_ratio + walk_in_ratio * (1 - adj_emergency_ratio):
                            visit_type = "walk_in"
                            urgency = random.choices(["routine", "urgent"], weights=[0.85, 0.15])[0]
                            lead_time = 0
                        else:
                            visit_type = "scheduled"
                            urgency = random.choices(["routine", "urgent"], weights=[0.90, 0.10])[0]
                            lead_time = random.choice([1, 2, 3, 5, 7, 14, 21])

                        attrs["visit_type"] = visit_type
                        attrs["urgency"] = urgency
                        attrs["appointment_lead_time_days"] = lead_time

                        # Random arrival within the hour
                        minute_offset = random.uniform(0, 60)
                        arrival_time = (hour - 8) * 60 + minute_offset

                        yield env.timeout(max(0, arrival_time - env.now))

                        if visit_type == "scheduled" and random.random() < self._noshow_probability(
                            attrs, day_dt, lead_time,
                        ):
                            self._record_noshow(
                                patient_id, attrs, day_dt + timedelta(minutes=env.now), queue_state,
                            )
                            continue

                        env.process(self._patient_process(
                            env, patient_id, attrs, resources, day_dt, queue_state,
                        ))

            env.process(_arrival_generator(env, day_date, base_rate))
            env.run(until=operating_hours * 60 + 120)  # run 2 extra hours for stragglers

            if (day + 1) % 30 == 0:
                logger.info(f"Day {day + 1}/{n_days} complete. Records so far: {len(self.records)}")

        df = pd.DataFrame(self.records)
        logger.info(f"Simulation complete. Total records: {len(df)}")
        return df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic radiology OPD dataset")
    parser.add_argument("--config", type=Path, default=Path("config/simulation_config.yaml"),
                        help="Path to simulation config YAML")
    parser.add_argument("--output", type=Path, default=Path("data/generated"),
                        help="Output directory for generated CSV files")
    parser.add_argument("--days", type=int, default=180, help="Number of days to simulate")
    parser.add_argument("--tier", type=str, default="tier_1",
                        choices=["tier_1", "tier_2", "tier_3"],
                        help="Hospital tier to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = _load_config(args.config)
    generator = RadiologyDepartmentGenerator(config, tier=args.tier, seed=args.seed)
    df = generator.generate(n_days=args.days)

    # Save output
    args.output.mkdir(parents=True, exist_ok=True)
    patients_path = args.output / "patients.csv"
    df.to_csv(patients_path, index=False)
    logger.info(f"Saved {len(df)} patient records to {patients_path}")

    # Summary statistics
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['registration_time'].min()} to {df['registration_time'].max()}")
    if "actual_wait_time_minutes" in df.columns:
        valid_waits = df["actual_wait_time_minutes"].dropna()
        logger.info(f"Wait time stats — mean: {valid_waits.mean():.1f}, "
                     f"median: {valid_waits.median():.1f}, "
                     f"p90: {valid_waits.quantile(0.9):.1f}")


if __name__ == "__main__":
    main()
