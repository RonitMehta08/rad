"""Entities and configuration for the SimPy radiology digital twin.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §8.1; config/hospital_profiles.yaml, config/simulation_config.yaml
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.scheduler.models import PatientState
from src.utils.constants import ModalityType, UrgencyLevel, VisitType

CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "config"
DEFAULT_START_DATE: datetime = datetime(2024, 1, 1)

# hospital_profiles.yaml resource keys per modality
MODALITY_RESOURCE_KEYS: dict[ModalityType, str] = {
    ModalityType.XRAY: "xray_rooms",
    ModalityType.CT: "ct_scanners",
    ModalityType.MRI: "mri_machines",
    ModalityType.ULTRASOUND: "us_rooms",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, raising an actionable error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Simulation config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ShiftStaffing:
    """Technologist/radiologist counts for one shift window."""

    start_hour: int
    end_hour: int
    technologists: int
    radiologists: int


@dataclass
class SimulationSettings:
    """All knobs of one simulated department, built from the YAML configs.

    Attributes mirror hospital_profiles.yaml (per tier) and simulation_config.yaml;
    what-if scenarios are applied on top via ``with_scenario``.
    """

    tier: str
    days: int = 1
    start_date: datetime = DEFAULT_START_DATE
    open_hour: int = 8
    close_hour: int = 20
    daily_volume: float = 200.0
    walk_in_ratio: float = 0.70
    emergency_ratio: float = 0.10
    noshow_rate: float = 0.20
    modality_mix: dict[str, float] = field(default_factory=dict)
    machines: dict[ModalityType, int] = field(default_factory=dict)
    registration_desks: int = 2
    shifts: list[ShiftStaffing] = field(default_factory=list)
    hourly_multipliers: dict[int, float] = field(default_factory=dict)
    service_times: dict[str, dict[str, float]] = field(default_factory=dict)
    process_times: dict[str, dict[str, float]] = field(default_factory=dict)
    patient_distributions: dict[str, Any] = field(default_factory=dict)
    noshow_config: dict[str, Any] = field(default_factory=dict)
    arrival_rate_multiplier: float = 1.0
    noshow_rate_multiplier: float = 1.0
    emergency_buffer_fraction: float = 0.0
    scenario_name: str = "baseline"

    @classmethod
    def from_config(
        cls,
        tier: str = "tier_1",
        days: int = 1,
        config_dir: Path = CONFIG_DIR,
    ) -> SimulationSettings:
        """Build settings for a hospital tier from the project YAML configs.

        Args:
            tier: ``tier_1`` | ``tier_2`` | ``tier_3``.
            days: Number of operating days to simulate.
            config_dir: Directory containing the YAML configs.

        Returns:
            Fully populated SimulationSettings.
        """
        hospital = _load_yaml(config_dir / "hospital_profiles.yaml")
        sim = _load_yaml(config_dir / "simulation_config.yaml")
        if tier not in hospital:
            raise ValueError(f"Unknown hospital tier '{tier}'. Available: tier_1, tier_2, tier_3")
        profile = hospital[tier]
        resources = profile.get("resources", {})
        shifts = [
            ShiftStaffing(v["start_hour"], v["end_hour"], v["technologists"], v["radiologists"])
            for v in profile.get("shifts", {}).values()
        ]
        hours = sim.get("operating_hours", {})
        return cls(
            tier=tier,
            days=days,
            open_hour=hours.get("open_hour", 8),
            close_hour=hours.get("close_hour", 20),
            daily_volume=profile.get("daily_patient_volume", 200),
            walk_in_ratio=profile.get("walk_in_ratio", 0.70),
            emergency_ratio=profile.get("emergency_ratio", 0.10),
            noshow_rate=profile.get("noshow_rate", 0.20),
            modality_mix=hospital.get("modality_distribution", {}).get(tier, {}),
            machines={m: int(resources.get(k, 1)) for m, k in MODALITY_RESOURCE_KEYS.items()},
            registration_desks=resources.get("registration_desks", 2),
            shifts=shifts,
            hourly_multipliers={
                int(h): float(v) for h, v in sim.get("arrival_rates", {}).get("hourly_multipliers", {}).items()
            },
            service_times=sim.get("service_times", {}),
            process_times=sim.get("process_times", {}),
            patient_distributions=sim.get("patient_distributions", {}),
            noshow_config=sim.get("noshow", {}),
        )

    def with_scenario(self, scenario: dict[str, Any]) -> SimulationSettings:
        """Return a copy with a what-if scenario applied.

        Supported keys (as in simulation_config.yaml ``whatif_scenarios``):
        ``machine_adjustments``, ``staff_adjustments``, ``arrival_rate_multiplier``,
        ``noshow_rate_multiplier``, ``emergency_buffer_fraction``.
        """
        new = copy.deepcopy(self)
        new.scenario_name = scenario.get("name", "custom")
        for modality, delta in scenario.get("machine_adjustments", {}).items():
            key = ModalityType(modality)
            new.machines[key] = max(0, new.machines.get(key, 0) + int(delta))
        staff = scenario.get("staff_adjustments", {})
        for shift in new.shifts:
            shift.technologists = max(1, shift.technologists + int(staff.get("technologists", 0)))
            shift.radiologists = max(1, shift.radiologists + int(staff.get("radiologists", 0)))
        new.arrival_rate_multiplier *= float(scenario.get("arrival_rate_multiplier", 1.0))
        new.noshow_rate_multiplier *= float(scenario.get("noshow_rate_multiplier", 1.0))
        if "emergency_buffer_fraction" in scenario:
            new.emergency_buffer_fraction = float(scenario["emergency_buffer_fraction"])
        return new

    def staffing_at(self, minute_of_day: float) -> ShiftStaffing:
        """Staffing for the shift covering ``minute_of_day`` (last shift covers overtime)."""
        hour = minute_of_day / 60.0
        for shift in self.shifts:
            if shift.start_hour <= hour < shift.end_hour:
                return shift
        return self.shifts[-1] if hour >= self.shifts[-1].start_hour else self.shifts[0]


@dataclass
class SimPatient:
    """One patient in the digital twin.

    All random quantities (service times etc.) are sampled up-front by
    ``arrival_patterns.generate_day_patients`` so that every scheduling policy
    sees an identical patient stream (common random numbers).
    """

    id: str
    day: int
    arrival_time: float  # minutes since midnight
    modality: ModalityType
    urgency: UrgencyLevel
    visit_type: VisitType
    will_show: bool = True
    is_overbooked: bool = False
    noshow_probability: float = 0.0
    registration_minutes: float = 5.0
    prep_minutes: float = 4.0
    scan_minutes: float = 10.0
    report_minutes: float = 12.0
    estimated_duration: float = 10.0  # what the scheduler can know at registration
    attributes: dict[str, Any] = field(default_factory=dict)

    # Timestamps filled during simulation (minutes since midnight)
    queue_entry_time: float | None = None
    scan_start_time: float | None = None
    imaging_end_time: float | None = None
    departure_time: float | None = None

    @property
    def queue_wait(self) -> float | None:
        """Minutes from joining the modality queue to starting prep/scan."""
        if self.queue_entry_time is None or self.scan_start_time is None:
            return None
        return self.scan_start_time - self.queue_entry_time

    def to_patient_state(self, current_time: float) -> PatientState:
        """Convert to the scheduler's PatientState."""
        entry = self.queue_entry_time if self.queue_entry_time is not None else self.arrival_time
        return PatientState(
            patient_id=self.id,
            modality=self.modality,
            urgency=self.urgency,
            visit_type=self.visit_type,
            arrival_time_minutes=entry,
            current_wait_minutes=max(0.0, current_time - entry),
            estimated_duration_minutes=self.estimated_duration,
            status="waiting",
            features=self.attributes,
        )
