"""KPI collection for the SimPy digital twin.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §8.1 (KPI collectors), §12.3 (scheduling metrics)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.scheduler.fairness import FairnessEngine
from src.simulation.entities import SimPatient
from src.utils.constants import (
    MAX_WAIT_EMERGENCY_MINUTES,
    MAX_WAIT_ROUTINE_MINUTES,
    MAX_WAIT_URGENT_MINUTES,
    ModalityType,
)

P90 = 90


class KPICollector:
    """Collects per-patient records, queue samples and machine busy time."""

    def __init__(self) -> None:
        self.patient_records: list[dict[str, Any]] = []
        self.noshow_count: int = 0
        self.queue_samples: list[dict[str, Any]] = []
        self.busy_minutes: dict[str, float] = defaultdict(float)
        self.capacity_minutes: dict[str, float] = defaultdict(float)
        self.days_simulated: int = 0

    def record_patient(self, patient: SimPatient) -> None:
        """Store the timeline of a patient who completed reporting."""
        self.patient_records.append({
            "patient_id": patient.id,
            "day": patient.day,
            "modality": patient.modality.value,
            "urgency": patient.urgency.value,
            "visit_type": patient.visit_type.value,
            "is_overbooked": patient.is_overbooked,
            "arrival_time": patient.arrival_time,
            "queue_entry_time": patient.queue_entry_time,
            "scan_start_time": patient.scan_start_time,
            "imaging_end_time": patient.imaging_end_time,
            "departure_time": patient.departure_time,
            "wait_time_minutes": patient.queue_wait,
            "door_to_scan_minutes": (patient.scan_start_time or 0.0) - patient.arrival_time,
            "turnaround_minutes": (patient.departure_time or 0.0) - patient.arrival_time,
        })

    def record_noshow(self) -> None:
        """Count a scheduled patient who did not arrive."""
        self.noshow_count += 1

    def record_queue_length(self, day: int, minute: float, modality: ModalityType, length: int) -> None:
        """Store a periodic queue-length sample."""
        self.queue_samples.append({"day": day, "minute": minute, "modality": modality.value, "queue_length": length})

    def record_busy_interval(
        self, modality: ModalityType, start: float, end: float, open_minute: float, close_minute: float,
    ) -> None:
        """Add machine busy time that falls inside operating hours."""
        overlap = max(0.0, min(end, close_minute) - max(start, open_minute))
        self.busy_minutes[modality.value] += overlap

    def record_capacity(self, modality: ModalityType, machine_minutes: float) -> None:
        """Add available machine-minutes for one operating day."""
        self.capacity_minutes[modality.value] += machine_minutes

    def to_dataframe(self) -> pd.DataFrame:
        """Per-patient records as a DataFrame."""
        return pd.DataFrame(self.patient_records)

    def summary(self) -> dict[str, float]:
        """Aggregate KPIs (§12.3) for this run."""
        df = self.to_dataframe()
        if df.empty:
            return {"patients_served": 0, "no_shows": self.noshow_count}
        waits = df["wait_time_minutes"].to_numpy(dtype=float)
        equity = FairnessEngine.compute_equity_metrics(list(waits))
        out: dict[str, float] = {
            "patients_served": float(len(df)),
            "patients_per_day": len(df) / max(1, self.days_simulated),
            "overbooked_served": float(df["is_overbooked"].sum()),
            "no_shows": float(self.noshow_count),
            "avg_wait": float(np.mean(waits)),
            "median_wait": float(np.median(waits)),
            "p90_wait": float(np.percentile(waits, P90)),
            "max_wait": float(np.max(waits)),
            "std_wait": float(np.std(waits)),
            "gini_wait": equity["gini"],
            "avg_turnaround": float(df["turnaround_minutes"].mean()),
        }
        out.update(self._urgency_kpis(df))
        total_busy = sum(self.busy_minutes.values())
        total_cap = sum(self.capacity_minutes.values())
        out["utilization"] = total_busy / total_cap if total_cap else 0.0
        for modality, cap in self.capacity_minutes.items():
            out[f"utilization_{modality}"] = self.busy_minutes[modality] / cap if cap else 0.0
        return out

    @staticmethod
    def _urgency_kpis(df: pd.DataFrame) -> dict[str, float]:
        """Emergency/urgent compliance and routine starvation rate."""
        out: dict[str, float] = {}
        limits = {
            "emergency": MAX_WAIT_EMERGENCY_MINUTES,
            "urgent": MAX_WAIT_URGENT_MINUTES,
            "routine": MAX_WAIT_ROUTINE_MINUTES,
        }
        for urgency, limit in limits.items():
            waits = df.loc[df["urgency"] == urgency, "wait_time_minutes"]
            if waits.empty:
                continue
            out[f"{urgency}_avg_wait"] = float(waits.mean())
            out[f"{urgency}_max_wait"] = float(waits.max())
            out[f"{urgency}_within_limit_rate"] = float((waits <= limit).mean())
        out["starvation_rate"] = 1.0 - out.get("routine_within_limit_rate", 1.0)
        return out
