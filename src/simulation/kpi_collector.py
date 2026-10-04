"""KPI Collector for simulation.

Owner: P4
"""
from typing import Any
from collections import defaultdict
from src.simulation.entities import SimPatient
from src.utils.constants import ModalityType

class KPICollector:
    def __init__(self):
        self.patient_records: list[dict[str, Any]] = []
        self.queue_lengths = defaultdict(list)
        self.utilization = defaultdict(list)

    def record_patient(self, patient: SimPatient):
        wait_time = None
        if patient.imaging_start is not None and patient.registration_end is not None:
            wait_time = patient.imaging_start - patient.registration_end

        record = {
            "patient_id": patient.id,
            "modality": patient.modality.value,
            "urgency": patient.urgency.value,
            "wait_time_minutes": wait_time,
            "arrival_time": patient.arrival_time,
            "departure_time": patient.departure_time,
        }
        self.patient_records.append(record)

    def record_queue_length(self, env_now: float, modality: ModalityType, length: int):
        self.queue_lengths[modality.value].append((env_now, length))
        
    def record_utilization(self, env_now: float, modality: ModalityType, in_use: int, total: int):
        rate = in_use / total if total > 0 else 0
        self.utilization[modality.value].append((env_now, rate))
