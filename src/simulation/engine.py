"""Core SimPy Simulation Engine.

Owner: P4
"""
import simpy
from typing import Any
from src.simulation.kpi_collector import KPICollector
from src.simulation.processes import patient_process
from src.simulation.arrival_patterns import generate_arrivals
from src.scheduler.policies import BaseSchedulingPolicy
from src.scheduler.models import DepartmentState
from src.simulation.entities import SimPatient
from src.utils.constants import ModalityType

class RadiologySimulation:
    def __init__(self, policy: BaseSchedulingPolicy):
        self.env = simpy.Environment()
        self.policy = policy
        self.kpi = KPICollector()
        
        self.resources = {
            "registration": simpy.Resource(self.env, capacity=3),
            "xray_rooms": simpy.Resource(self.env, capacity=3),
            "ct_rooms": simpy.Resource(self.env, capacity=2),
            "mri_rooms": simpy.Resource(self.env, capacity=1),
            "ultrasound_rooms": simpy.Resource(self.env, capacity=2),
            "radiologists": simpy.Resource(self.env, capacity=4),
        }
        
        self.queue: list[SimPatient] = []
        self.state = DepartmentState()
        
    def patient_arrival_handler(self, patient: SimPatient):
        self.queue.append(patient)
        patient.assigned_event = self.env.event()
        patient.finished_event = self.env.event()
        self.env.process(self._run_patient(patient))
        self._dispatch()

    def _run_patient(self, patient: SimPatient):
        # Start the patient process
        proc = self.env.process(patient_process(self.env, patient, self.resources, self.kpi))
        
        # When imaging is done (or the patient leaves), try dispatching another
        yield patient.finished_event
        
        modality_key = f"{patient.modality.value}_rooms"
        # Release happens implicitly or we can manage it. In processes.py we just yielded timeout.
        # Wait, if we use SimPy Resources for machines, we should acquire them in the engine or processes.
        # Since the engine handles dispatching, let's keep it simple.
        
        self._dispatch()

    def _dispatch(self):
        """Invoke the scheduling policy to dispatch waiting patients to available resources."""
        # Simple loop: check available machines per modality
        for modality in ModalityType:
            modality_key = f"{modality.value}_rooms"
            res = self.resources.get(modality_key)
            if not res: continue
            
            # Count free slots
            free = res.capacity - res.count
            
            # Filter queue for this modality
            modality_queue = [p for p in self.queue if p.modality == modality and not hasattr(p, 'dispatched')]
            
            if free > 0 and modality_queue:
                # Convert to PatientState for the policy
                patient_states = [p.to_patient_state(self.env.now) for p in modality_queue]
                
                ordered = self.policy.order_queue(patient_states, self.state, modality)
                
                for patient_state in ordered[:free]:
                    # Find the SimPatient
                    sim_p = next(p for p in modality_queue if p.id == patient_state.patient_id)
                    sim_p.dispatched = True
                    # Trigger them to proceed to imaging
                    req = res.request()
                    sim_p.req = req
                    sim_p.assigned_event.succeed()

    def run(self, until: int = 1440):
        self.env.process(generate_arrivals(self.env, rate=1/10.0, patient_handler=self.patient_arrival_handler, n_patients=200))
        self.env.run(until=until)
        return self.kpi
