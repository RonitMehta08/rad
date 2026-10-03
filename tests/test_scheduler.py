"""Unit and integration tests for the RadQueue AI Dynamic Scheduling Engine.

Covers:
- Comparative policies (FCFS, Priority, SJF, Wave)
- Fairness calculations and anti-starvation rules
- Level 1 MILP day-ahead optimizer
- Level 2 Real-time priority dispatching
- Level 3 Rescheduling triggers (Emergency preemption, No-show pull-forward, Equipment failure, Surge)
- Unified RadiologyScheduler facade

Owner: P3 (Scheduling Engineer)
"""

import pytest

from src.scheduler import (
    RadiologyScheduler,
    DayAheadOptimizer,
    RealTimeDispatcher,
    ReschedulingEngine,
    FairnessEngine,
    PatientState,
    Resource,
    Assignment,
    DepartmentState,
    get_policy,
)
from src.utils.constants import ModalityType, UrgencyLevel, VisitType


@pytest.fixture
def sample_patients():
    return [
        PatientState(
            patient_id="P_ROUTINE_1",
            modality=ModalityType.XRAY,
            urgency=UrgencyLevel.ROUTINE,
            arrival_time_minutes=10.0,
            current_wait_minutes=15.0,
        ),
        PatientState(
            patient_id="P_EMERGENCY_1",
            modality=ModalityType.XRAY,
            urgency=UrgencyLevel.EMERGENCY,
            arrival_time_minutes=25.0,
            current_wait_minutes=0.0,
        ),
        PatientState(
            patient_id="P_URGENT_1",
            modality=ModalityType.XRAY,
            urgency=UrgencyLevel.URGENT,
            arrival_time_minutes=5.0,
            current_wait_minutes=20.0,
        ),
        PatientState(
            patient_id="P_ROUTINE_LONG_WAIT",
            modality=ModalityType.XRAY,
            urgency=UrgencyLevel.ROUTINE,
            arrival_time_minutes=0.0,
            current_wait_minutes=70.0,  # Exceeds 60min starvation threshold
        ),
    ]


@pytest.fixture
def sample_resources():
    return [
        Resource(resource_id="XRAY_1", modality=ModalityType.XRAY, is_available=True),
        Resource(resource_id="XRAY_2", modality=ModalityType.XRAY, is_available=True),
        Resource(resource_id="CT_1", modality=ModalityType.CT, is_available=True),
    ]


@pytest.fixture
def department_state(sample_patients, sample_resources):
    return DepartmentState(
        current_time_minutes=30.0,
        avg_wait_minutes=20.0,
        queue_length={
            ModalityType.XRAY: 4,
            ModalityType.CT: 0,
            ModalityType.MRI: 0,
            ModalityType.ULTRASOUND: 0,
        },
        waiting_patients=sample_patients,
        resources=sample_resources,
    )


# ---------------------------------------------------------------------------
# Task 2: Comparative Policies Tests
# ---------------------------------------------------------------------------

def test_fcfs_policy_orders_strictly_by_arrival_time(sample_patients):
    fcfs = get_policy("fcfs")
    ordered = fcfs.order_queue(sample_patients)
    arrival_times = [p.arrival_time_minutes for p in ordered]
    assert arrival_times == sorted(arrival_times)
    assert ordered[0].patient_id == "P_ROUTINE_LONG_WAIT"  # arrived at min 0.0


def test_priority_policy_orders_emergency_before_urgent_before_routine(sample_patients):
    priority_policy = get_policy("priority")
    ordered = priority_policy.order_queue(sample_patients)
    assert ordered[0].urgency == UrgencyLevel.EMERGENCY
    assert ordered[1].urgency == UrgencyLevel.URGENT
    assert ordered[2].urgency == UrgencyLevel.ROUTINE


def test_sjf_policy_prioritizes_shortest_duration():
    sjf = get_policy("sjf")
    patients = [
        PatientState(patient_id="P_CT", modality=ModalityType.CT, arrival_time_minutes=0.0),  # ~23 min
        PatientState(patient_id="P_XRAY", modality=ModalityType.XRAY, arrival_time_minutes=0.0),  # ~6 min
        PatientState(patient_id="P_MRI", modality=ModalityType.MRI, arrival_time_minutes=0.0),  # ~45 min
    ]
    ordered = sjf.order_queue(patients)
    assert ordered[0].patient_id == "P_XRAY"
    assert ordered[1].patient_id == "P_CT"
    assert ordered[2].patient_id == "P_MRI"


def test_wave_policy_batches_by_window():
    wave = get_policy("wave", interval_minutes=30)
    p1 = PatientState(patient_id="P1", modality=ModalityType.XRAY, arrival_time_minutes=10.0, urgency=UrgencyLevel.ROUTINE)
    p2 = PatientState(patient_id="P2", modality=ModalityType.XRAY, arrival_time_minutes=15.0, urgency=UrgencyLevel.EMERGENCY)
    p3 = PatientState(patient_id="P3", modality=ModalityType.XRAY, arrival_time_minutes=45.0, urgency=UrgencyLevel.EMERGENCY)

    ordered = wave.order_queue([p1, p2, p3])
    # Wave 0 (0-30m): P2 (Emergency) comes before P1 (Routine)
    # Wave 1 (30-60m): P3 comes last because it's in wave 1
    assert [p.patient_id for p in ordered] == ["P2", "P1", "P3"]


# ---------------------------------------------------------------------------
# Task 3: Fairness & Starvation Tests
# ---------------------------------------------------------------------------

def test_fairness_penalty_increases_when_wait_exceeds_average():
    engine = FairnessEngine(penalty_rate=0.1)
    bonus_low = engine.compute_fairness_bonus(current_wait_minutes=10.0, avg_wait_minutes=20.0)
    bonus_high = engine.compute_fairness_bonus(current_wait_minutes=50.0, avg_wait_minutes=20.0)

    assert bonus_low == 0.0  # not above average
    assert bonus_high == (50.0 - 20.0) * 0.1 == 3.0


def test_starvation_detection_flags_patients_over_threshold(sample_patients):
    engine = FairnessEngine(starvation_threshold_minutes=60.0)
    starving = engine.get_starving_patients(sample_patients)
    assert len(starving) == 1
    assert starving[0].patient_id == "P_ROUTINE_LONG_WAIT"


def test_equity_metrics_calculation():
    metrics = FairnessEngine.compute_equity_metrics([10.0, 20.0, 30.0, 40.0])
    assert metrics["avg_wait"] == 25.0
    assert metrics["max_wait"] == 40.0
    assert 0.0 <= metrics["gini"] <= 1.0


# ---------------------------------------------------------------------------
# Task 4: Level 2 Real-Time Dispatcher Tests
# ---------------------------------------------------------------------------

def test_dispatcher_dynamic_priority_boosts_starving_patient(sample_patients, department_state):
    dispatcher = RealTimeDispatcher()
    scores = {
        p.patient_id: dispatcher.compute_priority_score(p, department_state)
        for p in sample_patients
    }
    # Emergency should have high score
    assert scores["P_EMERGENCY_1"] >= 10.0
    # Starving patient should receive starvation boost + fairness bonus
    assert scores["P_ROUTINE_LONG_WAIT"] > scores["P_ROUTINE_1"]


def test_dispatcher_dispatches_compatible_resource(sample_patients, sample_resources, department_state):
    dispatcher = RealTimeDispatcher()
    assignment = dispatcher.dispatch_next(
        sample_patients, sample_resources, department_state, modality=ModalityType.XRAY
    )
    assert assignment is not None
    assert assignment.resource_id in ["XRAY_1", "XRAY_2"]
    # Emergency or high priority patient dispatched
    assert assignment.urgency in [UrgencyLevel.EMERGENCY, UrgencyLevel.ROUTINE]


# ---------------------------------------------------------------------------
# Task 5: Level 1 Day-Ahead MILP Optimizer Tests
# ---------------------------------------------------------------------------

def test_day_ahead_optimizer_solves_multi_patient_schedule():
    optimizer = DayAheadOptimizer()
    patients = [
        PatientState(patient_id="X1", modality=ModalityType.XRAY, urgency=UrgencyLevel.ROUTINE, arrival_time_minutes=0.0),
        PatientState(patient_id="X2", modality=ModalityType.XRAY, urgency=UrgencyLevel.URGENT, arrival_time_minutes=15.0),
        PatientState(patient_id="C1", modality=ModalityType.CT, urgency=UrgencyLevel.ROUTINE, arrival_time_minutes=0.0),
    ]
    resources = [
        Resource(resource_id="XRAY_1", modality=ModalityType.XRAY),
        Resource(resource_id="CT_1", modality=ModalityType.CT),
    ]

    result = optimizer.optimize_schedule(patients, resources)
    assert result["status"] == "Optimal"
    assert len(result["assignments"]) == 3
    # Verify CT assigned to CT_1 and XRAY to XRAY_1
    for asgn in result["assignments"]:
        if asgn.patient_id == "C1":
            assert asgn.resource_id == "CT_1"
        else:
            assert asgn.resource_id == "XRAY_1"


# ---------------------------------------------------------------------------
# Task 6: Level 3 Dynamic Rescheduler Tests
# ---------------------------------------------------------------------------

def test_emergency_preemption_replaces_routine_assignment(department_state):
    rescheduler = ReschedulingEngine()
    routine_p = PatientState(patient_id="ROUTINE_P", modality=ModalityType.XRAY, urgency=UrgencyLevel.ROUTINE)
    active_assignments = [
        Assignment(
            patient_id="ROUTINE_P",
            resource_id="XRAY_1",
            start_time_minutes=25.0,
            estimated_duration_minutes=6.0,
            urgency=UrgencyLevel.ROUTINE,
        )
    ]
    emergency_p = PatientState(patient_id="EMERGENCY_NEW", modality=ModalityType.XRAY, urgency=UrgencyLevel.EMERGENCY)

    emergency_asgn, preempted = rescheduler.handle_emergency_preemption(
        emergency_p,
        active_assignments,
        waiting_queue=[routine_p],
        resources=[Resource(resource_id="XRAY_1", modality=ModalityType.XRAY, is_available=False)],
        state=department_state,
    )

    assert emergency_asgn is not None
    assert emergency_asgn.patient_id == "EMERGENCY_NEW"
    assert emergency_asgn.resource_id == "XRAY_1"
    assert preempted is not None
    assert preempted.patient_id == "ROUTINE_P"
    assert preempted.status == "preempted"


def test_noshow_pullforward_triggers_after_grace_window(department_state):
    rescheduler = ReschedulingEngine()
    late_patient = PatientState(
        patient_id="LATE_P",
        modality=ModalityType.XRAY,
        arrival_time_minutes=10.0,
    )
    # Department time is 30.0, delay = 20.0 > 15.0 min threshold
    waiting_p = PatientState(
        patient_id="WAITING_NEXT",
        modality=ModalityType.XRAY,
        arrival_time_minutes=15.0,
    )
    avail_resources = [Resource(resource_id="XRAY_1", modality=ModalityType.XRAY, is_available=True)]

    asgn = rescheduler.handle_noshow_pullforward(
        scheduled_patient=late_patient,
        waiting_queue=[late_patient, waiting_p],
        available_resources=avail_resources,
        state=department_state,
    )

    assert late_patient.status == "noshow"
    assert asgn is not None
    assert asgn.patient_id == "WAITING_NEXT"


def test_equipment_failure_redistributes_to_surviving_machine(department_state):
    rescheduler = ReschedulingEngine()
    resources = [
        Resource(resource_id="XRAY_1", modality=ModalityType.XRAY, is_available=True),
        Resource(resource_id="XRAY_2", modality=ModalityType.XRAY, is_available=True),
    ]
    displaced_p = PatientState(patient_id="DISPLACED_P", modality=ModalityType.XRAY)
    active_assignments = [
        Assignment(patient_id="DISPLACED_P", resource_id="XRAY_1", start_time_minutes=30.0, estimated_duration_minutes=6.0)
    ]

    new_asgns = rescheduler.handle_equipment_failure(
        failed_resource_id="XRAY_1",
        resources=resources,
        active_assignments=active_assignments,
        waiting_queue=[displaced_p],
        state=department_state,
    )

    assert resources[0].is_available is False  # XRAY_1 marked unavailable
    assert len(new_asgns) == 1
    assert new_asgns[0].resource_id == "XRAY_2"  # Re-routed to surviving machine


def test_surge_detection_two_sigma():
    rescheduler = ReschedulingEngine()
    # Mean = 10, std = 2 -> 2-sigma cutoff = 14
    normal_res = rescheduler.detect_surge(current_arrival_rate=12.0, mean_rate=10.0, std_rate=2.0)
    assert normal_res["is_surge"] is False

    surge_res = rescheduler.detect_surge(current_arrival_rate=16.0, mean_rate=10.0, std_rate=2.0)
    assert surge_res["is_surge"] is True
    assert surge_res["overflow_protocol_active"] is True


# ---------------------------------------------------------------------------
# Task 7: RadiologyScheduler Facade Integration Test
# ---------------------------------------------------------------------------

def test_unified_radiology_scheduler_facade(sample_patients, sample_resources, department_state):
    scheduler = RadiologyScheduler()

    # Test Level 2 Dispatch
    asgn = scheduler.dispatch_next(sample_patients, sample_resources, department_state)
    assert asgn is not None

    # Test Policy Ordering
    ordered_fcfs = scheduler.order_by_policy("fcfs", sample_patients)
    assert ordered_fcfs[0].patient_id == "P_ROUTINE_LONG_WAIT"

    # Test Equity Metrics
    equity = scheduler.compute_equity_metrics([p.current_wait_minutes for p in sample_patients])
    assert "max_wait" in equity
    assert "gini" in equity

    # Test Level 1 Day-Ahead MILP
    plan = scheduler.optimize_day_ahead(sample_patients[:2], sample_resources[:2])
    assert plan["status"] == "Optimal"
