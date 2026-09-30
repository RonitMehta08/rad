# Dynamic Scheduling Engine Implementation Plan (`feat/scheduler`)

> **Role:** P3 (Scheduling Engineer)  
> **Domain:** Module 3: Dynamic Scheduler (Optimizer) & Phase 3 Deliverables  
> **Reference:** `MASTER_PROMPT.md` §5.3, §7.3, §14, §15; `config/scheduler_config.yaml`; `.agents/rules/radqueue_coding_rules.md`

**Goal:** Build the complete, multi-tiered dynamic radiology scheduling system that minimizes patient waiting times and starvation while maximizing machine utilization across modalities (X-ray, CT, MRI, Ultrasound).

**Architecture:**
- **Level 1 (Day-Ahead MILP Optimizer):** Solves slot assignment via PuLP (CBC solver) across machine and technologist shifts, reserving 10% emergency buffer with fairness bounds.
- **Level 2 (Real-Time Dynamic Dispatch):** Priority queue scoring combining urgency, dynamic wait fairness bonus, and modality factors, assigning resources using multi-objective optimization.
- **Level 3 (Event-Driven Rescheduler):** Handles emergency preemption, no-show pull-forward & overbooking, machine breakdown queue redistribution, and surge overflow.
- **Fairness & Comparative Policies:** Implements FCFS, Priority, SJF, Wave, and RadQueue AI with Rawlsian fairness constraints (max routine wait <= 90m, urgent <= 30m, emergency <= 10m).

**Tech Stack:** Python 3.12, PuLP 3.3.2, Pydantic 2.12.5, PyYAML, NumPy, Pandas, Pytest.

---

## Detailed Task Breakdown

### Task 1: Domain Entities & Config Loader
- **Files:** `src/scheduler/models.py`, `src/scheduler/config.py`
- **Responsibilities:**
  - Define Pydantic models for `PatientState`, `ResourceSlot`, `Assignment`, `DepartmentState`, and `QueueSnapshot`.
  - Load `config/scheduler_config.yaml` with typed validations.

### Task 2: Comparative Scheduling Policies
- **Files:** `src/scheduler/policies.py`
- **Responsibilities:**
  - `FCFSPolicy`: First-come, first-served benchmark.
  - `PriorityPolicy`: Strict clinical priority (Emergency > Urgent > Routine).
  - `SJFPolicy`: Shortest Job First based on modality median duration.
  - `WavePolicy`: 30-minute interval batching.
  - Policy protocol / abstract base class for drop-in dispatching.

### Task 3: Rawlsian Fairness & Starvation Engine
- **Files:** `src/scheduler/fairness.py`
- **Responsibilities:**
  - Dynamic fairness penalty calculation ($+0.1$ priority per minute over avg wait).
  - Max wait ceiling enforcement (Routine: 90m, Urgent: 30m, Emergency: 10m).
  - Starvation detection and alerting (threshold: 60m).
  - Rawlsian objective evaluation (optimizing for the worst-off patient in queue).

### Task 4: Level 2 Real-Time Priority Dispatcher
- **Files:** `src/scheduler/dispatcher.py`
- **Responsibilities:**
  - Dynamic priority scoring function: $\text{Base} + \text{WaitBonus} + \text{ModalityFactor}$.
  - Multi-objective greedy resource assignment ($\alpha \cdot \text{wait} + \beta \cdot \text{cascade} + \gamma \cdot (1 - \text{util})$).
  - Resource availability matching and queue state evaluation.

### Task 5: Level 1 Day-Ahead MILP Optimizer
- **Files:** `src/scheduler/optimizer.py`
- **Responsibilities:**
  - MILP model formulation via `PuLP`.
  - Variables: $x_{p, s, m} \in \{0, 1\}$ (patient $p$ to slot $s$ on machine $m$).
  - Machine capacity, technologist availability, emergency buffer reservation (10%), prep time, and max wait constraints.
  - Objective: minimize total wait + max wait + wait variance - utilization.

### Task 6: Level 3 Dynamic Event Rescheduler
- **Files:** `src/scheduler/rescheduler.py`
- **Responsibilities:**
  - `handle_emergency(emergency_patient)`: preempt routine prep, cascade reschedule displaced patients.
  - `handle_noshow(slot)`: detect no-show after grace period (15m), pull forward next queued patient, manage controlled overbooking.
  - `handle_equipment_failure(machine_id)`: redistribute affected modality queue.
  - `handle_surge(current_arrival_rate)`: overflow protocol when arrival rate $> \mu + 2\sigma$.

### Task 7: Unified Facade & Package Exports
- **Files:** `src/scheduler/__init__.py`
- **Responsibilities:**
  - Unified `RadiologyScheduler` class combining Levels 1, 2, 3 and comparison policies.
  - Clean API for P4 (Simulation) and P5 (FastAPI / Dashboard).

### Task 8: Comprehensive Test Suite
- **Files:** `tests/test_scheduler.py`
- **Responsibilities:**
  - Unit tests for policies (FCFS, Priority, SJF, Wave).
  - Unit tests for fairness and anti-starvation scoring.
  - Unit tests for MILP optimizer feasibility and solution extraction.
  - Unit tests for dynamic dispatching and preemption / no-show triggers.
  - Target: 100% passing tests with pytest.
