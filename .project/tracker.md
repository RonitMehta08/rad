# Project Tracker — RadQueue AI

## Active Feature: `feat/scheduler` (P3: Scheduling Engineer)

### Target Deliverables (Phase 3: Scheduling Engine)
- [x] Task 1: Scaffolding, Domain Entities & Config Loader (`src/scheduler/models.py`, `src/scheduler/config.py`)
- [x] Task 2: Baseline & Comparative Scheduling Policies (`src/scheduler/policies.py`)
- [x] Task 3: Rawlsian Fairness Constraints & Anti-Starvation (`src/scheduler/fairness.py`)
- [x] Task 4: Level 2 Real-Time Priority Dispatcher (`src/scheduler/dispatcher.py`)
- [x] Task 5: Level 1 Day-Ahead MILP Optimizer (`src/scheduler/optimizer.py`)
- [x] Task 6: Level 3 Dynamic Event Rescheduler (`src/scheduler/rescheduler.py`)
- [x] Task 7: Unified `RadiologyScheduler` Interface & Package Exports (`src/scheduler/__init__.py`)
- [x] Task 8: Comprehensive Test Suite & Benchmark Verification (`tests/test_scheduler.py`)

### Status Log
- **2026-09-30**: Initialized branch `feat/scheduler`. Implemented complete multi-tiered dynamic scheduling engine (MILP day-ahead optimizer, real-time priority dispatcher with Rawlsian fairness constraints, event-triggered rescheduler for emergency preemption, no-show pull-forward, equipment failure, and surge detection). Created 15 comprehensive unit & integration tests in `tests/test_scheduler.py` — all 15 tests passing.
