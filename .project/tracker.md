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
- **2026-09-30**: Initialized branch `feat/scheduler`. Implemented complete multi-tiered dynamic scheduling engine (MILP day-ahead optimizer, real-time priority dispatcher with Rawlsian fairness constraints, event-triggered rescheduler for emergency preemption, no-show pull-forward, equipment failure, and surge detection).
- **2026-10-03**: Conducted comprehensive code review audit and resolved all 15 review items:
  1. Fixed missing `Any` import in `optimizer.py`.
  2. Realigned multi-objective scoring formula and weights ($\alpha=0.40, \beta=0.25, \gamma=0.15, \delta=0.20$) in `dispatcher.py`.
  3. Corrected overbooking rate comparison vs capacity fraction in `rescheduler.py`.
  4. Optimized MILP solver complexity via O(1) patient lookups and resource-slot occupancy indexing in `optimizer.py`.
  5. Fixed active patient resolution during preemption in `rescheduler.py`.
  6. Standardized no-show predictor method calls in `rescheduler.py`.
  7. Forwarded wave and config kwargs in `RadiologyScheduler.order_by_policy`.
  8. Optimized Gini coefficient computation to O(n log n) with O(n) memory in `fairness.py`.
  9. Preserved explicit `is_preemptable` overrides in `PatientState`.
  10. Grounded `DEFAULT_CONFIG_PATH` to project root in `config.py`.
  11. Registered `radqueue_ai` and `radqueue_noshow` policies in `policies.py`.
  12. Tightened priority test assertions.
  13. Expanded unit and integration test suite to 19 test cases — all 19 tests passing.

---

## Active Feature: `feat/complete-modules` (integration & completion pass)

### Status Log
- **2026-10-05**: Review of the Phase 4–6 commit found the simulation, API and dashboard were stubs and several correctness issues. Fixed and completed:
  1. **Scheduler safety:** emergencies now form a strict top tier in `RadQueueAIPolicy` and `RealTimeDispatcher.rank_queue` (the uncapped fairness bonus let 60-min urgent / 70-min routine patients outrank a new emergency). Regression test added.
  2. **Closed loop:** the dispatcher now passes real model feature names to the wait-time predictor (`build_scheduler_context` + `src/data/feature_builder.py`); predictor failures are logged as warnings.
  3. **Target leakage removed:** rolling/lag wait features only use patients whose imaging started before the registration; outlier cap and missing-value fill use training statistics only. Test R² is honestly ~0.81 (was an inflated 0.87).
  4. **No-show realism:** the generator applies the config's lead-time / monsoon / Monday rules plus history and distance; no-shows carry the real queue state; the model trains on scheduled appointments only with a validation-tuned threshold (F1 0.47 vs 0.0).
  5. **Digital twin rewritten:** config-driven tiers, shifts and stochastic service times; policy decides on every free machine; common random numbers + replications + 95% CIs; what-if presets; overbooking for RadQueue + No-Show.
  6. **API** (`src/api`) and **dashboard** (7 pages) now call real models, scheduler and simulation through the framework-free `src/services` layer.
  7. Trainers save test metrics, residual-quantile prediction intervals and feature encoders; `reports/final_report.md` is auto-generated.
  8. Tests: 19 → 50 (API, generator, preprocessor leakage, simulation, models). Requirements pinned to verified versions.
- **Open items:** RadQueue AI's within-modality ordering is effectively Priority + aging, so SJF still has the lowest average/P90 wait. Adding a duration-aware term is the natural next step for P3. Other open items: GPU training runs with 100 trials, notebooks, MLflow.
