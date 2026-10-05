# RadQueue AI 🏥

**Intelligent Radiology OPD Waiting-Time Prediction & AI-Based Dynamic Scheduling System**

Predicts patient waiting times and dynamically schedules radiology resources (X-ray, CT, MRI, ultrasound) to reduce OPD delays in Indian hospitals. Full design: [MASTER_PROMPT.md](MASTER_PROMPT.md). Latest numbers: [reports/final_report.md](reports/final_report.md).

## What's inside

| Layer | Module | Highlights |
|---|---|---|
| Data | `src/data/` | SimPy generator calibrated to Indian OPDs (tiers, holidays, monsoon, Monday peaks, config-driven no-shows); leakage-safe feature engineering; `feature_builder` for inference |
| ML | `src/models/` | Elastic Net / LightGBM / XGBoost (Optuna) + stacking ensemble; SHAP with clinician-language text; calibrated no-show XGBoost; PSI/KS/χ²/ADWIN/CUSUM drift monitor |
| Scheduler | `src/scheduler/` | FCFS, Priority, SJF, Wave, RadQueue AI, RadQueue + No-Show policies; day-ahead MILP (PuLP); real-time dispatcher; emergency preemption, no-show pull-forward, equipment failure, surge detection |
| Digital twin | `src/simulation/` | Config-driven SimPy department (shifts, technologists, radiologists, stochastic scan times); common-random-number policy comparisons with 95% CIs; what-if scenarios |
| Services | `src/services/` | Framework-free orchestration shared by API and dashboard (model registry, predictions, live department, monitoring, final report) |
| API | `src/api/` | FastAPI: `/predict`, `/schedule`, `/simulate`, `/monitor` |
| Dashboard | `dashboard/` | 7-page Streamlit app (dark theme, Plotly) |

## Quick start

```bash
git clone https://github.com/RonitMehta08/rad.git
cd rad
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Train everything (≈5–10 min on CPU; see [MANUAL_COMMANDS.md](MANUAL_COMMANDS.md) for GPU and full-size runs):

```bash
python -m src.data.generator --days 180
python -m src.data.preprocessor
python -m src.models.wait_time.trainer --model all --device cpu --optuna-trials 40
python -m src.models.wait_time.ensemble
python -m src.models.noshow.trainer --device cpu --optuna-trials 30
python -m src.models.wait_time.explainer --model models/wait_time/xgboost_best.pkl --data data/processed/test.parquet --output reports/figures/
python -m src.simulation.scenarios --policies all --days 5 --replications 10 --output reports/
python -m src.services.report_service
```

Run the apps (from the project root):

```bash
streamlit run dashboard/app.py --server.port 8501      # dashboard (no API server needed)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000     # API, docs at /docs
```

Run the tests:

```bash
pytest
```

Model-dependent tests are skipped automatically until models are trained.

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict/wait-time` | Prediction + empirical 90% interval + SHAP + clinician text |
| POST | `/predict/no-show` | No-show probability + overbooking recommendation |
| POST | `/schedule/patients` | Register a patient in the live department (dispatch / emergency preemption) |
| GET | `/schedule/state`, `/schedule/patients/{id}/eta` | Queues, machines, events; patient-facing ETA |
| POST | `/schedule/resources/{id}/complete`, `/failure`, `/schedule/clock/advance` | Scan completion, equipment failure, time |
| POST | `/schedule/day-ahead` | MILP day-ahead slot schedule |
| POST | `/simulate/run`, `/simulate/compare`, `/simulate/whatif` | Digital-twin runs |
| GET | `/monitor/health`, `/monitor/models`, `/monitor/drift` | Model availability, metrics, drift |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RADQUEUE_MODELS_DIR` | `models/` | Where trained artifacts are loaded from |
| `RADQUEUE_TIER` | `tier_1` | Hospital tier for the API's live department |
| `RADQUEUE_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Allowed CORS origins |

## Team & branches

| Role | Branch |
|---|---|
| P1 Data & Config Lead | `feat/data-pipeline` |
| P2 ML Engineer | `feat/ml-models` |
| P3 Scheduling Engineer | `feat/scheduler` |
| P4 Simulation Engineer | `feat/simulation` |
| P5 Dashboard & API | `feat/dashboard-api` |

Tech stack: Python 3.11 · LightGBM · XGBoost · SHAP · SimPy · PuLP · FastAPI · Streamlit · Plotly

## License

MIT
