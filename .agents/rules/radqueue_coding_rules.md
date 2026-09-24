# RadQueue AI — Workspace Coding Rules

> These rules are automatically loaded by Antigravity agents when working in this workspace.
> They supplement the global `GEMINI.md` rules with project-specific constraints.

---

## Project Identity

- **Project:** RadQueue AI — Intelligent Radiology OPD Waiting-Time Prediction & Dynamic Scheduling
- **Domain:** Healthcare Operations, Indian Hospital Radiology Departments
- **Architecture Reference:** Always read `MASTER_PROMPT.md` before making structural decisions

---

## Mandatory Constraints

### Data & Domain
- ALL synthetic data generation MUST use Indian hospital parameters (arrival rates, service times, walk-in ratios, holiday calendar) as specified in MASTER_PROMPT.md Section 6.3
- ALL patient timestamps must use IST (Indian Standard Time, UTC+5:30)
- Modality types are strictly: `xray`, `ct`, `mri`, `ultrasound` — use the `Modality` enum from `src/utils/constants.py`
- Urgency levels are strictly: `EMERGENCY`, `URGENT`, `ROUTINE` — use the `UrgencyLevel` enum
- Visit types are strictly: `scheduled`, `walk_in`, `emergency` — use the `VisitType` enum
- Hospital tiers: `tier_1` (government teaching), `tier_2` (district), `tier_3` (private multi-specialty)

### ML & Training
- ALL models must be trained with GPU acceleration on RTX 4050:
  - LightGBM: `device='gpu'`
  - XGBoost: `tree_method='gpu_hist'`
- ALL hyperparameter tuning must use Optuna with `MedianPruner` for early stopping
- Time-based train/test split ONLY — never random split for time-series operational data
- SHAP explanations required for every prediction model — both plots AND clinician-friendly text
- Target metrics (non-negotiable floors):
  - Wait Time MAE < 5 minutes
  - Wait Time R² > 0.85
  - No-Show AUC-ROC > 0.80
  - Scheduling wait reduction ≥ 30% vs FCFS baseline

### Code Structure
- Source code in `src/` — domain logic must never import Streamlit or FastAPI
- Dashboard code in `dashboard/` — may import from `src/`
- Config files in `config/` — YAML format, loaded via Pydantic models
- Generated data in `data/generated/` — never committed to git
- Model artifacts in `models/` — never committed to git
- All figures in `reports/figures/` — PNG format, 300 DPI for publication quality

### Heavy Commands
- GPU training, full simulation runs, and Optuna sweeps are HEAVY commands
- These must be provided as CLI commands for the user to run manually
- Never auto-execute training commands — always present them to the user
- Document expected runtime and output for every heavy command

### Visualization
- Use Plotly for all interactive dashboard charts
- Use Seaborn/Matplotlib for static publication-quality report figures
- Dark theme for Streamlit dashboard (premium feel)
- Every plot must have: title, axis labels, legend (where applicable), readable font sizes
- Color palette: use a consistent, accessible palette across all visualizations

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase` (e.g., `WaitTimePredictor`, `RadiologyScheduler`)
- Functions: `snake_case` (e.g., `predict_wait_time`, `compute_priority_score`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ROUTINE_WAIT_MINUTES = 90`)
- Config keys: `snake_case` in YAML files
- Test files: `test_<module_name>.py`
- Test functions: `test_should_<expected_behavior>_when_<condition>`

### Dependencies
- Pin ALL versions in `requirements.txt`
- No unnecessary dependencies — justify every new package
- Prefer stdlib or existing deps over new packages for small tasks
