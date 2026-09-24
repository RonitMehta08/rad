# RadQueue AI 🏥

**Intelligent Radiology OPD Waiting-Time Prediction & AI-Based Dynamic Scheduling System**

> Predict patient waiting times and dynamically schedule radiology resources to minimize OPD delays in Indian hospitals.

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/radqueue-ai.git
cd radqueue-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python -m src.data.generator --config config/simulation_config.yaml --output data/generated/ --days 180
```

### 3. Run Feature Engineering
```bash
python -m src.data.preprocessor --input data/generated/patients.csv --output data/processed/
```

### 4. Train Models (GPU required — RTX 4050)
```bash
python -m src.models.wait_time.trainer --model lightgbm --device gpu --optuna-trials 100
python -m src.models.wait_time.trainer --model xgboost --device gpu --optuna-trials 100
python -m src.models.wait_time.ensemble --data data/processed/ --models models/wait_time/
python -m src.models.noshow.trainer --device gpu --optuna-trials 50
```

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```

### 6. Launch API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

See [MASTER_PROMPT.md](MASTER_PROMPT.md) §14 for the full directory tree.

## Team & Branches

| Role | Branch | Owner |
|------|--------|-------|
| Data & Config Lead | `feat/data-pipeline` | P1 |
| ML Engineer | `feat/ml-models` | P2 |
| Scheduling Engineer | `feat/scheduler` | P3 |
| Simulation Engineer | `feat/simulation` | P4 |
| Dashboard & API | `feat/dashboard-api` | P5 |

## Tech Stack

Python 3.11+ · LightGBM · XGBoost · SimPy · PuLP · FastAPI · Streamlit · Plotly · SHAP

## License

MIT
