# Manual Commands (heavy / long-running)

All commands run from the project root inside the virtual environment. Times are for a laptop CPU unless stated; GPU flags apply to the RTX 4050 (MASTER_PROMPT §11).

| # | Step | Command | Time | Output |
|---|---|---|---|---|
| 1 | Install | `pip install -r requirements.txt` | 3–5 min | — |
| 2 | Generate data | `python -m src.data.generator --days 180 --tier tier_1` | ~5 s | `data/generated/patients.csv` (~31K rows; use `--days 300` for 50K+) |
| 3 | Features | `python -m src.data.preprocessor` | ~5 s | `data/processed/{train,val,test}.parquet`, `feature_encoders.json` |
| 4 | Baseline | `python -m src.models.wait_time.trainer --model elastic_net` | ~1 min | `models/wait_time/elastic_net_best.*` |
| 5 | LightGBM | `python -m src.models.wait_time.trainer --model lightgbm --device gpu --optuna-trials 100` | 30–60 min GPU · ~2 min CPU @40 trials | `lightgbm_best.pkl` + metadata (val + test metrics, residual quantiles) |
| 6 | XGBoost | `python -m src.models.wait_time.trainer --model xgboost --device gpu --optuna-trials 100` | 45–90 min GPU · ~2 min CPU @40 trials | `xgboost_best.pkl` + metadata |
| 7 | Ensemble | `python -m src.models.wait_time.ensemble` | ~1 min | `ensemble_best.pkl/.json` |
| 8 | No-show | `python -m src.models.noshow.trainer --device gpu --optuna-trials 50` | 5–15 min | `xgboost_noshow.pkl`, `calibrator.pkl`, metadata with tuned threshold |
| 9 | SHAP | `python -m src.models.wait_time.explainer --model models/wait_time/xgboost_best.pkl --data data/processed/test.parquet --output reports/figures/` | 1–5 min | SHAP plots + `feature_importance.json` |
| 10 | Test-set evaluation | `python -m src.models.wait_time.evaluator --models models/ --data data/processed/test.parquet --output reports/` | ~1 min | `reports/evaluation_results.json`, figures |
| 11 | Digital twin | `python -m src.simulation.scenarios --policies all --days 5 --replications 10 --output reports/` | ~30 s | `simulation_comparison.csv`, `simulation_runs.csv`, `whatif_comparison.csv`, `figures/scheduling_comparison.png` |
| 12 | Final report | `python -m src.services.report_service` | instant | `reports/final_report.md` |
| 13 | Dashboard | `streamlit run dashboard/app.py --server.port 8501` | — | http://localhost:8501 |
| 14 | API | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` | — | http://localhost:8000/docs |

Use `--device cpu` on machines without a CUDA GPU. Steps 5–8 must be rerun whenever step 3 changes, because the encoders and feature definitions must match the models.
