# RadQueue AI — Final Results Report

_Auto-generated 2026-10-05 18:00 by `python -m src.services.report_service`._

## 1. Wait-time prediction

| Model | Split | MAE | RMSE | R² | MAPE | Median AE | P90 AE |
|---|---|---|---|---|---|---|---|
| Elastic Net | validation | 12.01 | 23.13 | 0.528 | 53.8% | 6.84 | 21.48 |
| Elastic Net | test | 11.49 | 21.05 | 0.465 | 63.8% | 6.73 | 21.75 |
| LightGBM | validation | 6.92 | 13.02 | 0.850 | 26.8% | 4.07 | 13.55 |
| LightGBM | test | 6.56 | 12.40 | 0.814 | 30.8% | 3.84 | 12.89 |
| XGBoost | validation | 6.88 | 12.99 | 0.851 | 26.7% | 4.03 | 13.35 |
| XGBoost | test | 6.60 | 12.55 | 0.810 | 30.7% | 3.80 | 12.73 |
| Stacking ensemble | validation | 6.87 | 12.94 | 0.852 | 26.6% | 4.00 | 13.24 |
| Stacking ensemble | test | 6.58 | 12.48 | 0.812 | 30.5% | 3.78 | 12.69 |

**Ensemble (test) vs targets:** MAE 6.58 ❌ (target < 5.0) · RMSE 12.48 ❌ (< 8.0) · R² 0.812 ❌ (> 0.85) · MAPE 30.5% ❌ (< 15.0%)

Literature benchmarks (MASTER_PROMPT §12.1): MAE 7–12 min, R² 0.65–0.80.

## 2. No-show prediction (scheduled appointments only)

| Split | AUC-ROC | AUC-PR | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| validation | 0.634 | 0.340 | 0.421 | 0.274 | 0.907 | 0.156 |
| test | 0.605 | 0.385 | 0.465 | 0.313 | 0.904 | 0.156 |

Targets: AUC-ROC > 0.82 ❌, F1 > 0.7 ❌.

## 3. Scheduling policy comparison (digital twin)

Mean over seeded replications; every policy sees the identical patient stream.

| Policy | Avg wait (min) | P90 wait (min) | Max wait (min) | Emergency avg wait (min) | Emergencies ≤10 min | Urgent ≤30 min | Routine >90 min | Utilisation | Patients/day |
|---|---|---|---|---|---|---|---|---|---|
| FCFS | 32.00 | 91.14 | 680.24 | 34.34 | 0.76 | 0.82 | 0.09 | 0.52 | 190.94 |
| Priority | 32.10 | 81.90 | 710.01 | 5.43 | 0.84 | 0.93 | 0.11 | 0.52 | 190.94 |
| SJF | 29.78 | 54.90 | 1072.09 | 5.18 | 0.85 | 0.86 | 0.08 | 0.52 | 190.94 |
| Wave | 32.02 | 91.59 | 680.24 | 30.78 | 0.78 | 0.84 | 0.10 | 0.52 | 190.94 |
| RadQueue AI | 32.03 | 90.54 | 686.88 | 5.36 | 0.85 | 0.88 | 0.11 | 0.52 | 190.94 |
| RadQueue + No-Show | 36.33 | 105.93 | 759.12 | 6.10 | 0.84 | 0.87 | 0.12 | 0.54 | 201.68 |

## 4. What-if scenarios (RadQueue AI policy)

| Scenario | KPI | Baseline | Scenario | Change |
|---|---|---|---|---|
| add_1_mri | Avg wait (min) | 32.03 | 18.00 | -43.8% |
| add_1_mri | P90 wait (min) | 90.54 | 62.90 | -30.5% |
| add_1_mri | Emergency avg wait (min) | 5.36 | 4.11 | -23.3% |
| add_1_mri | Patients/day | 190.94 | 190.94 | +0.0% |
| add_2_technologists | Avg wait (min) | 32.03 | 31.40 | -2.0% |
| add_2_technologists | P90 wait (min) | 90.54 | 90.03 | -0.6% |
| add_2_technologists | Emergency avg wait (min) | 5.36 | 4.74 | -11.5% |
| add_2_technologists | Patients/day | 190.94 | 190.94 | +0.0% |
| increase_emergency_buffer | Avg wait (min) | 32.03 | 32.82 | +2.5% |
| increase_emergency_buffer | P90 wait (min) | 90.54 | 90.51 | -0.0% |
| increase_emergency_buffer | Emergency avg wait (min) | 5.36 | 5.33 | -0.5% |
| increase_emergency_buffer | Patients/day | 190.94 | 190.94 | +0.0% |
| high_volume_day | Avg wait (min) | 32.03 | 88.06 | +174.9% |
| high_volume_day | P90 wait (min) | 90.54 | 332.34 | +267.1% |
| high_volume_day | Emergency avg wait (min) | 5.36 | 25.84 | +382.3% |
| high_volume_day | Patients/day | 190.94 | 293.60 | +53.8% |
| reduce_noshow | Avg wait (min) | 32.03 | 36.65 | +14.4% |
| reduce_noshow | P90 wait (min) | 90.54 | 106.61 | +17.8% |
| reduce_noshow | Emergency avg wait (min) | 5.36 | 5.49 | +2.4% |
| reduce_noshow | Patients/day | 190.94 | 202.40 | +6.0% |

## 5. Caveats

- All data is synthetic (SimPy generator calibrated to published Indian hospital figures); real-world accuracy is unverified.
- Wait-based features use only waits known at registration time (no target leakage); metrics are from a chronological hold-out (last 15% of days).
- The no-show signal in the generator is modest (lead time, history, distance, monsoon, Monday), which caps achievable AUC.
