"""Generate reports/final_report.md from trained-model metadata and simulation results.

Owner: P5 (Dashboard & API)
Reference: MASTER_PROMPT §12, §15 Phase 6

Usage:
    python -m src.services.report_service
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src.services.model_registry import PROJECT_ROOT
from src.utils.constants import (
    TARGET_MAE_MINUTES,
    TARGET_MAPE_PERCENT,
    TARGET_NOSHOW_AUC_ROC,
    TARGET_NOSHOW_F1,
    TARGET_R_SQUARED,
    TARGET_RMSE_MINUTES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
WAIT_MODELS = {
    "elastic_net_best_metadata.json": "Elastic Net",
    "lightgbm_best_metadata.json": "LightGBM",
    "xgboost_best_metadata.json": "XGBoost",
    "ensemble_best.json": "Stacking ensemble",
}
SIM_KPIS = {
    "avg_wait": "Avg wait (min)", "p90_wait": "P90 wait (min)", "max_wait": "Max wait (min)",
    "emergency_avg_wait": "Emergency avg wait (min)", "emergency_within_limit_rate": "Emergencies ≤10 min",
    "urgent_within_limit_rate": "Urgent ≤30 min", "starvation_rate": "Routine >90 min",
    "utilization": "Utilisation", "patients_per_day": "Patients/day",
}


def _check(value: float, target: float, higher_is_better: bool) -> str:
    ok = value >= target if higher_is_better else value <= target
    return "✅" if ok else "❌"


def _wait_time_section() -> list[str]:
    lines = ["## 1. Wait-time prediction", "",
             "| Model | Split | MAE | RMSE | R² | MAPE | Median AE | P90 AE |", "|---|---|---|---|---|---|---|---|"]
    for file, label in WAIT_MODELS.items():
        path = MODELS_DIR / "wait_time" / file
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        for split, key in (("validation", "metrics"), ("test", "test_metrics")):
            m = meta.get(key)
            if m:
                lines.append(f"| {label} | {split} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r_squared']:.3f} | "
                             f"{m['mape']:.1f}% | {m['median_ae']:.2f} | {m['p90_ae']:.2f} |")
    ens = MODELS_DIR / "wait_time" / "ensemble_best.json"
    if ens.exists():
        t = json.loads(ens.read_text(encoding="utf-8")).get("test_metrics", {})
        if t:
            lines += ["", "**Ensemble (test) vs targets:** "
                      f"MAE {t['mae']:.2f} {_check(t['mae'], TARGET_MAE_MINUTES, False)} (target < {TARGET_MAE_MINUTES}) · "
                      f"RMSE {t['rmse']:.2f} {_check(t['rmse'], TARGET_RMSE_MINUTES, False)} (< {TARGET_RMSE_MINUTES}) · "
                      f"R² {t['r_squared']:.3f} {_check(t['r_squared'], TARGET_R_SQUARED, True)} (> {TARGET_R_SQUARED}) · "
                      f"MAPE {t['mape']:.1f}% {_check(t['mape'], TARGET_MAPE_PERCENT, False)} (< {TARGET_MAPE_PERCENT}%)",
                      "", "Literature benchmarks (MASTER_PROMPT §12.1): MAE 7–12 min, R² 0.65–0.80.", ""]
    return lines


def _noshow_section() -> list[str]:
    path = MODELS_DIR / "noshow" / "noshow_metadata.json"
    if not path.exists():
        return []
    meta = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## 2. No-show prediction (scheduled appointments only)", "",
             "| Split | AUC-ROC | AUC-PR | F1 | Precision | Recall | Threshold |", "|---|---|---|---|---|---|---|"]
    for split, key in (("validation", "metrics"), ("test", "test_metrics")):
        m = meta.get(key)
        if m:
            lines.append(f"| {split} | {m['auc_roc']:.3f} | {m['auc_pr']:.3f} | {m['f1']:.3f} | "
                         f"{m['precision']:.3f} | {m['recall']:.3f} | {m['threshold_used']:.3f} |")
    t = meta.get("test_metrics", meta["metrics"])
    lines += ["", f"Targets: AUC-ROC > {TARGET_NOSHOW_AUC_ROC} {_check(t['auc_roc'], TARGET_NOSHOW_AUC_ROC, True)}, "
              f"F1 > {TARGET_NOSHOW_F1} {_check(t['f1'], TARGET_NOSHOW_F1, True)}.", ""]
    return lines


def _simulation_section() -> list[str]:
    lines: list[str] = []
    comp = REPORTS_DIR / "simulation_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        kpis = [k for k in SIM_KPIS if k in df.columns]
        lines += ["## 3. Scheduling policy comparison (digital twin)", "",
                  "Mean over seeded replications; every policy sees the identical patient stream.", "",
                  "| Policy | " + " | ".join(SIM_KPIS[k] for k in kpis) + " |",
                  "|---|" + "---|" * len(kpis)]
        for _, row in df.iterrows():
            lines.append(f"| {row['policy_name']} | " + " | ".join(f"{row[k]:.2f}" for k in kpis) + " |")
        lines.append("")
    whatif = REPORTS_DIR / "whatif_comparison.csv"
    if whatif.exists():
        w = pd.read_csv(whatif)
        w = w[w["kpi"].isin(["avg_wait", "p90_wait", "emergency_avg_wait", "patients_per_day"])]
        lines += ["## 4. What-if scenarios (RadQueue AI policy)", "",
                  "| Scenario | KPI | Baseline | Scenario | Change |", "|---|---|---|---|---|"]
        for _, r in w.iterrows():
            lines.append(f"| {r['scenario_name']} | {SIM_KPIS.get(r['kpi'], r['kpi'])} | {r['baseline']:.2f} | "
                         f"{r['scenario']:.2f} | {r['delta_pct']:+.1f}% |")
        lines.append("")
    return lines


def build_final_report() -> str:
    """Assemble the markdown report from on-disk artifacts."""
    lines = [
        "# RadQueue AI — Final Results Report", "",
        f"_Auto-generated {datetime.now():%Y-%m-%d %H:%M} by `python -m src.services.report_service`._", "",
    ]
    lines += _wait_time_section() + _noshow_section() + _simulation_section()
    lines += [
        "## 5. Caveats", "",
        "- All data is synthetic (SimPy generator calibrated to published Indian hospital figures); "
        "real-world accuracy is unverified.",
        "- Wait-based features use only waits known at registration time (no target leakage); "
        "metrics are from a chronological hold-out (last 15% of days).",
        "- The no-show signal in the generator is modest (lead time, history, distance, monsoon, Monday), "
        "which caps achievable AUC.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write reports/final_report.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "final_report.md"
    path.write_text(build_final_report(), encoding="utf-8")
    logger.info(f"Final report written to {path}")


if __name__ == "__main__":
    main()
