"""Model Performance & Drift Monitoring.

Owner: P5 (Dashboard & API) on top of P2's evaluation/monitoring
"""

from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.data import REPORTS_DIR, get_registry, load_json
from src.services.model_registry import ModelRegistry
from src.services.monitoring_service import rolling_accuracy, run_drift_report
from src.utils.constants import (
    TARGET_MAE_MINUTES,
    TARGET_MAPE_PERCENT,
    TARGET_NOSHOW_AUC_ROC,
    TARGET_R_SQUARED,
    TARGET_RMSE_MINUTES,
)
from src.utils.exceptions import RadQueueError
from src.utils.plotting import apply_radqueue_theme

st.set_page_config(page_title="Model Performance", page_icon="🧠", layout="wide")
st.title("🧠 Model Performance & Drift Monitoring")

registry = get_registry()
meta = registry.model_metadata()


@st.cache_data(show_spinner="Scoring test period…")
def cached_rolling_accuracy(_registry: ModelRegistry) -> pd.DataFrame:  # "_" arg: not hashed by Streamlit
    return rolling_accuracy(_registry)


@st.cache_data(show_spinner="Running PSI / KS / ADWIN / CUSUM…")
def cached_drift_report(_registry: ModelRegistry) -> dict[str, Any]:
    return run_drift_report(_registry)


if not meta:
    st.warning("No model metadata found in models/. Train the models first (MANUAL_COMMANDS.md).")
    st.stop()

st.subheader("Wait-time models (validation vs held-out test)")
rows = []
for name, m in meta.items():
    if name == "noshow" or "metrics" not in m:
        continue
    for split, key in (("validation", "metrics"), ("test", "test_metrics")):
        if key in m:
            rows.append({"model": name, "split": split, **{k: round(v, 3) for k, v in m[key].items()}})
table = pd.DataFrame(rows)
st.dataframe(table, hide_index=True, use_container_width=True)
st.caption(f"Targets: MAE < {TARGET_MAE_MINUTES} min, RMSE < {TARGET_RMSE_MINUTES} min, R² > {TARGET_R_SQUARED}, "
           f"MAPE < {TARGET_MAPE_PERCENT}%. Literature benchmarks: MAE 7–12 min, R² 0.65–0.80.")

if "noshow" in meta:
    st.subheader("No-show model (scheduled appointments)")
    ns = meta["noshow"]
    cols = st.columns(4)
    test = ns.get("test_metrics", ns["metrics"])
    cols[0].metric("AUC-ROC (test)", f"{test['auc_roc']:.2f}", f"target {TARGET_NOSHOW_AUC_ROC}", delta_color="off")
    cols[1].metric("AUC-PR (test)", f"{test['auc_pr']:.2f}")
    cols[2].metric("F1 (test)", f"{test['f1']:.2f}")
    cols[3].metric("Decision threshold", f"{ns.get('decision_threshold', test['threshold_used']):.2f}")

importance = load_json(REPORTS_DIR / "figures" / "feature_importance.json")
if importance:
    st.subheader("Global feature importance (mean |SHAP|, minutes)")
    imp = pd.DataFrame(importance).iloc[::-1]
    fig = px.bar(imp, x="mean_abs_shap", y="display_name", orientation="h", height=480,
                 labels={"mean_abs_shap": "Average impact on prediction (min)", "display_name": ""})
    st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)

st.subheader("Monitoring")
try:
    acc = cached_rolling_accuracy(registry)
    fig = px.line(acc, x="date", y="mae", title="Daily MAE on the held-out period (min)",
                  labels={"date": "Date", "mae": "MAE (min)"})
    fig.add_hline(y=TARGET_MAE_MINUTES, line_dash="dash", annotation_text="target")
    st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)

    report = cached_drift_report(registry)
    status = report["overall_status"]
    (st.error if status == "critical" else st.warning if status == "warning" else st.success)(
        f"Drift status: **{status}** — {report['recommendation']} "
        f"({report['features_drifted']}/{report['total_features_tested']} tests flagged)"
    )
    drift = pd.DataFrame(report["drift_results"])
    st.dataframe(drift[["feature_name", "test_name", "statistic", "p_value", "severity", "details"]],
                 hide_index=True, use_container_width=True)
except RadQueueError as exc:
    st.info(f"Monitoring unavailable: {exc}")
