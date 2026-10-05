"""RadQueue AI — Streamlit dashboard home.

Owner: P5 (Dashboard & API)

Run from the project root:
    streamlit run dashboard/app.py --server.port 8501
"""

import _bootstrap  # noqa: F401  (adds project root to sys.path)
import streamlit as st

from dashboard.components.data import get_registry

st.set_page_config(page_title="RadQueue AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.title("🏥 RadQueue AI")
st.caption("Intelligent Radiology OPD wait-time prediction & AI-based dynamic scheduling for Indian hospitals")

registry = get_registry()
status = registry.model_status()
metadata = registry.model_metadata()

cols = st.columns(3)
for col, (name, ok) in zip(cols, status.items()):
    col.metric(f"{name.replace('_', ' ').title()} model", "Ready" if ok else "Not trained")
if not all(status.values()):
    st.warning("Some models are not trained yet — prediction pages will be limited. See MANUAL_COMMANDS.md.")

ens = metadata.get("ensemble_best", {}).get("test_metrics")
noshow = metadata.get("noshow", {}).get("test_metrics")
if ens or noshow:
    st.subheader("Held-out test performance")
    items = []
    if ens:
        items += [("Wait-time MAE", f"{ens['mae']:.1f} min"), ("Wait-time R²", f"{ens['r_squared']:.2f}")]
    if noshow:
        items += [("No-show AUC-ROC", f"{noshow['auc_roc']:.2f}"), ("No-show F1", f"{noshow['f1']:.2f}")]
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)

st.markdown("""
### Pages
| Page | What it does |
|---|---|
| **Command Center** | Digital-twin replay of a full OPD day: KPIs, live queues, utilisation gauges, patient-flow Sankey, alerts |
| **Wait Time Predictor** | ML prediction with a 90% interval, SHAP explanation in clinician language, no-show risk |
| **Smart Scheduler** | Live dispatch (emergency preemption, equipment failure, ETAs) and day-ahead MILP Gantt |
| **What-If Simulator** | Policy comparison and staffing/equipment scenarios on identical patient streams |
| **Analytics** | Trends, modality distributions, peak-hour heatmap, no-show drivers from the dataset |
| **Model Performance** | Validation vs test metrics against targets, SHAP importance, drift monitoring |
| **Indian Context** | Hospital tiers, holiday/monsoon demand calendar, tier-specific simulation |
""")
