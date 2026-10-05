"""Wait Time Predictor — ML prediction, interval, SHAP and no-show risk.

Owner: P5 (Dashboard & API)
"""

from datetime import datetime, time

import _bootstrap  # noqa: F401
import streamlit as st

from dashboard.components.charts import shap_bar
from dashboard.components.data import get_registry, require_models
from src.services.prediction_service import predict_noshow, predict_wait_time

st.set_page_config(page_title="Wait Time Predictor", page_icon="⏱️", layout="wide")
st.title("⏱️ Wait Time Predictor")

registry = get_registry()
if not require_models(registry, "wait_time", "explainer"):
    st.stop()

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Patient**")
        modality = st.selectbox("Modality", ["xray", "ct", "mri", "ultrasound"], index=2)
        urgency = st.selectbox("Urgency", ["routine", "urgent", "emergency"])
        visit_type = st.selectbox("Visit type", ["walk_in", "scheduled", "emergency"])
        age_group = st.selectbox("Age group", ["0-18", "18-40", "40-60", "60+"], index=1)
        contrast = st.checkbox("Requires contrast")
        prep = st.checkbox("Requires preparation")
    with c2:
        st.markdown("**Appointment**")
        visit_date = st.date_input("Date", datetime(2024, 7, 15))
        visit_time = st.time_input("Time", time(10, 30))
        lead_time = st.number_input("Booking lead time (days)", 0, 60, 0)
        prev_noshow = st.number_input("Previous no-shows", 0, 20, 0)
        distance = st.selectbox("Distance", ["local", "city", "outstation"])
    with c3:
        st.markdown("**Department right now**")
        q_same = st.number_input("Queue (same modality)", 0, 100, 6)
        q_total = st.number_input("Queue (all modalities)", 0, 300, 15)
        machines = st.number_input("Machines for this modality", 0, 10, 1)
        in_service = st.number_input("Patients being scanned", 0, 20, 4)
        recent_wait = st.number_input("Avg wait of last 30 min (min)", 0.0, 300.0, 30.0)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    raw = {
        "modality": modality, "urgency": urgency, "visit_type": visit_type, "age_group": age_group,
        "requires_contrast": contrast, "requires_prep": prep, "appointment_lead_time_days": lead_time,
        "previous_no_show_count": prev_noshow, "distance_category": distance,
        "timestamp": datetime.combine(visit_date, visit_time).isoformat(),
        "current_queue_length_same_modality": q_same, "current_queue_length_total": q_total,
        "num_machines_available": machines, "patients_in_service_count": in_service,
        "rolling_avg_wait_30min": recent_wait,
    }
    result = predict_wait_time(registry, raw)
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted wait", f"{result['predicted_wait_minutes']:.0f} min")
        st.caption(f"90% interval: {result['lower_bound_minutes']:.0f}–{result['upper_bound_minutes']:.0f} min "
                   f"({result['interval_method'].replace('_', ' ')}) · model: {result['model_name']}")
        st.info(result["explanation"]["clinician_text"])
    with right:
        st.plotly_chart(shap_bar(result["explanation"]["contributions"]), use_container_width=True)

    if visit_type == "scheduled" and registry.model_status()["noshow"]:
        ns = predict_noshow(registry, raw)
        st.subheader("No-show risk")
        c1, c2 = st.columns(2)
        c1.metric("No-show probability", f"{100 * ns['no_show_probability']:.0f}%")
        c2.metric("Overbook this slot?", "Yes" if ns["should_overbook"] else "No")
        st.caption(ns["reason"])

ens = registry.model_metadata().get("ensemble_best", {})
if "test_metrics" in ens:
    t = ens["test_metrics"]
    st.caption(f"Held-out test accuracy: MAE {t['mae']:.1f} min, median error {t['median_ae']:.1f} min, "
               f"90% of errors under {t['p90_ae']:.0f} min.")
