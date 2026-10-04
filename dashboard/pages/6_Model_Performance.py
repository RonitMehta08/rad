"""Model Performance & Drift Monitoring page.

Owner: P5
"""
import streamlit as st

st.set_page_config(page_title="Model Performance", page_icon="🧠", layout="wide")
st.title("Model Performance & Drift Monitoring")

st.markdown("### Prediction Accuracy Tracker")
col1, col2, col3 = st.columns(3)
col1.metric("Current MAE", "4.2 min", "-0.3 min")
col2.metric("Current RMSE", "7.1 min", "-0.5 min")
col3.metric("R-squared", "0.88", "+0.02")

st.markdown("---")

st.markdown("### Data Drift Alerts")
drift_data = [
    {"Feature": "arrival_rate_last_15min", "PSI": 0.05, "Status": "Normal"},
    {"Feature": "current_queue_length_total", "PSI": 0.08, "Status": "Normal"},
    {"Feature": "visit_type", "PSI": 0.22, "Status": "Warning (Drift Detected)"},
]
st.table(drift_data)

if st.button("Retrain Model"):
    st.info("Retraining pipeline triggered. This may take 30-60 minutes.")
