"""Main Streamlit App.

Owner: P5
"""
import streamlit as st

st.set_page_config(
    page_title="RadQueue AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RadQueue AI")
st.markdown("""
Welcome to **RadQueue AI**, the intelligent Radiology OPD Waiting-Time Prediction & AI-Based Dynamic Scheduling System.

Please select a tool from the sidebar to continue:
- **Command Center**: Real-time KPI and queue monitoring.
- **Wait Time Predictor**: Predict patient wait time.
- **Smart Scheduler**: Dynamic scheduling and resource allocation.
- **What-If Simulator**: Run scenario tests on scheduling policies.
""")

st.info("System is connected to FastAPI backend.")
