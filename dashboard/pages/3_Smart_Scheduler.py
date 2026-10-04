"""Smart Scheduler page.

Owner: P5
"""
import streamlit as st
import requests

st.set_page_config(page_title="Smart Scheduler", page_icon="📅")
st.title("Smart Scheduler")

st.markdown("Assign a patient dynamically:")
with st.form("schedule_form"):
    patient_id = st.text_input("Patient ID", "P-1002")
    modality = st.selectbox("Modality", ["xray", "ct", "mri", "ultrasound"])
    urgency = st.selectbox("Urgency", ["routine", "urgent", "emergency"])
    submitted = st.form_submit_button("Assign")

if submitted:
    try:
        res = requests.post(
            "http://localhost:8000/schedule/assign",
            json={
                "patient_id": patient_id,
                "modality": modality,
                "urgency": urgency
            }
        )
        if res.status_code == 200:
            data = res.json()
            st.success(f"Assigned to: {data['assigned_machine_id']}")
            st.info(f"Predicted Wait: {data['predicted_wait_minutes']} mins")
        else:
            st.error("Error connecting to scheduling API")
    except Exception as e:
        st.error(f"Failed to call API: {e}")
