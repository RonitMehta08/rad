"""Wait Time Predictor page.

Owner: P5
"""
import streamlit as st
import requests

st.set_page_config(page_title="Wait Time Predictor", page_icon="⏱️")
st.title("Wait Time Predictor")

with st.form("predict_form"):
    patient_id = st.text_input("Patient ID", "P-1001")
    modality = st.selectbox("Modality", ["xray", "ct", "mri", "ultrasound"])
    urgency = st.selectbox("Urgency", ["routine", "urgent", "emergency"])
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        res = requests.post(
            "http://localhost:8000/predict/wait-time",
            json={
                "patient_id": patient_id,
                "modality": modality,
                "urgency": urgency
            }
        )
        if res.status_code == 200:
            data = res.json()
            st.success(f"Predicted Wait Time: {data['predicted_wait_minutes']} minutes")
            st.info(data['shap_summary'])
        else:
            st.error("Error connecting to prediction API")
    except Exception as e:
        st.error(f"Failed to call API: {e}")
