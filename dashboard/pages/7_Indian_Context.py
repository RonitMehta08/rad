"""Indian Context Configuration page.

Owner: P5
"""
import streamlit as st
import yaml

st.set_page_config(page_title="Indian Context Config", page_icon="🇮🇳", layout="wide")
st.title("Indian Context Configuration")

st.markdown("Customize simulation parameters for Indian hospital contexts.")

with st.form("context_form"):
    hospital_tier = st.selectbox("Hospital Tier", ["Tier 1 (Govt Teaching)", "Tier 2 (District)", "Tier 3 (Private)"])
    walk_in_ratio = st.slider("Walk-in Ratio (%)", 0, 100, 70)
    monsoon_multiplier = st.slider("Monsoon Emergency Multiplier", 1.0, 3.0, 1.5)
    
    st.markdown("### Active Holidays")
    holi = st.checkbox("Holi", value=True)
    diwali = st.checkbox("Diwali", value=True)
    independence_day = st.checkbox("Independence Day", value=True)
    
    submitted = st.form_submit_button("Save Configuration")

if submitted:
    st.success("Context configuration saved successfully! (Simulated)")
