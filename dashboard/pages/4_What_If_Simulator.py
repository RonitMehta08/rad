"""What-If Simulator page.

Owner: P5
"""
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="What-If Simulator", page_icon="🔬")
st.title("What-If Simulator")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Configure Scenario")
    policy = st.selectbox("Scheduling Policy", ["FCFS", "Priority", "SJF", "RadQueue AI"])
    days = st.slider("Simulation Days", 1, 30, 1)
    
    if st.button("Run Simulation"):
        with st.spinner(f"Running simulation with {policy} for {days} days..."):
            try:
                res = requests.post(
                    "http://localhost:8000/simulate/run",
                    json={"policy": policy, "days": days}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success("Simulation Complete")
                    st.session_state['sim_result'] = data
                else:
                    st.error("Error from Simulation API")
            except Exception as e:
                st.error(f"Failed to call API: {e}")

with col2:
    st.subheader("Results")
    if 'sim_result' in st.session_state:
        data = st.session_state['sim_result']
        st.metric("Average Wait Time", f"{data['avg_wait_minutes']} min")
        st.metric("Max Wait Time", f"{data['max_wait_minutes']} min")
        st.metric("Patients Served", data['patients_served'])
        
        df = pd.DataFrame([data])
        st.dataframe(df)
    else:
        st.info("Run a simulation to see results.")
