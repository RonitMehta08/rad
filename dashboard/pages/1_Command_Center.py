"""Command Center page.

Owner: P5
"""
import streamlit as st

st.set_page_config(page_title="Command Center", page_icon="📊", layout="wide")
st.title("Command Center")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", "142", "+12")
col2.metric("Avg Wait Time", "42 min", "-5 min")
col3.metric("MRI Utilization", "95%", "+2%")
col4.metric("Emergency Queue", "2", "-1")

st.subheader("Live Queue Status")
st.bar_chart({"X-Ray": 15, "CT": 8, "MRI": 4, "Ultrasound": 12})
