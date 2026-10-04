"""Analytics page.

Owner: P5
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analytics & Reports", page_icon="📈", layout="wide")
st.title("Analytics & Reports")

st.markdown("### Historical Wait Time Trends")
# Generate dummy data for trends
dates = pd.date_range(start="2024-01-01", periods=30)
wait_times = np.random.normal(loc=40, scale=10, size=30)
df_trends = pd.DataFrame({"Date": dates, "Average Wait (min)": wait_times})
st.line_chart(df_trends.set_index("Date"))

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Modality-wise Wait Times")
    # Dummy data
    modality_data = pd.DataFrame({
        "Modality": ["X-Ray", "CT", "MRI", "Ultrasound"],
        "Avg Wait (min)": [15, 30, 60, 20]
    })
    st.bar_chart(modality_data.set_index("Modality"))

with col2:
    st.markdown("### Resource Utilization")
    util_data = pd.DataFrame({
        "Resource": ["X-Ray", "CT", "MRI", "Ultrasound", "Radiologists"],
        "Utilization (%)": [75, 82, 95, 60, 88]
    })
    st.bar_chart(util_data.set_index("Resource"))
