"""Command Center — digital-twin replay of one OPD day.

Owner: P5 (Dashboard & API)
"""

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import (
    CHART_MARGIN,
    clock,
    kpi_row,
    patient_flow_sankey,
    queue_over_time,
    utilization_gauges,
)
from dashboard.components.data import simulate_day
from src.simulation.scenarios import POLICY_DISPLAY_NAMES
from src.utils.plotting import MODALITY_COLORS, apply_radqueue_theme

OPEN_MINUTE, CLOSE_MINUTE = 8 * 60, 20 * 60

st.set_page_config(page_title="Command Center", page_icon="📊", layout="wide")
st.title("📊 Command Center")

with st.sidebar:
    tier = st.selectbox("Hospital tier", ["tier_1", "tier_2", "tier_3"])
    policy = st.selectbox("Scheduling policy", list(POLICY_DISPLAY_NAMES), index=4,
                          format_func=POLICY_DISPLAY_NAMES.get)
    seed = st.number_input("Day (seed)", 1, 999, 42)

patients, samples, summary = simulate_day(policy, tier, 1, int(seed))
if patients.empty:
    st.error("No patients were simulated for this day (e.g. holiday with zero demand).")
    st.stop()

now = st.select_slider("Replay time", options=list(range(OPEN_MINUTE, CLOSE_MINUTE + 1, 15)), value=11 * 60,
                       format_func=clock)
st.caption(f"Department state at **{clock(now)}**")

present = patients[(patients["arrival_time"] <= now) & (patients["departure_time"] > now)]
waiting = patients[(patients["queue_entry_time"] <= now) & (patients["scan_start_time"] > now)]
recent = patients[(patients["scan_start_time"] <= now) & (patients["scan_start_time"] > now - 60)]
kpi_row([
    ("Patients in department", f"{len(present)}", None),
    ("Waiting for a machine", f"{len(waiting)}", None),
    ("Avg wait (last hour)", f"{recent['wait_time_minutes'].mean():.0f} min" if len(recent) else "—", None),
    ("Emergencies waiting", f"{int((waiting['urgency'] == 'emergency').sum())}", None),
])

alerts = []
if summary.get("emergency_within_limit_rate", 1.0) < 1.0:
    alerts.append(f"{100 * (1 - summary['emergency_within_limit_rate']):.0f}% of emergencies waited > 10 min today")
if summary.get("starvation_rate", 0.0) > 0:
    alerts.append(f"{100 * summary['starvation_rate']:.0f}% of routine patients waited > 90 min")
busiest = max(("xray", "ct", "mri", "ultrasound"), key=lambda m: summary.get(f"utilization_{m}", 0))
if summary.get(f"utilization_{busiest}", 0) > 0.9:
    alerts.append(f"{busiest.upper()} is saturated ({100 * summary[f'utilization_{busiest}']:.0f}% utilised) — "
                  "see What-If Simulator for added capacity")
for a in alerts:
    st.warning(a)

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(queue_over_time(samples), use_container_width=True)
with right:
    snap = waiting.groupby("modality").size().reindex(list(MODALITY_COLORS), fill_value=0).reset_index(name="waiting")
    fig = px.bar(snap, x="modality", y="waiting", color="modality", color_discrete_map=MODALITY_COLORS,
                 title=f"Queue at {clock(now)}")
    st.plotly_chart(apply_radqueue_theme(fig, margin=CHART_MARGIN), use_container_width=True)

st.plotly_chart(utilization_gauges(summary), use_container_width=True)
st.plotly_chart(patient_flow_sankey(patients, int(summary.get("no_shows", 0))), use_container_width=True)

with st.expander("Day summary KPIs"):
    st.dataframe(pd.Series(summary).round(3).rename("value"), use_container_width=True)
