"""What-If Simulator — policy comparison and capacity scenarios on the digital twin.

Owner: P5 with P4 (Simulation Engineer)
"""

import json

import _bootstrap  # noqa: F401
import plotly.express as px
import streamlit as st

from dashboard.components.charts import policy_bars
from dashboard.components.data import compare_policies, whatif
from src.simulation.scenarios import POLICY_DISPLAY_NAMES, POLICY_NAMES, load_whatif_scenarios
from src.utils.plotting import apply_radqueue_theme

st.set_page_config(page_title="What-If Simulator", page_icon="🔬", layout="wide")
st.title("🔬 What-If Simulator")
st.caption("Every arm runs on the same seeded patient streams (common random numbers); bars show mean ± 95% CI.")

with st.sidebar:
    tier = st.selectbox("Hospital tier", ["tier_1", "tier_2", "tier_3"])
    days = st.slider("Days per replication", 1, 10, 3)
    reps = st.slider("Replications", 2, 20, 5)
    seed = st.number_input("Base seed", 1, 999, 42)

compare_tab, scenario_tab = st.tabs(["Policy comparison", "Capacity scenario"])

with compare_tab:
    chosen = st.multiselect("Policies", POLICY_NAMES, default=POLICY_NAMES, format_func=POLICY_DISPLAY_NAMES.get)
    if st.button("Run comparison", type="primary") and chosen:
        summary = compare_policies(tuple(chosen), tier, days, reps, int(seed))
        c1, c2 = st.columns(2)
        c1.plotly_chart(policy_bars(summary, "avg_wait", "Average wait (min)"), use_container_width=True)
        c2.plotly_chart(policy_bars(summary, "p90_wait", "90th percentile wait (min)"), use_container_width=True)
        c1.plotly_chart(policy_bars(summary, "emergency_avg_wait", "Emergency average wait (min)"),
                        use_container_width=True)
        c2.plotly_chart(policy_bars(summary, "patients_per_day", "Patients served per day"), use_container_width=True)
        cols = ["policy_name", "avg_wait", "p90_wait", "max_wait", "emergency_avg_wait",
                "emergency_within_limit_rate", "urgent_within_limit_rate", "starvation_rate",
                "utilization", "patients_per_day"]
        st.dataframe(summary[[c for c in cols if c in summary.columns]].round(3), hide_index=True,
                     use_container_width=True)

with scenario_tab:
    presets = {s["name"]: s for s in load_whatif_scenarios()}
    mode = st.radio("Scenario", ["Preset", "Custom"], horizontal=True)
    if mode == "Preset":
        name = st.selectbox("Preset", list(presets), format_func=lambda n: presets[n].get("description", n))
        scenario = presets[name]
    else:
        c1, c2, c3 = st.columns(3)
        scenario = {
            "name": "custom",
            "machine_adjustments": {m: c1.number_input(f"Extra {m.upper()} machines", -2, 5, 0)
                                    for m in ("xray", "ct", "mri", "ultrasound")},
            "staff_adjustments": {"technologists": c2.number_input("Extra technologists", -3, 10, 0),
                                  "radiologists": c2.number_input("Extra radiologists", -2, 5, 0)},
            "arrival_rate_multiplier": c3.slider("Patient volume ×", 0.5, 2.0, 1.0, 0.1),
            "noshow_rate_multiplier": c3.slider("No-show rate ×", 0.0, 2.0, 1.0, 0.1),
        }
    policy = st.selectbox("Policy", POLICY_NAMES, index=4, format_func=POLICY_DISPLAY_NAMES.get)
    if st.button("Run scenario", type="primary"):
        table = whatif(json.dumps(scenario), policy, tier, days, reps, int(seed))
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
        fig = px.bar(table, x="kpi", y="delta_pct", title="Change vs baseline (%)", color="delta_pct",
                     color_continuous_scale="RdYlGn_r")
        st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)
