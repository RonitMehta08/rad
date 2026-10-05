"""Smart Scheduler — live dispatch and day-ahead MILP schedule.

Owner: P5 (Dashboard & API) on top of P3's scheduler
"""

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import gantt
from dashboard.components.data import get_registry
from src.scheduler import DayAheadOptimizer, PatientState, Resource
from src.services.live_department import LiveDepartment
from src.simulation.entities import SimulationSettings
from src.utils.exceptions import RadQueueError
from src.utils.plotting import apply_radqueue_theme

st.set_page_config(page_title="Smart Scheduler", page_icon="📅", layout="wide")
st.title("📅 Smart Scheduler")

live_tab, plan_tab = st.tabs(["Live dispatch", "Day-ahead plan (MILP)"])

with live_tab:
    if "department" not in st.session_state or st.sidebar.button("Reset department"):
        st.session_state.department = LiveDepartment(wait_time_predictor=get_registry().try_wait_time_predictor())
        st.session_state.counter = 0
    dept: LiveDepartment = st.session_state.department

    with st.form("register"):
        c1, c2, c3 = st.columns(3)
        modality = c1.selectbox("Modality", ["xray", "ct", "mri", "ultrasound"], index=2)
        urgency = c2.selectbox("Urgency", ["routine", "urgent", "emergency"])
        visit = c3.selectbox("Visit type", ["walk_in", "scheduled", "emergency"])
        if st.form_submit_button("Register patient", type="primary"):
            st.session_state.counter += 1
            try:
                res = dept.register_patient({"patient_id": f"P{st.session_state.counter:03d}",
                                             "modality": modality, "urgency": urgency, "visit_type": visit})
                msg = f"{res['patient_id']}: {res['status']}"
                if res["preempted_patient_id"]:
                    msg += f" — emergency preempted {res['preempted_patient_id']}"
                if res["eta"]:
                    msg += f" — position {res['eta']['queue_position']}, ETA {res['eta']['eta_minutes']} min"
                    if res["eta"]["ml_estimate_minutes"] is not None:
                        msg += f" (ML registration estimate {res['eta']['ml_estimate_minutes']} min)"
                st.success(msg)
            except RadQueueError as exc:
                st.error(str(exc))

    c1, c2 = st.columns(2)
    minutes = c1.number_input("Advance clock by (min)", 1, 240, 15)
    if c1.button("Advance clock"):
        st.success(f"Completed: {dept.advance_clock(minutes)['completed'] or 'none'}")
    snap = dept.snapshot()
    failed = c2.selectbox("Machine", [r["resource_id"] for r in snap["resources"]])
    if c2.button("Report equipment failure"):
        st.warning(f"Re-queued: {dept.fail_equipment(failed)['requeued_patient_id'] or 'nobody'}")
        snap = dept.snapshot()

    st.caption(f"Department clock: {snap['now']:.0f} min after opening · average wait {snap['avg_wait_minutes']} min")
    left, right = st.columns(2)
    left.markdown("**Machines**")
    left.dataframe(pd.DataFrame(snap["resources"]), use_container_width=True, hide_index=True)
    right.markdown("**Waiting queue (with ETA)**")
    queue = pd.DataFrame(snap["waiting"])
    if not queue.empty:
        etas = [dept.estimate_wait(pid) for pid in queue["patient_id"]]
        queue["position"] = [e["queue_position"] for e in etas]
        queue["eta_min"] = [e["eta_minutes"] for e in etas]
        queue["ml_estimate_min"] = [e["ml_estimate_minutes"] for e in etas]
        queue = queue.sort_values("position")
    right.dataframe(queue, use_container_width=True, hide_index=True)
    with st.expander("Event log"):
        st.code("\n".join(snap["events"]) or "No events yet")

with plan_tab:
    c1, c2, c3 = st.columns(3)
    tier = c1.selectbox("Tier", ["tier_1", "tier_2", "tier_3"])
    n = c2.slider("Booked appointments", 10, 80, 30)
    seed = c3.number_input("Seed", 1, 999, 7)
    if st.button("Optimise day-ahead schedule", type="primary"):
        settings = SimulationSettings.from_config(tier)
        rng = np.random.default_rng(int(seed))
        mods = list(settings.modality_mix)
        probs = np.array(list(settings.modality_mix.values()))
        patients = [
            PatientState(patient_id=f"A{i:02d}", modality=str(rng.choice(mods, p=probs / probs.sum())),
                         urgency=str(rng.choice(["routine", "urgent"], p=[0.9, 0.1])),
                         arrival_time_minutes=float(rng.integers(0, 10 * 60)))
            for i in range(n)
        ]
        resources = [Resource(resource_id=f"{m.value}_{i + 1}", modality=m)
                     for m, c in settings.machines.items() for i in range(c)]
        with st.spinner("Solving MILP (PuLP/CBC)…"):
            result = DayAheadOptimizer().optimize_schedule(patients, resources)
        st.caption(f"Solver status: **{result['status']}** · objective {result['objective_value']:.0f} · "
                   f"unassigned: {result['unassigned_patients'] or 'none'}")
        df = pd.DataFrame([a.model_dump() for a in result["assignments"]])
        if not df.empty:
            df["end"] = df["start_time_minutes"] + df["estimated_duration_minutes"]
            st.plotly_chart(gantt(df, "start_time_minutes", "end", "resource_id", "patient_id"),
                            use_container_width=True)
            st.metric("Average planned wait", f"{df['predicted_wait_minutes'].mean():.0f} min")
            slots = pd.DataFrame(result["slot_schedule"]).notna().astype(int)
            slots["hour"] = (slots.index * result["slot_duration_minutes"]) // 60 + 8
            heat = slots.groupby("hour").mean().T
            fig = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                            labels={"color": "occupancy"}, title="Utilisation heatmap (machine × hour)")
            st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)
