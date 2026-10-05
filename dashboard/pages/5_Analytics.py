"""Analytics & Reports — historical patterns from the generated OPD dataset.

Owner: P5 (Dashboard & API)
"""

import _bootstrap  # noqa: F401
import plotly.express as px
import streamlit as st

from dashboard.components.charts import kpi_row
from dashboard.components.data import load_generated_patients
from src.utils.plotting import MODALITY_COLORS, apply_radqueue_theme

TARGET_WAIT_MINUTES = 30
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LABELS = {"actual_wait_time_minutes": "Wait (min)"}

st.set_page_config(page_title="Analytics & Reports", page_icon="📈", layout="wide")
st.title("📈 Analytics & Reports")

df = load_generated_patients()
if df is None:
    st.warning("data/generated/patients.csv not found. Run `python -m src.data.generator` first.")
    st.stop()

served = df[df["showed_up"].astype(bool)]
scheduled = df[df["visit_type"] == "scheduled"]
kpi_row([
    ("Patient records", f"{len(df):,}", None),
    ("Average wait", f"{served['actual_wait_time_minutes'].mean():.0f} min", None),
    (f"Seen within {TARGET_WAIT_MINUTES} min", f"{100 * (served['actual_wait_time_minutes'] <= TARGET_WAIT_MINUTES).mean():.0f}%", None),
    ("Scheduled no-show rate", f"{100 * (1 - scheduled['showed_up'].astype(bool).mean()):.0f}%", None),
])

daily = served.groupby("date")["actual_wait_time_minutes"].mean().reset_index()
daily = daily.rename(columns={"actual_wait_time_minutes": "Daily average"})
daily["7-day average"] = daily["Daily average"].rolling(7, min_periods=1).mean()
fig = px.line(daily, x="date", y=["Daily average", "7-day average"], title="Daily average wait (min)",
              labels={"value": "Wait (min)", "date": "Date", "variable": ""})
st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)

c1, c2 = st.columns(2)
fig = px.violin(served, x="modality", y="actual_wait_time_minutes", color="modality", box=True,
                color_discrete_map=MODALITY_COLORS, title="Wait by modality",
                labels={**LABELS, "modality": "Modality"})
fig.update_yaxes(range=[0, served["actual_wait_time_minutes"].quantile(0.99)])
c1.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)

heat = served.pivot_table(index="day_of_week", columns="hour_of_day", values="actual_wait_time_minutes", aggfunc="mean")
heat.index = [DAY_NAMES[i] for i in heat.index]
fig = px.imshow(heat, aspect="auto", color_continuous_scale="Inferno",
                labels={"color": "Avg wait (min)", "x": "Hour of day", "y": ""}, title="Peak-hour heatmap")
c2.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)

c1, c2 = st.columns(2)
ns = scheduled.groupby("appointment_lead_time_days")["showed_up"].apply(lambda s: 1 - s.astype(bool).mean())
fig = px.bar(ns.reset_index(name="no_show_rate"), x="appointment_lead_time_days", y="no_show_rate",
             title="No-show rate by lead time",
             labels={"appointment_lead_time_days": "Days between booking and visit", "no_show_rate": "No-show rate"})
fig.update_yaxes(tickformat=".0%")
c1.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)
by_visit = served.groupby(["visit_type", "urgency"])["actual_wait_time_minutes"].mean().reset_index()
fig = px.bar(by_visit, x="visit_type", y="actual_wait_time_minutes", color="urgency", barmode="group",
             title="Wait by visit type & urgency",
             labels={**LABELS, "visit_type": "Visit type", "urgency": "Urgency"})
c2.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)
