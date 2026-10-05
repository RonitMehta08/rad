"""Indian Context — hospital tiers, holiday/monsoon demand calendar, tier simulation.

Owner: P5 (Dashboard & API)
"""

from datetime import date, timedelta

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from dashboard.components import PROJECT_ROOT
from dashboard.components.data import simulate_day
from src.data.indian_context import get_demand_multiplier, get_weather_category, is_holiday
from src.utils.plotting import apply_radqueue_theme

CONFIG_DIR = PROJECT_ROOT / "config"
DAYS_PER_YEAR = 366

st.set_page_config(page_title="Indian Context", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Indian Context Configuration")

profiles = yaml.safe_load((CONFIG_DIR / "hospital_profiles.yaml").read_text(encoding="utf-8"))
holidays = yaml.safe_load((CONFIG_DIR / "indian_holidays.yaml").read_text(encoding="utf-8"))

st.subheader("Hospital tier profiles")
tier_rows = []
for tier in ("tier_1", "tier_2", "tier_3"):
    p = profiles[tier]
    tier_rows.append({"tier": tier, "name": p["name"], "patients/day": p["daily_patient_volume"],
                      "walk-in": p["walk_in_ratio"], "emergency": p["emergency_ratio"], "no-show": p["noshow_rate"],
                      **p["resources"], **p["staff"]})
st.dataframe(pd.DataFrame(tier_rows), hide_index=True, use_container_width=True)

st.subheader("Demand calendar")
year = st.selectbox("Year", [2024, 2025, 2026], index=0)
days = [date(year, 1, 1) + timedelta(days=i) for i in range(DAYS_PER_YEAR) if (date(year, 1, 1) + timedelta(days=i)).year == year]
cal = pd.DataFrame({"date": days, "demand_multiplier": [get_demand_multiplier(d) for d in days],
                    "season": [get_weather_category(d).value.title() for d in days],
                    "day type": ["Holiday" if is_holiday(d) else "Regular day" for d in days]})
fig = px.scatter(cal, x="date", y="demand_multiplier", color="season", symbol="day type",
                 title="Daily OPD demand vs a normal weekday (weekday pattern × holidays × monsoon)",
                 labels={"date": "Date", "demand_multiplier": "Demand multiplier", "season": "Season"})
st.plotly_chart(apply_radqueue_theme(fig), use_container_width=True)
st.dataframe(pd.DataFrame(holidays["national_holidays"]), hide_index=True, use_container_width=True)
st.caption(f"Post-holiday surge ×{holidays['post_holiday_surge_multiplier']} for {holidays['post_holiday_surge_days']} "
           f"days · Monsoon emergencies ×{holidays['monsoon_season']['emergency_multiplier']}")

st.subheader("How each tier copes (RadQueue AI, one simulated day)")
seed = st.number_input("Seed", 1, 999, 42)
rows = []
for tier in ("tier_1", "tier_2", "tier_3"):
    _, _, s = simulate_day("radqueue_ai", tier, 1, int(seed))
    rows.append({"tier": tier, "patients served": s.get("patients_served"), "avg wait (min)": s.get("avg_wait"),
                 "p90 wait (min)": s.get("p90_wait"), "MRI utilisation": s.get("utilization_mri"),
                 "emergencies ≤10 min": s.get("emergency_within_limit_rate")})
st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
st.caption("Profiles are edited in config/hospital_profiles.yaml and config/indian_holidays.yaml; "
           "custom capacity changes can be tested on the What-If Simulator page.")
