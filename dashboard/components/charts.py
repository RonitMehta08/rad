"""Plotly chart builders and KPI cards using the RadQueue dark theme."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.plotting import MODALITY_COLORS, POLICY_COLORS, RADQUEUE_COLORS, apply_radqueue_theme

MINUTES_PER_HOUR = 60
GAUGE_GAP = 0.04
CHART_MARGIN = {"l": 80, "r": 30, "t": 60, "b": 60}


def kpi_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Render a row of metric cards: (label, value, optional delta)."""
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta)


def clock(minutes: float) -> str:
    """Minutes since midnight -> HH:MM."""
    return f"{int(minutes // MINUTES_PER_HOUR):02d}:{int(minutes % MINUTES_PER_HOUR):02d}"


def queue_over_time(samples: pd.DataFrame, day: int = 0) -> go.Figure:
    """Queue length per modality across the day."""
    df = samples[samples["day"] == day].copy()
    df["time"] = df["minute"].map(clock)
    fig = px.line(df, x="time", y="queue_length", color="modality", color_discrete_map=MODALITY_COLORS,
                  title="Live queue length by modality", labels={"queue_length": "Patients waiting", "time": "Time"})
    fig.update_xaxes(nticks=13)
    return apply_radqueue_theme(fig, margin=CHART_MARGIN)


def utilization_gauges(summary: dict[str, float]) -> go.Figure:
    """One gauge per modality showing machine utilisation."""
    modalities = ["xray", "ct", "mri", "ultrasound"]
    width = 1.0 / len(modalities)
    fig = go.Figure()
    for i, m in enumerate(modalities):
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=100 * summary.get(f"utilization_{m}", 0.0),
            number={"suffix": "%", "valueformat": ".0f", "font": {"size": 34}},
            title={"text": m.upper(), "font": {"size": 16}},
            gauge={"axis": {"range": [0, 100], "tickvals": [0, 50, 100]}, "bar": {"color": MODALITY_COLORS[m]}},
            # Gaps between gauges so the 0/100 tick labels of neighbours don't collide
            domain={"x": [i * width + GAUGE_GAP, (i + 1) * width - GAUGE_GAP], "y": [0, 0.8]},
        ))
    fig.update_layout(height=290, title="Machine utilisation (share of operating hours)")
    return apply_radqueue_theme(fig, margin={"l": 30, "r": 30, "t": 70, "b": 10})


def patient_flow_sankey(patients: pd.DataFrame, noshows: int) -> go.Figure:
    """Arrival type -> modality -> outcome flow."""
    visit_types = sorted(patients["visit_type"].unique())
    modalities = sorted(patients["modality"].unique())
    labels = [v.replace("_", "-").title() for v in visit_types] + [m.upper() for m in modalities] + [
        "Reported", "No-show"]
    idx = {name: i for i, name in enumerate(visit_types + modalities + ["reported", "noshow"])}
    flows = patients.groupby(["visit_type", "modality"]).size().reset_index(name="n")
    source = [idx[v] for v in flows["visit_type"]] + [idx[m] for m in modalities]
    target = [idx[m] for m in flows["modality"]] + [idx["reported"]] * len(modalities)
    value = list(flows["n"]) + [int((patients["modality"] == m).sum()) for m in modalities]
    if noshows and "scheduled" in idx:
        source.append(idx["scheduled"])
        target.append(idx["noshow"])
        value.append(noshows)
    def _rgba(hex_color: str, alpha: float = 0.45) -> str:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"

    modality_of_link = list(flows["modality"]) + modalities
    link_colors = [_rgba(MODALITY_COLORS.get(m, RADQUEUE_COLORS["primary"])) for m in modality_of_link]
    if len(link_colors) < len(value):
        link_colors.append(_rgba(RADQUEUE_COLORS["danger"]))
    node_colors = ([RADQUEUE_COLORS["primary"]] * len(visit_types)
                   + [MODALITY_COLORS.get(m, RADQUEUE_COLORS["primary"]) for m in modalities]
                   + [RADQUEUE_COLORS["success"], RADQUEUE_COLORS["danger"]])
    fig = go.Figure(go.Sankey(
        node={"label": labels, "pad": 18, "thickness": 18, "color": node_colors, "line": {"width": 0}},
        link={"source": source, "target": target, "value": value, "color": link_colors},
        textfont={"color": RADQUEUE_COLORS["text_primary"], "size": 14},
    ))
    fig.update_layout(title="Patient flow: arrival → modality → report", height=380)
    return apply_radqueue_theme(fig)


def gantt(assignments: pd.DataFrame, start_col: str, end_col: str, resource_col: str, label_col: str) -> go.Figure:
    """Gantt-style schedule per machine (minutes after opening)."""
    df = assignments.copy()
    df["duration"] = df[end_col] - df[start_col]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["duration"]], y=[row[resource_col]], base=[row[start_col]], orientation="h",
            name=row[label_col], text=row[label_col], textposition="inside",
            marker_color=MODALITY_COLORS.get(str(row[resource_col]).split("_")[0], RADQUEUE_COLORS["primary"]),
            hovertemplate=f"{row[label_col]}<br>start %{{base:.0f}} min<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(barmode="overlay", title="Schedule by machine", xaxis_title="Minutes after opening",
                      height=max(300, 40 * df[resource_col].nunique()))
    return apply_radqueue_theme(fig)


def policy_bars(summary: pd.DataFrame, kpi: str, title: str) -> go.Figure:
    """Bar chart of one KPI across policies with 95% CI error bars."""
    fig = go.Figure(go.Bar(
        x=summary["policy_name"], y=summary[kpi],
        error_y={"type": "data", "array": summary.get(f"{kpi}_ci95")},
        marker_color=[POLICY_COLORS.get(p, RADQUEUE_COLORS["primary"]) for p in summary["policy"]],
    ))
    fig.update_layout(title=title, height=340)
    return apply_radqueue_theme(fig)


def shap_bar(contributions: list[dict]) -> go.Figure:
    """Horizontal bar of per-feature SHAP contributions (minutes)."""
    df = pd.DataFrame(contributions).iloc[::-1]
    colors = [RADQUEUE_COLORS["danger"] if v > 0 else RADQUEUE_COLORS["success"] for v in df["shap_value"]]
    fig = go.Figure(go.Bar(x=df["shap_value"], y=df["display_name"], orientation="h", marker_color=colors,
                           text=[f"{v:+.1f} min" for v in df["shap_value"]], textposition="outside"))
    fig.update_layout(title="Why this prediction? (SHAP contribution, minutes)", height=360)
    return apply_radqueue_theme(fig)
