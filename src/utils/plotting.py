"""Shared Plotly theming and plotting utilities for RadQueue AI.

Owner: P1 (Data & Config Lead)
Consumers: P2 (report figures), P4 (simulation plots), P5 (dashboard)
Reference: MASTER_PROMPT §9.2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# RadQueue Premium Dark Theme
# ---------------------------------------------------------------------------

RADQUEUE_COLORS = {
    "primary": "#6366F1",      # Indigo-500
    "secondary": "#8B5CF6",    # Violet-500
    "accent": "#06B6D4",       # Cyan-500
    "success": "#10B981",      # Emerald-500
    "warning": "#F59E0B",      # Amber-500
    "danger": "#EF4444",       # Red-500
    "bg_dark": "#0F172A",      # Slate-900
    "bg_card": "#1E293B",      # Slate-800
    "bg_surface": "#334155",   # Slate-700
    "text_primary": "#F8FAFC", # Slate-50
    "text_secondary": "#94A3B8",  # Slate-400
    "grid": "#334155",         # Slate-700
}

MODALITY_COLORS: dict[str, str] = {
    "xray": "#3B82F6",       # Blue
    "ct": "#F59E0B",         # Amber
    "mri": "#EF4444",        # Red
    "ultrasound": "#10B981", # Emerald
}

POLICY_COLORS: dict[str, str] = {
    "fcfs": "#94A3B8",            # Slate (baseline)
    "priority": "#F59E0B",        # Amber
    "sjf": "#3B82F6",             # Blue
    "wave": "#8B5CF6",            # Violet
    "radqueue_ai": "#10B981",     # Emerald (ours — highlight)
    "radqueue_noshow": "#06B6D4", # Cyan
}


def get_radqueue_layout(**overrides: Any) -> dict[str, Any]:
    """Return a Plotly layout dict with the RadQueue dark theme.

    Args:
        **overrides: Any layout properties to override.

    Returns:
        Dictionary suitable for ``fig.update_layout(**layout)``.
    """
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": RADQUEUE_COLORS["bg_dark"],
        "plot_bgcolor": RADQUEUE_COLORS["bg_card"],
        "font": {
            "family": "Inter, system-ui, sans-serif",
            "color": RADQUEUE_COLORS["text_primary"],
            "size": 13,
        },
        "title_font_size": 18,
        "title_x": 0.5,
        "xaxis": {
            "gridcolor": RADQUEUE_COLORS["grid"],
            "zerolinecolor": RADQUEUE_COLORS["grid"],
        },
        "yaxis": {
            "gridcolor": RADQUEUE_COLORS["grid"],
            "zerolinecolor": RADQUEUE_COLORS["grid"],
        },
        "colorway": [
            RADQUEUE_COLORS["primary"],
            RADQUEUE_COLORS["accent"],
            RADQUEUE_COLORS["success"],
            RADQUEUE_COLORS["warning"],
            RADQUEUE_COLORS["danger"],
            RADQUEUE_COLORS["secondary"],
        ],
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "hoverlabel": {
            "bgcolor": RADQUEUE_COLORS["bg_surface"],
            "font_size": 12,
        },
    }
    layout.update(overrides)
    return layout


def apply_radqueue_theme(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply the RadQueue dark theme to a Plotly figure in-place.

    Args:
        fig: Plotly Figure to style.
        **overrides: Layout overrides.

    Returns:
        The same figure, styled.
    """
    fig.update_layout(**get_radqueue_layout(**overrides))
    return fig


# ---------------------------------------------------------------------------
# Common Chart Builders
# ---------------------------------------------------------------------------


def create_bar_comparison(
    labels: list[str],
    values: list[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
) -> go.Figure:
    """Create a styled horizontal bar chart for comparing values.

    Args:
        labels: Category labels.
        values: Numeric values per category.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        colors: Optional list of colors per bar.

    Returns:
        Styled Plotly Figure.
    """
    bar_colors = colors or [RADQUEUE_COLORS["primary"]] * len(labels)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
        )
    )
    apply_radqueue_theme(fig, title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig


def create_scatter_pred_vs_actual(
    y_true: list[float],
    y_pred: list[float],
    title: str = "Predicted vs Actual Wait Time",
) -> go.Figure:
    """Create a prediction-vs-actual scatter plot with diagonal reference line.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        title: Chart title.

    Returns:
        Styled Plotly Figure.
    """
    max_val = max(max(y_true), max(y_pred)) * 1.05

    fig = go.Figure()

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line={"dash": "dash", "color": RADQUEUE_COLORS["text_secondary"], "width": 1},
        name="Perfect Prediction",
        showlegend=True,
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker={
            "color": RADQUEUE_COLORS["primary"],
            "size": 5,
            "opacity": 0.6,
        },
        name="Predictions",
    ))

    apply_radqueue_theme(
        fig,
        title=title,
        xaxis_title="Actual Wait Time (min)",
        yaxis_title="Predicted Wait Time (min)",
    )
    return fig


def save_figure(
    fig: go.Figure,
    filepath: str | Path,
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> None:
    """Save a Plotly figure to a static image file.

    Args:
        fig: Figure to save.
        filepath: Output path (PNG, SVG, or HTML).
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution multiplier.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == ".html":
        fig.write_html(str(filepath))
    else:
        try:
            fig.write_image(str(filepath), width=width, height=height, scale=scale)
        except ValueError:
            # Kaleido not installed — fall back to HTML
            html_path = filepath.with_suffix(".html")
            fig.write_html(str(html_path))
