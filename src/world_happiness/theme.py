"""One visual system for the dashboard: dark surfaces, a blue sequential ramp, fixed hues.

Values follow a validated reference palette (dark-mode steps). Change them here only.
"""

from __future__ import annotations

from typing import Any

from .countries import CONTINENTS

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
LAND = "#2c2c2a"
OCEAN = "#121211"

GOOD = "#0ca30c"  # happiest-N ring and bars (always paired with a ▲ label)
CRITICAL = "#d03b3b"  # least-happy-N ring and bars (always paired with a ▼ label)

# Sequential blue, dark -> light: on a dark surface low values recede, happy countries glow.
SEQUENTIAL = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

# Categorical dark-mode steps in fixed order; colour follows the continent, never its rank.
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
CONTINENT_COLORS = dict(zip(CONTINENTS, CATEGORICAL, strict=True))

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def layout(**overrides: Any) -> dict[str, Any]:
    """Plotly layout defaults for every dashboard chart."""
    axis = {
        "gridcolor": GRID,
        "linecolor": AXIS,
        "zerolinecolor": AXIS,
        "tickfont": {"color": MUTED},
        "title": {"font": {"color": INK_2}, "standoff": 12},
        "automargin": True,
    }
    base: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "color": INK, "size": 13},
        "title": {"font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        "margin": {"l": 10, "r": 20, "t": 56, "b": 40},
        "xaxis": axis,
        "yaxis": axis,
        "legend": {"orientation": "h", "y": -0.18, "font": {"color": INK_2}},
        "hoverlabel": {"bgcolor": SURFACE, "bordercolor": AXIS, "font": {"color": INK}},
        "hovermode": "closest",
    }
    for key, value in overrides.items():  # merge one level deep so titles/axes keep their style
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base
