"""One visual system for the dashboard in two modes (dark and light).

Values follow a validated reference palette: each mode has its own steps, chosen for its
surface, not an automatic flip. Change colours here only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FACTORS, RESIDUAL
from .countries import CONTINENTS

# Status hues are mode-invariant and always ship with a ▲/▼ label.
GOOD = "#0ca30c"  # happiest-N rings and bars
CRITICAL = "#d03b3b"  # least-happy-N rings and bars
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


@dataclass(frozen=True)
class Palette:
    mode: str
    page: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    land: str
    ocean: str
    glow: str  # centre of the map's radial background
    card: str  # translucent overlay card on the map
    line: str  # hairline borders
    sequential: tuple[str, ...]  # low -> high; the low end recedes into the surface
    categorical: tuple[str, ...]  # fixed order, never cycled
    positive: str  # diverging pair: gap in favour of the higher group
    negative: str  # ... and against it

    @property
    def continent_colors(self) -> dict[str, str]:
        return dict(zip(CONTINENTS, self.categorical[: len(CONTINENTS)], strict=True))

    @property
    def factor_colors(self) -> dict[str, str]:
        """Six factors take categorical slots 1-6; the unexplained residual is neutral grey."""
        colors = dict(zip(FACTORS, self.categorical[: len(FACTORS)], strict=True))
        return colors | {RESIDUAL: self.muted}


DARK = Palette(
    mode="dark",
    page="#0d0d0d",
    surface="#1a1a19",
    ink="#f0efec",
    ink_2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    land="#2c2c2a",
    ocean="#121211",
    glow="#16181d",
    card="rgba(26,26,25,.84)",
    line="rgba(255,255,255,.10)",
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"),
    positive="#3987e5",
    negative="#e66767",
)

LIGHT = Palette(
    mode="light",
    page="#f9f9f7",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    land="#e1e0d9",
    ocean="#eef2f6",
    glow="#ffffff",
    card="rgba(252,252,251,.90)",
    line="rgba(11,11,11,.10)",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
    positive="#2a78d6",
    negative="#e34948",
)


def palette(mode: str | None) -> Palette:
    """``"light"`` -> :data:`LIGHT`; anything else (incl. ``None``) -> :data:`DARK`."""
    return LIGHT if mode == "light" else DARK


def layout(pal: Palette = DARK, **overrides: Any) -> dict[str, Any]:
    """Plotly layout defaults for every dashboard chart, in the given palette."""
    axis = {
        "gridcolor": pal.grid,
        "linecolor": pal.axis,
        "zerolinecolor": pal.axis,
        "tickfont": {"color": pal.muted},
        "title": {"font": {"color": pal.ink_2}, "standoff": 12},
        "automargin": True,
    }
    base: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "color": pal.ink, "size": 13},
        "title": {"font": {"size": 16, "color": pal.ink}, "x": 0, "xanchor": "left"},
        "margin": {"l": 10, "r": 20, "t": 56, "b": 40},
        "xaxis": axis,
        "yaxis": axis,
        "legend": {"orientation": "h", "y": -0.18, "font": {"color": pal.ink_2}},
        "hoverlabel": {
            "bgcolor": pal.surface,
            "bordercolor": pal.axis,
            "font": {"color": pal.ink},
        },
        "hovermode": "closest",
    }
    for key, value in overrides.items():  # merge one level deep so titles/axes keep their style
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base
