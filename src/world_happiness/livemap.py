"""A choropleth that never sits still: it loops through the years or spins as a globe.

Python precomputes every frame (scores, ranks, highlight rings); a small self-contained
HTML page runs the loop in the browser with ``Plotly.restyle`` / ``Plotly.relayout``, so the
Streamlit script does not rerun on every tick. Hovering, dragging or the pause button stops
the motion, and ``prefers-reduced-motion`` starts it paused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from . import theme as T
from .analysis import WORLD
from .config import TARGET

Mode = Literal["timeline", "globe"]

# Map window per scope: (lon range, lat range) for the flat map, centre for the globe.
VIEWS: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    WORLD: ((-180, 180), (-58, 84), (10, 20)),
    "Africa": ((-20, 55), (-36, 38), (18, 3)),
    "Asia": ((25, 150), (-12, 56), (88, 28)),
    "Europe": ((-25, 45), (34, 72), (15, 50)),
    "North America": ((-170, -50), (5, 75), (-95, 35)),
    "South America": ((-85, -32), (-57, 14), (-60, -18)),
    "Oceania": ((108, 180), (-50, -5), (145, -25)),
}


@dataclass(frozen=True)
class Frame:
    year: int
    z: list[float | None]
    customdata: list[list[object]]
    line_color: list[str]
    line_width: list[float]
    top: list[tuple[str, float]]
    bottom: list[tuple[str, float]]


def build_frames(df: pd.DataFrame, scope: str, n: int) -> tuple[list[str], list[Frame]]:
    """Fixed location list for the scope + one :class:`Frame` per year.

    Locations stay constant across frames (missing years are ``None``), so the animation only
    swaps values and never re-projects the map.
    """
    d = df if scope == WORLD else df[df["continent"] == scope]
    d = d.dropna(subset=[TARGET])
    locs = sorted(d["iso3"].unique())
    names = d.drop_duplicates("iso3", keep="last").set_index("iso3")["country"]
    frames = []
    for year, g in d.groupby("year", sort=True):
        g = g.sort_values(TARGET, ascending=False).reset_index(drop=True)
        g["scope_rank"] = np.arange(1, len(g) + 1)
        top, bottom = g.head(n), g.tail(min(n, max(len(g) - n, 0)))
        by = g.set_index("iso3")
        z: list[float | None] = []
        cd: list[list[object]] = []
        lc: list[str] = []
        lw: list[float] = []
        for iso in locs:
            if iso not in by.index:
                z.append(None)
                cd.append([names[iso], "-", len(g), "-", "no survey this year"])
                lc.append(T.SURFACE)
                lw.append(0.4)
                continue
            r = by.loc[iso]
            z.append(round(float(r[TARGET]), 3))
            ci = f"95% CI {r.ci_lower:.2f}-{r.ci_upper:.2f}" if pd.notna(r.ci_lower) else "no CI"
            cd.append([r.country, int(r.scope_rank), len(g), int(r["rank"]), ci])
            if iso in set(top.iso3):
                lc.append(T.GOOD)
                lw.append(2.6)
            elif iso in set(bottom.iso3):
                lc.append(T.CRITICAL)
                lw.append(2.6)
            else:
                lc.append(T.SURFACE)
                lw.append(0.4)
        frames.append(
            Frame(
                year=int(year),
                z=z,
                customdata=cd,
                line_color=lc,
                line_width=lw,
                top=[
                    (c, round(float(s), 2)) for c, s in zip(top.country, top[TARGET], strict=True)
                ],
                bottom=[
                    (c, round(float(s), 2))
                    for c, s in zip(bottom.country, bottom[TARGET], strict=True)
                ],
            )
        )
    return locs, frames


def base_figure(
    locs: list[str], frame: Frame, scope: str, mode: Mode, zrange: tuple[float, float]
) -> go.Figure:
    lon, lat, centre = VIEWS[scope]
    hover = (
        "<b>%{customdata[0]}</b><br>Life evaluation <b>%{z:.2f}</b> / 10<br>"
        "#%{customdata[1]} of %{customdata[2]} in "
        + ("the world" if scope == WORLD else scope)
        + " · world rank #%{customdata[3]}<br><span style='color:#898781'>"
        "%{customdata[4]}</span><extra></extra>"
    )
    fig = go.Figure(
        go.Choropleth(
            locations=locs,
            z=frame.z,
            customdata=frame.customdata,
            hovertemplate=hover,
            colorscale=[[i / (len(T.SEQUENTIAL) - 1), c] for i, c in enumerate(T.SEQUENTIAL)],
            zmin=zrange[0],
            zmax=zrange[1],
            marker={"line": {"color": frame.line_color, "width": frame.line_width}},
            colorbar={
                "title": {"text": "Score", "side": "top", "font": {"color": T.INK_2}},
                "orientation": "v",
                "x": 0.012,
                "xanchor": "left",
                "y": 0.06,
                "yanchor": "bottom",
                "len": 0.42,
                "thickness": 10,
                "outlinewidth": 0,
                "tickfont": {"color": T.MUTED},
            },
        )
    )
    geo = {
        "showframe": False,
        "showcoastlines": False,
        "showcountries": True,
        "countrycolor": T.AXIS,
        "showland": True,
        "landcolor": T.LAND,
        "showocean": True,
        "oceancolor": T.OCEAN,
        "showlakes": False,
        "bgcolor": "rgba(0,0,0,0)",
        "domain": {"x": [0.04, 0.78], "y": [0, 1]},  # right edge is kept for the ranking card
    }
    if mode == "globe":
        tilt = max(-30, min(35, centre[1]))  # lean towards the scope, keep the equator in view
        geo |= {
            "projection": {"type": "orthographic", "rotation": {"lon": centre[0], "lat": tilt}},
            "lataxis": {"showgrid": True, "gridcolor": T.GRID, "dtick": 30},
            "lonaxis": {"showgrid": True, "gridcolor": T.GRID, "dtick": 30},
        }
    else:
        geo |= {
            "projection": {"type": "natural earth"},
            "lonaxis": {"range": list(lon)},
            "lataxis": {"range": list(lat)},
        }
    fig.update_layout(
        **T.layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, geo=geo, dragmode="pan"),
    )
    return fig


TEMPLATE = Path(__file__).with_name("livemap.html")


def render_html(
    df: pd.DataFrame,
    scope: str = WORLD,
    n: int = 5,
    year: int | None = None,
    mode: Mode = "timeline",
    height: int = 600,
    interval_ms: int = 1600,
    spin_deg_per_s: float = 8.0,
) -> str:
    """Self-contained HTML for the live map (embed with ``st.components.v1.html``).

    ``timeline`` loops through every year, starting at ``year``; ``globe`` shows ``year`` on
    an orthographic globe that spins (world) or sways around the continent (continent scope);
    the slider still changes the year. The colour range is
    fixed per scope across all years, so colours are comparable between frames.
    """
    if scope not in VIEWS:
        raise ValueError(f"unknown scope {scope!r}")
    locs, frames = build_frames(df, scope, n)
    if not frames:
        raise ValueError(f"no data for scope {scope!r}")
    years = [f.year for f in frames]
    start = years.index(year) if year in years else len(years) - 1
    scores = df[TARGET] if scope == WORLD else df.loc[df["continent"] == scope, TARGET]
    zrange = (float(np.floor(scores.min() * 2) / 2), float(np.ceil(scores.max() * 2) / 2))
    fig = base_figure(locs, frames[start], scope, mode, zrange)
    payload = {
        "fig": json.loads(fig.to_json()),
        "frames": [f.__dict__ for f in frames],
        "start": start,
        "mode": mode,
        "scope": "World" if scope == WORLD else scope,
        "interval": interval_ms,
        "spin": spin_deg_per_s,
        "lon0": VIEWS[scope][2][0],
        "sway": 0 if scope == WORLD else 45,  # continents rock +-45 deg so they stay in view
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    subs = {
        "__PLOTLYJS__": get_plotlyjs_version(),
        "__HEIGHT__": str(height),
        "__N__": str(n),
        "__PAGE__": T.PAGE,
        "__SURFACE__": T.SURFACE,
        "__INK__": T.INK,
        "__INK2__": T.INK_2,
        "__MUTED__": T.MUTED,
        "__GOOD__": T.GOOD,
        "__BAD__": T.CRITICAL,
        "__ACCENT__": T.SEQUENTIAL[3],
        "__FONT__": T.FONT,
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for k, v in subs.items():
        html = html.replace(k, v)
    return html.replace("__DATA__", data)  # last: data must never be re-substituted
