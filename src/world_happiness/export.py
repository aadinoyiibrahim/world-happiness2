from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from . import theme as T
from .analysis import WORLD
from .livemap import VIEWS, Frame, Mode, base_figure, build_frames, color_range


def _card(frame: Frame, pal: T.Palette) -> str:
    """
    The ranking card, as Plotly annotation HTML (the live page draws it in HTML/CSS).
    """
    top = "<br>".join(f"{i}. {c}  {s:.2f}" for i, (c, s) in enumerate(frame.top, 1))
    worst = "<br>".join(f"{c}  {s:.2f}" for c, s in reversed(frame.bottom))
    text = f"<b style='color:{T.GOOD}'>▲ HAPPIEST</b><br>{top}"
    if worst:
        text += f"<br><br><b style='color:{T.CRITICAL}'>▼ LEAST HAPPY</b><br>{worst}"
    return text


def frame_figure(
    locs: list[str],
    frame: Frame,
    scope: str,
    mode: Mode,
    zrange: tuple[float, float],
    pal: T.Palette,
    lon: float | None = None,
) -> go.Figure:
    """
    the dashboard's map plus the year and the ranking card as annotations.
    """
    fig = base_figure(locs, frame, scope, mode, zrange, pal)
    if lon is not None:
        fig.update_geos(projection_rotation_lon=lon)
    where = "World" if scope == WORLD else scope
    surveyed = sum(v is not None for v in frame.z)
    return fig.update_layout(
        paper_bgcolor=pal.page,
        annotations=[
            {
                "text": f"<b>{frame.year}</b>",
                "x": 0.01,
                "y": 0.99,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 46, "color": pal.ink, "family": T.FONT},
            },
            {
                "text": f"{where} · {surveyed} countries surveyed",
                "x": 0.012,
                "y": 0.87,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "showarrow": False,
                "font": {"size": 13, "color": pal.muted, "family": T.FONT},
            },
            {
                "text": _card(frame, pal),
                "x": 0.99,
                "y": 0.97,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "right",
                "yanchor": "top",
                "align": "left",
                "showarrow": False,
                "bgcolor": pal.surface,
                "bordercolor": pal.axis,
                "borderpad": 10,
                "font": {"size": 12, "color": pal.ink, "family": T.FONT},
            },
        ],
    )


def export_map(
    df: pd.DataFrame,
    outdir_: str,
    scope: str = WORLD,
    mode: Mode = "timeline",
    theme: str = "dark",
    n: int = 5,
    year: int | None = None,
    width: int = 1200,
    height: int = 620,
    frame_ms: int = 900,
    globe_steps: int = 36,
) -> Path:
    """
    Write ``<outdir_>happiness_<scope>_<mode>_<theme>.gif`` and return its path.

    ``timeline`` shows one frame per WHR year (the last year is held 3x longer); ``globe``
    turns the globe once (world) or rocks it around the continent, at ``year``
    (default: latest), in ``globe_steps`` frames.
    """
    if scope not in VIEWS:
        raise ValueError(f"unknown scope {scope!r}")
    pal = T.palette(theme)
    locs, frames = build_frames(df, scope, n, pal)
    zrange = color_range(df, scope)

    if mode == "timeline":
        figs = [frame_figure(locs, f, scope, mode, zrange, pal) for f in frames]
        durations = [frame_ms] * (len(figs) - 1) + [frame_ms * 3]
    else:
        years = [f.year for f in frames]
        frame = frames[years.index(year) if year in years else -1]
        lon0 = VIEWS[scope][2][0]
        if scope == WORLD:
            lons = [((lon0 + 360 * i / globe_steps + 180) % 360) - 180 for i in range(globe_steps)]
        else:  # sway +-45 degrees, like the live page
            lons = [lon0 + 45 * math.sin(2 * math.pi * i / globe_steps) for i in range(globe_steps)]
        figs = [frame_figure(locs, frame, scope, mode, zrange, pal, lon=x) for x in lons]
        durations = [120] * len(figs)

    images = [
        Image.open(io.BytesIO(pio.to_image(f, format="png", width=width, height=height)))
        for f in figs
    ]
    # One shared palette keeps colours stable between frames and the file small.
    first = images[0].convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    stills = [first] + [im.convert("RGB").quantize(palette=first) for im in images[1:]]

    out = Path(outdir_)
    out.mkdir(parents=True, exist_ok=True)
    name = f"happiness_{scope.lower().replace(' ', '_')}_{mode}_{pal.mode}.gif"
    path = out / name
    stills[0].save(
        path, save_all=True, append_images=stills[1:], duration=durations, loop=0, optimize=True
    )
    return path
