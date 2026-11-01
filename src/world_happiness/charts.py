"""Plotly charts for the Streamlit dashboard (dark theme, see :mod:`world_happiness.theme`)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme as T
from .config import TARGET
from .countries import CONTINENTS


def ranking_bars(ranked: pd.DataFrame, bottom: bool, title: str) -> go.Figure:
    """Horizontal bars for a top/bottom-N table from :meth:`Explorer.top`, best at the top.

    Whiskers are WHR's 95% confidence interval: overlapping whiskers mean the ranking
    between those two countries is not statistically settled.
    """
    d = ranked.iloc[::-1]  # plotly draws the first category at the bottom
    color = T.CRITICAL if bottom else T.GOOD
    labels = [f"{r}. {c}" for r, c in zip(d.scope_rank, d.country, strict=True)]
    has_ci = d.ci_lower.notna()
    fig = go.Figure(
        go.Bar(
            x=d[TARGET],
            y=labels,
            orientation="h",
            marker={"color": color, "line": {"width": 0}},
            width=0.62,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (d.ci_upper - d[TARGET]).where(has_ci, 0),
                "arrayminus": (d[TARGET] - d.ci_lower).where(has_ci, 0),
                "color": T.INK_2,
                "thickness": 1.5,
                "width": 4,
            },
            text=[f"{v:.2f}" for v in d[TARGET]],
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": "#ffffff"},
            customdata=d[["continent", "rank"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Score %{x:.2f}<br>%{customdata[0]} · world rank "
                "#%{customdata[1]}<extra></extra>"
            ),
        )
    )
    return fig.update_layout(
        **T.layout(
            title={"text": title},
            height=max(260, 46 * len(d) + 90),
            barcornerradius=4,
            xaxis={"range": [0, 8.5], "title": {"text": "Life evaluation (0-10)"}},
            yaxis={"gridcolor": "rgba(0,0,0,0)", "automargin": True},
            showlegend=False,
        )
    )


def continent_range(summary: pd.DataFrame) -> go.Figure:
    """Dumbbell: least happy -> happiest country per continent, with the continent mean."""
    d = summary.iloc[::-1]
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_scatter(
            x=[r.worst_score, r.best_score],
            y=[r.continent, r.continent],
            mode="lines",
            line={"color": T.AXIS, "width": 6},
            hoverinfo="skip",
            showlegend=False,
        )
    fig.add_scatter(
        x=d.worst_score,
        y=d.continent,
        mode="markers+text",
        name="▼ Least happy",
        marker={"size": 13, "color": T.CRITICAL, "line": {"width": 2, "color": T.SURFACE}},
        text=d.worst,
        textposition="middle left",
        textfont={"color": T.INK_2, "size": 12},
        cliponaxis=False,  # long names near the axis ends stay readable
        hovertemplate="%{text}: %{x:.2f}<extra>least happy</extra>",
    )
    fig.add_scatter(
        x=d.best_score,
        y=d.continent,
        mode="markers+text",
        name="▲ Happiest",
        marker={"size": 13, "color": T.GOOD, "line": {"width": 2, "color": T.SURFACE}},
        text=d.best,
        textposition="middle right",
        textfont={"color": T.INK_2, "size": 12},
        cliponaxis=False,  # long names near the axis ends stay readable
        hovertemplate="%{text}: %{x:.2f}<extra>happiest</extra>",
    )
    fig.add_scatter(
        x=d["mean"],
        y=d.continent,
        mode="markers",
        name="◆ Continent mean",
        marker={
            "size": 12,
            "symbol": "diamond",
            "color": T.INK,
            "line": {"width": 2, "color": T.SURFACE},
        },
        customdata=d[["countries"]].to_numpy(),
        hovertemplate="Mean %{x:.2f} over %{customdata[0]} countries<extra></extra>",
    )
    return fig.update_layout(
        **T.layout(
            title={"text": "Spread inside each continent"},
            height=420,
            legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
            xaxis={"range": [-0.6, 10.2], "title": {"text": "Life evaluation (0-10)"}},
            yaxis={"gridcolor": "rgba(0,0,0,0)"},
            margin={"l": 10, "r": 20, "t": 90, "b": 40},
        )
    )


def continent_distribution(year_df: pd.DataFrame, continents: list[str]) -> go.Figure:
    """Every country as a dot over a box per continent: shows shape, not just the mean."""
    fig = go.Figure()
    for c in [c for c in CONTINENTS if c in continents]:
        g = year_df[year_df.continent == c]
        fig.add_box(
            y=g[TARGET],
            name=c,
            boxpoints="all",
            jitter=0.45,
            pointpos=0,
            marker={
                "color": T.CONTINENT_COLORS[c],
                "size": 8,
                "line": {"width": 1.5, "color": T.SURFACE},
            },
            line={"color": T.CONTINENT_COLORS[c], "width": 2},
            fillcolor="rgba(0,0,0,0)",
            text=g.country,
            hovertemplate="<b>%{text}</b><br>%{y:.2f}<extra>" + c + "</extra>",
        )
    return fig.update_layout(
        **T.layout(
            title={"text": "Distribution of country scores"},
            height=420,
            showlegend=False,
            yaxis={"title": {"text": "Life evaluation (0-10)"}, "range": [0, 8.5]},
        )
    )


def continent_trends(trend: pd.DataFrame, continents: list[str]) -> go.Figure:
    """Mean score per continent over time; one fixed colour per continent."""
    fig = go.Figure()
    for c in [c for c in CONTINENTS if c in continents]:
        g = trend[trend.continent == c].sort_values("year")
        fig.add_scatter(
            x=g.year,
            y=g["mean"],
            name=c,
            mode="lines+markers",
            line={"color": T.CONTINENT_COLORS[c], "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": T.SURFACE}},
            customdata=g[["count"]].to_numpy(),
            hovertemplate=c + " %{x}: <b>%{y:.2f}</b> (n=%{customdata[0]})<extra></extra>",
        )
    return fig.update_layout(
        **T.layout(
            title={"text": "Continent mean over time"},
            height=420,
            hovermode="x unified",
            yaxis={"title": {"text": "Mean life evaluation"}},
            xaxis={"dtick": 2},
        )
    )
