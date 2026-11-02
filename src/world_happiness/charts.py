
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme as T
from .analysis import PARTS
from .config import RESIDUAL, TARGET
from .countries import CONTINENTS

PART_LABELS = {
    "gdp_per_capita": "GDP per capita",
    "social_support": "Social support",
    "healthy_life_expectancy": "Healthy life expectancy",
    "freedom": "Freedom of choice",
    "generosity": "Generosity",
    "perceptions_of_corruption": "Low corruption",
    RESIDUAL: "Dystopia + residual",
}
SCORE_AXIS = {"title": {"text": "Life evaluation (0-10)"}}
NO_GRID = {"gridcolor": "rgba(0,0,0,0)"}


def _ring(pal: T.Palette, size: int, color: str, **extra: object) -> dict[str, object]:
    return {"size": size, "color": color, "line": {"width": 2, "color": pal.surface}, **extra}


def ranking_bars(ranked: pd.DataFrame, bottom: bool, title: str, pal: T.Palette = T.DARK) -> go.Figure:
    """
    Horizontal bars for a top/bottom-N table 
    from :meth:`Explorer.top`, best at the top.

    Whiskers are WHR's 95% confidence interval: overlapping whiskers mean 
    the ranking between those two countries is not statistically settled.
    """
    d = ranked.iloc[::-1]  # the first category at the bottom
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
                "color": pal.ink_2,
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
            pal,
            title={"text": title},
            height=max(260, 46 * len(d) + 90),
            barcornerradius=4,
            xaxis={"range": [0, 8.5], **SCORE_AXIS},
            yaxis=NO_GRID,
            showlegend=False,
        )
    )


def continent_range(summary: pd.DataFrame, pal: T.Palette = T.DARK) -> go.Figure:
    """
    Dumbbell: least happy -> happiest country per continent, 
    with the continent mean.
    """
    d = summary.iloc[::-1]
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_scatter(
            x=[r.worst_score, r.best_score],
            y=[r.continent, r.continent],
            mode="lines",
            line={"color": pal.axis, "width": 6},
            hoverinfo="skip",
            showlegend=False,
        )
    for col, name, score, color, pos in (
        ("worst", "▼ Least happy", "worst_score", T.CRITICAL, "middle left"),
        ("best", "▲ Happiest", "best_score", T.GOOD, "middle right"),
    ):
        fig.add_scatter(
            x=d[score],
            y=d.continent,
            mode="markers+text",
            name=name,
            marker=_ring(pal, 13, color),
            text=d[col],
            textposition=pos,
            textfont={"color": pal.ink_2, "size": 12},
            cliponaxis=False,  # long names near the axis ends stay readable
            hovertemplate="%{text}: %{x:.2f}<extra>" + name[2:].lower() + "</extra>",
        )
    fig.add_scatter(
        x=d["mean"],
        y=d.continent,
        mode="markers",
        name="◆ Continent mean",
        marker=_ring(pal, 12, pal.ink, symbol="diamond"),
        customdata=d[["countries"]].to_numpy(),
        hovertemplate="Mean %{x:.2f} over %{customdata[0]} countries<extra></extra>",
    )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "Spread inside each continent"},
            height=420,
            legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
            xaxis={"range": [-0.6, 10.2], **SCORE_AXIS},
            yaxis=NO_GRID,
            margin={"l": 10, "r": 20, "t": 90, "b": 40},
        )
    )


def continent_distribution(year_df: pd.DataFrame, continents: list[str], pal: T.Palette = T.DARK) -> go.Figure:
    """
    Every country as a dot over a box per continent: 
    shows shape, not just the mean.
    """
    fig = go.Figure()
    colors = pal.continent_colors
    for c in [c for c in CONTINENTS if c in continents]:
        g = year_df[year_df.continent == c]
        fig.add_box(
            y=g[TARGET],
            name=c,
            boxpoints="all",
            jitter=0.45,
            pointpos=0,
            marker={
                "color": colors[c],
                "size": 8,
                "line": {"width": 1.5, "color": pal.surface},
            },
            line={"color": colors[c], "width": 2},
            fillcolor="rgba(0,0,0,0)",
            text=g.country,
            hovertemplate="<b>%{text}</b><br>%{y:.2f}<extra>" + c + "</extra>",
        )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "Distribution of country scores"},
            height=420,
            showlegend=False,
            yaxis={"range": [0, 8.5], **SCORE_AXIS},
        )
    )


def continent_trends(trend: pd.DataFrame, continents: list[str], pal: T.Palette = T.DARK) -> go.Figure:
    """
    Mean score per continent over time; one fixed colour per continent.
    """
    fig = go.Figure()
    colors = pal.continent_colors
    for c in [c for c in CONTINENTS if c in continents]:
        g = trend[trend.continent == c].sort_values("year")
        fig.add_scatter(
            x=g.year,
            y=g["mean"],
            name=c,
            mode="lines+markers",
            line={"color": colors[c], "width": 2},
            marker=_ring(pal, 8, colors[c]),
            customdata=g[["count"]].to_numpy(),
            hovertemplate=c + " %{x}: <b>%{y:.2f}</b> (n=%{customdata[0]})<extra></extra>",
        )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "Continent mean over time"},
            height=420,
            hovermode="x unified",
            yaxis={"title": {"text": "Mean life evaluation"}},
            xaxis={"dtick": 2},
        )
    )


def decomposition_bars(means: pd.DataFrame, highlight: list[str], pal: T.Palette = T.DARK) -> go.Figure:
    """
    Stacked bars: what each group's mean score is made of (WHR decomposition).

    ``means`` has a ``group`` column (continents and/or "World") and one column per part.
    Groups in ``highlight`` are labelled in bold; the others are drawn at reduced opacity.
    """
    d = means.sort_values(TARGET)
    labels = [
        f"<b>{g}</b>" if g in highlight else g for g in d["group"]
    ]  # bold = the pair being compared
    opacity = [1.0 if g in highlight else 0.45 for g in d["group"]]
    fig = go.Figure()
    for part in PARTS:
        fig.add_bar(
            x=d[part],
            y=labels,
            orientation="h",
            name=PART_LABELS[part],
            marker={
                "color": pal.factor_colors[part],
                "opacity": opacity,
                "line": {"width": 1, "color": pal.surface},
            },
            customdata=d[["countries", TARGET]].to_numpy(),
            hovertemplate=(
                f"<b>%{{y}}</b><br>{PART_LABELS[part]}: %{{x:.2f}}<br>"
                "score %{customdata[1]:.2f} · n=%{customdata[0]}<extra></extra>"
            ),
        )
    fig.add_scatter(
        x=d[TARGET],
        y=labels,
        mode="text",
        text=[f"  {v:.2f}" for v in d[TARGET]],
        textposition="middle right",
        textfont={"color": pal.ink, "size": 12},
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
    )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "What each mean score is made of"},
            barmode="stack",
            height=520,
            xaxis={"range": [0, 8.2], **SCORE_AXIS},
            yaxis=NO_GRID,
            legend={
                "orientation": "h",
                "y": -0.16,
                "yanchor": "top",
                "x": 0,
                "traceorder": "normal",  # same order as the stack, left to right
            },
            margin={"l": 10, "r": 30, "t": 56, "b": 20},
        )
    )


def gap_waterfall(gap: pd.DataFrame, high: str, low: str, pal: T.Palette = T.DARK) -> go.Figure:
    """
    Waterfall from ``low``'s score to ``high``'s: each step is 
    one part's share of the gap.
    """
    total = gap["gap"].sum()
    labels = [PART_LABELS[p] for p in gap["part"]]
    text = [
        f"{g:+.2f} ({s:.0%})" if pd.notna(s) else f"{g:+.2f}"
        for g, s in zip(gap["gap"], gap["share"], strict=True)
    ]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=[f"{low} score", *labels, f"{high} score"],
            x=[gap["low"].sum(), *gap["gap"], 0],
            measure=["absolute", *["relative"] * len(gap), "total"],
            text=[f"{gap['low'].sum():.2f}", *text, f"{gap['high'].sum():.2f}"],
            textposition="outside",
            textfont={"color": pal.ink_2, "size": 12},
            cliponaxis=False,
            increasing={"marker": {"color": pal.positive}},
            decreasing={"marker": {"color": pal.negative}},
            totals={"marker": {"color": pal.muted}},
            connector={"line": {"color": pal.axis, "width": 1}},
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    lo = max(0.0, min(gap["low"].sum(), gap["high"].sum()) - 1.0)
    hi = max(gap["low"].sum(), gap["high"].sum()) + 0.9
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": f"Where the {total:.2f}-point gap comes from: {high} vs {low}"},
            height=480,
            showlegend=False,
            xaxis={"range": [lo, hi], **SCORE_AXIS},
            yaxis={"autorange": "reversed", **NO_GRID},
            margin={"l": 10, "r": 30, "t": 56, "b": 40},
        )
    )


def gap_trend(trend: pd.DataFrame, high: str, low: str, pal: T.Palette = T.DARK) -> go.Figure:
    """
    Each part's contribution to the ``high`` - ``low`` gap, year by year.
    """
    fig = go.Figure()
    for part in PARTS:
        g = trend[trend.part == part].sort_values("year")
        fig.add_scatter(
            x=g.year,
            y=g["gap"],
            name=PART_LABELS[part],
            mode="lines+markers",
            line={
                "color": pal.factor_colors[part],
                "width": 2,
                "dash": "dot" if part == RESIDUAL else "solid",
            },
            marker=_ring(pal, 8, pal.factor_colors[part]),
            hovertemplate=PART_LABELS[part] + " %{x}: <b>%{y:+.2f}</b><extra></extra>",
        )
    fig.add_hline(y=0, line_color=pal.axis)
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": f"Is the gap stable? Contribution by year, {high} − {low}"},
            height=420,
            hovermode="x unified",
            yaxis={"title": {"text": "Points of the gap"}},
            xaxis={"dtick": 1},
            legend={"orientation": "h", "y": -0.2, "x": 0},
        )
    )


def factor_profile(rows: pd.DataFrame, groups: list[str], pal: T.Palette = T.DARK, by: str = "continent") -> go.Figure:
    """
    Every country's contribution per factor, 
    one colour per group: uniform or outliers?
    """
    fig = go.Figure()
    colors = pal.continent_colors
    fallback = list(pal.categorical)
    for i, grp in enumerate(groups):
        d = rows[rows[by] == grp]
        long = d.melt(["country"], PARTS[:-1], var_name="part", value_name="points")
        fig.add_box(
            x=long["part"].map(PART_LABELS),
            y=long["points"],
            name=grp,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker={
                "color": colors.get(grp, fallback[i]),
                "size": 6,
                "opacity": 0.85,
                "line": {"width": 1, "color": pal.surface},
            },
            line={"color": colors.get(grp, fallback[i]), "width": 1.5},
            fillcolor="rgba(0,0,0,0)",
            text=long["country"],
            hovertemplate="<b>%{text}</b><br>%{x}: %{y:.2f}<extra>" + grp + "</extra>",
        )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "Country by country: is the gap broad or driven by a few?"},
            boxmode="group",
            height=440,
            yaxis={"title": {"text": "Points contributed to the score"}},
            legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
            margin={"l": 10, "r": 20, "t": 90, "b": 40},
        )
    )


def country_trends(df: pd.DataFrame, countries: list[str], mean: pd.DataFrame, pal: T.Palette = T.DARK, mean_label: str = "Continent mean") -> go.Figure:
    """
    Score over time with 95% CI bands for a few countries, 
    against a reference mean.
    """
    fig = go.Figure()
    colors = [T.GOOD, T.CRITICAL, *pal.categorical]
    for country, color in zip(countries, colors, strict=False):
        g = df[df.country == country].sort_values("year")
        fig.add_scatter(
            x=[*g.year, *g.year[::-1]],
            y=[*g.ci_upper, *g.ci_lower[::-1]],
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=g.year,
            y=g[TARGET],
            name=country,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker=_ring(pal, 8, color),
            hovertemplate=country + " %{x}: <b>%{y:.2f}</b><extra></extra>",
        )
    fig.add_scatter(
        x=mean.year,
        y=mean["mean"],
        name=mean_label,
        mode="lines",
        line={"color": pal.muted, "width": 2, "dash": "dash"},
        hovertemplate=mean_label + " %{x}: %{y:.2f}<extra></extra>",
    )
    return fig.update_layout(
        **T.layout(
            pal,
            title={"text": "Over time (band = 95% CI)"},
            height=400,
            hovermode="x unified",
            yaxis={"range": [0, 8.5], **SCORE_AXIS},
            xaxis={"dtick": 2},
            legend={"orientation": "h", "y": -0.2, "x": 0},
        )
    )
