"""World Happiness Atlas: Streamlit dashboard.

Run with ``uv run wh explore`` or ``uv run streamlit run streamlit_app.py``.

Every control is mirrored in the URL (``?scope=Europe&show=worst&n=5&year=2025&map=globe``),
so a view can be bookmarked or shared.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from world_happiness import charts
from world_happiness import theme as T
from world_happiness.analysis import WORLD, Explorer
from world_happiness.config import TARGET, Settings
from world_happiness.countries import CONTINENTS
from world_happiness.data import WHRDataset
from world_happiness.livemap import render_html

REPO_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
SIZES = (3, 5, 10)
MAP_MODES = {"timeline": "▶ Timeline", "globe": "◍ Globe"}

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1400px; }
  h1 { letter-spacing: -0.03em; margin-bottom: 0 !important; }
  .lede { color: #c3c2b7; margin: 0.2rem 0 1.2rem; }
  div[data-testid="stMetric"] { background: #1a1a19; border: 1px solid rgba(255,255,255,.08);
      border-radius: 12px; padding: 14px 16px; }
  div[data-testid="stMetricLabel"] p { color: #898781; text-transform: uppercase;
      letter-spacing: .05em; font-size: 12px; }
  .note { color: #898781; font-size: 13px; }
</style>
"""


@st.cache_data(show_spinner="Loading the World Happiness Report…")
def load_panel() -> pd.DataFrame:
    """Validated panel straight from the raw WHR file (no pipeline run needed)."""
    path = Settings().raw_path
    if not path.exists():  # e.g. launched from another working directory
        path = REPO_RAW / Settings().raw_file
    return WHRDataset.from_excel(path).df


@st.cache_data(show_spinner=False)
def live_map(df: pd.DataFrame, scope: str, n: int, year: int, mode: str) -> str:
    return render_html(df, scope=scope, n=n, year=year, mode=mode)  # type: ignore[arg-type]


def _param(key: str, allowed: list[str] | tuple[str, ...], default: str) -> str:
    value = st.query_params.get(key, default)
    return value if value in allowed else default


def controls(years: list[int]) -> tuple[str, bool, int, int, str]:
    """The button row. Returns (scope, worst, n, year, map mode) and syncs the URL."""
    scopes = [WORLD, *CONTINENTS]
    q_year = st.query_params.get("year", "")
    year_default = int(q_year) if q_year.isdigit() and int(q_year) in years else years[-1]

    st.markdown("**Where**")
    scope = (
        st.segmented_control(
            "Scope",
            scopes,
            default=_param("scope", scopes, WORLD),
            format_func=lambda s: "🌍 World" if s == WORLD else s,
            label_visibility="collapsed",
            key="scope",
        )
        or WORLD
    )

    c1, c2, c3, c4 = st.columns([1.1, 1, 2.6, 1.2], vertical_alignment="bottom")
    show = (
        c1.segmented_control(
            "Show",
            ["top", "worst"],
            default=_param("show", ["top", "worst"], "top"),
            format_func=lambda s: "▲ Top" if s == "top" else "▼ Worst",
            key="show",
        )
        or "top"
    )
    n = (
        c2.segmented_control(
            "How many",
            SIZES,
            default=int(_param("n", [str(s) for s in SIZES], "5")),
            key="n",
        )
        or 5
    )
    year = c3.select_slider("Year", years, value=year_default, key="year")
    mode = (
        c4.segmented_control(
            "Map",
            list(MAP_MODES),
            default=_param("map", list(MAP_MODES), "timeline"),
            format_func=lambda m: MAP_MODES[m],
            key="map",
        )
        or "timeline"
    )

    st.query_params.from_dict(
        {"scope": scope, "show": show, "n": str(n), "year": str(year), "map": mode}
    )
    return scope, show == "worst", int(n), int(year), mode


def kpis(ex: Explorer, scope: str, year: int) -> None:
    d = ex.scope(scope, year)
    world = ex.scope(WORLD, year)
    cols = st.columns(4)
    cols[0].metric("Countries surveyed", len(d))
    delta = d[TARGET].mean() - world[TARGET].mean()
    cols[1].metric(
        "Mean life evaluation",
        f"{d[TARGET].mean():.2f}",
        None if scope == WORLD else f"{delta:+.2f} vs world",
    )
    for col, label, i in ((cols[2], "▲ Happiest", 0), (cols[3], "▼ Least happy", -1)):
        col.metric(
            label,
            d.country.iat[i],
            f"{d[TARGET].iat[i]:.2f} / 10",
            delta_color="off",
            delta_arrow="off",
        )


def ranking_table(ex: Explorer, ranked: pd.DataFrame, year: int) -> pd.DataFrame:
    """Ranked rows plus the change since the previous WHR year for each country."""
    prev_years = [y for y in ex.years if y < year]
    out = ranked[["scope_rank", "country", "continent", "rank", TARGET]].assign(
        ci=[
            f"{lo:.2f} – {hi:.2f}" if pd.notna(lo) else "–"
            for lo, hi in zip(ranked.ci_lower, ranked.ci_upper, strict=True)
        ]
    )
    if prev_years:
        prev = ex.df[ex.df.year == prev_years[-1]].set_index("iso3")[TARGET]
        out = out.assign(change=ranked.iso3.map(prev).rsub(ranked[TARGET]).to_numpy())
    return out


def tab_rankings(ex: Explorer, scope: str, worst: bool, n: int, year: int, mode: str) -> None:
    # our own generated page (escaped data, never user input), so running its JS is safe
    st.iframe(live_map(ex.df, scope, n, year, mode), height=606)
    st.markdown(
        '<p class="note">Rings: ▲ green = top N, ▼ red = bottom N of the selected scope in the '
        "year shown. Grey countries have no survey that year. Hover or drag to pause. "
        "Colours use one fixed range per scope, so they compare across years.</p>",
        unsafe_allow_html=True,
    )
    ranked = ex.top(n, year, bottom=worst, scope=scope)
    where = "the world" if scope == WORLD else scope
    title = f"{'▼ Least happy' if worst else '▲ Happiest'} {len(ranked)} in {where}, {year}"
    left, right = st.columns([1.15, 1])
    left.plotly_chart(
        charts.ranking_bars(ranked, worst, title), theme=None, config={"displayModeBar": False}
    )
    with right:
        st.markdown("##### Details")
        table = ranking_table(ex, ranked, year)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "scope_rank": st.column_config.NumberColumn("#", width="small"),
                "country": "Country",
                "continent": "Continent",
                "rank": st.column_config.NumberColumn("World rank", width="small"),
                TARGET: st.column_config.NumberColumn("Score", format="%.2f"),
                "ci": st.column_config.TextColumn("95% CI"),
                "change": st.column_config.NumberColumn(
                    "Δ year", format="%+.2f", help="Change since the previous WHR year"
                ),
            },
        )
        total = len(ex.scope(scope, year))
        if total < n:
            st.info(f"{where} has only {total} surveyed countries in {year}; all are shown.")
        elif total < 2 * n:
            st.caption(f"{where} has {total} countries, so top and bottom {n} overlap.")


def tab_compare(ex: Explorer, worst: bool, n: int, year: int) -> None:
    picked = st.pills(
        "Continents to compare",
        CONTINENTS,
        selection_mode="multi",
        default=list(CONTINENTS),
        key="compare",
    )
    picked = [c for c in CONTINENTS if c in picked] or list(CONTINENTS)
    summary = ex.continent_summary(year)
    summary = summary[summary.continent.isin(picked)]
    world = ex.scope(WORLD, year)

    cfg = {"displayModeBar": False}
    a, b = st.columns(2)
    a.plotly_chart(charts.continent_range(summary), theme=None, config=cfg)
    b.plotly_chart(charts.continent_distribution(world, picked), theme=None, config=cfg)
    st.plotly_chart(charts.continent_trends(ex.continent_trend(), picked), theme=None, config=cfg)
    st.caption(
        "Means are over the countries surveyed each year, and that set changes: part of a "
        "yearly move can be a change in who was surveyed. Oceania is only Australia and NZ."
    )

    st.markdown(f"#### {'▼ Worst' if worst else '▲ Top'} {n} in each continent, {year}")
    cols = st.columns(3)
    for i, c in enumerate(picked):
        r = ex.top(n, year, bottom=worst, scope=c)
        with cols[i % 3]:
            st.markdown(
                f"<span style='color:{T.CONTINENT_COLORS[c]}'>●</span> **{c}**",
                unsafe_allow_html=True,
            )
            st.dataframe(
                r[["scope_rank", "country", "rank", TARGET]],
                hide_index=True,
                width="stretch",
                column_config={
                    "scope_rank": st.column_config.NumberColumn("#", width="small"),
                    "country": "Country",
                    "rank": st.column_config.NumberColumn("World", width="small"),
                    TARGET: st.column_config.NumberColumn("Score", format="%.2f"),
                },
            )

    with st.expander("Continent summary table"):
        st.dataframe(summary.round(2), hide_index=True, width="stretch")


def tab_data(ex: Explorer) -> None:
    df = ex.df
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    st.download_button(
        "⬇ Download CSV", df.to_csv(index=False), "world_happiness_panel.csv", "text/csv"
    )
    st.markdown(
        "**About the data.** World Happiness Report 2026, *Data for Figure 2.1*: the Cantril "
        "ladder life evaluation (0-10), as a 3-year average, where `year` is the last survey "
        "year. There is no 2013. `rank` is the WHR world rank; `#` in the rankings is the "
        "rank inside the selected scope. Continents use the seven-continent split (Americas "
        "split into North, including Central America and the Caribbean, and South). Source: "
        "[worldhappiness.report](https://www.worldhappiness.report/data-sharing/)."
    )


def main() -> None:
    st.set_page_config(page_title="World Happiness Atlas", page_icon="🌍", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    df = load_panel()
    ex = Explorer(df)

    st.title("World Happiness Atlas")
    st.markdown(
        f'<p class="lede">How people in {df.iso3.nunique()} countries rate their lives, '
        f"{df.year.min()}–{df.year.max()} · World Happiness Report 2026</p>",
        unsafe_allow_html=True,
    )
    scope, worst, n, year, mode = controls(ex.years)
    kpis(ex, scope, year)

    rankings, compare, data = st.tabs(["🗺 Map & rankings", "⚖ Compare continents", "📋 Data"])
    with rankings:
        tab_rankings(ex, scope, worst, n, year, mode)
    with compare:
        tab_compare(ex, worst, n, year)
    with data:
        tab_data(ex)


if __name__ == "__main__":
    main()
