
from __future__ import annotations

import pandas as pd
import streamlit as st

from world_happiness import charts
from world_happiness import theme as T
from world_happiness.analysis import PARTS, WORLD, Explorer
from world_happiness.config import TARGET
from world_happiness.countries import CONTINENTS
from world_happiness.insights import explain_gap

CFG = {"displayModeBar": False}
MIN_COUNTRIES = 5  


def _pick(label: str, options: list[str], default: str, key: str) -> str:
    return st.selectbox(label, options, index=options.index(default), key=key)


def _narrative(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {line}" for line in lines))


def render(ex: Explorer, year: int, pal: T.Palette) -> None:
    fy = ex.factor_year(year)
    if fy != year:
        st.info(
            f"WHR publishes the factor breakdown from {ex.factor_years[0]} onwards, so this tab "
            f"uses **{fy}** (you selected {year})."
        )
    _continents(ex, fy, pal)
    st.divider()
    _countries(ex, fy, pal)
    _method(ex, fy)

def _continents(ex: Explorer, fy: int, pal: T.Palette) -> None:
    means = ex.factor_means(fy)
    m = means.set_index("group")
    cont = m.drop(index=WORLD)
    big = cont[cont["countries"] >= MIN_COUNTRIES]
    if len(big) < 2:  
        big = cont
    st.markdown(f"### 1 · Best vs worst continent, {fy}")
    st.caption(
        f"Defaults: the highest and lowest mean among continents with at least "
        f"{MIN_COUNTRIES} countries. Change either one to compare any pair."
    )
    a, b = st.columns(2)
    names = [c for c in CONTINENTS if c in cont.index]
    with a:
        high = _pick("Higher", names, str(big[TARGET].idxmax()), "why_high")
    with b:
        low = _pick("Lower", names, str(big[TARGET].idxmin()), "why_low")
    if high == low:
        st.warning("Pick two different continents.")
        return
    if m.loc[high, TARGET] < m.loc[low, TARGET]:
        high, low = low, high  
    gap = ex.gap(m.loc[high], m.loc[low])
    _narrative(explain_gap(gap, high, low))
    left, right = st.columns(2)
    left.plotly_chart(charts.decomposition_bars(means, [high, low], pal), theme=None, config=CFG)
    right.plotly_chart(charts.gap_waterfall(gap, high, low, pal), theme=None, config=CFG)
    st.plotly_chart(
        charts.gap_trend(ex.continent_gap_trend(high, low), high, low, pal),
        theme=None,
        config=CFG,
    )
    st.plotly_chart(
        charts.factor_profile(ex.decomposed(fy), [high, low], pal), theme=None, config=CFG
    )
    st.caption(
        f"Means count each country once (not population-weighted): n = "
        f"{int(m.loc[high, 'countries'])} for {high}, {int(m.loc[low, 'countries'])} for {low}."
    )


def _countries(ex: Explorer, fy: int, pal: T.Palette) -> None:
    st.markdown(f"### 2 · Inside one continent: its best vs its worst country, {fy}")
    means = ex.factor_means(fy).set_index("group")
    lowest = str(means.drop(index=WORLD)[TARGET].idxmin())
    cont = (
        st.segmented_control("Continent", list(CONTINENTS), default=lowest, key="why_continent")
        or lowest
    )
    rows = ex.decomposed(fy, cont)
    if len(rows) < 2:
        st.info(f"{cont} has fewer than two countries with the breakdown in {fy}.")
        return
    names = rows["country"].tolist()
    a, b = st.columns(2)
    best = a.selectbox("Country A (default: happiest)", names, index=0, key="why_best")
    worst = b.selectbox(
        "Country B (default: least happy)", names, index=len(names) - 1, key="why_worst"
    )
    surveyed = ex.scope(cont, fy)
    missing = sorted(set(surveyed.country) - set(rows.country))
    if missing:
        st.caption(f"No factor breakdown in {fy} for: {', '.join(missing)} (left out).")
    if best == worst:
        st.warning("Pick two different countries.")
        return
    ra, rb = (rows[rows.country == c].iloc[0] for c in (best, worst))
    if ra[TARGET] < rb[TARGET]:
        (best, ra), (worst, rb) = (worst, rb), (best, ra)
    gap = ex.gap(ra, rb)
    _narrative(explain_gap(gap, best, worst))

    profile = pd.DataFrame(
        [
            {"group": best, "countries": 1, TARGET: ra[TARGET], **{p: ra[p] for p in PARTS}},
            {"group": worst, "countries": 1, TARGET: rb[TARGET], **{p: rb[p] for p in PARTS}},
            {"group": f"{cont} mean", **means.loc[cont, [TARGET, "countries", *PARTS]]},
            {"group": "World mean", **means.loc[WORLD, [TARGET, "countries", *PARTS]]},
        ]
    )
    left, right = st.columns(2)
    left.plotly_chart(
        charts.decomposition_bars(profile, [best, worst], pal).update_layout(
            title_text="The two countries against the continent and world means", height=480
        ),
        theme=None,
        config=CFG,
    )
    right.plotly_chart(charts.gap_waterfall(gap, best, worst, pal), theme=None, config=CFG)
    trend = ex.continent_trend()
    st.plotly_chart(
        charts.country_trends(
            ex.df, [best, worst], trend[trend.continent == cont], pal, f"{cont} mean"
        ),
        theme=None,
        config=CFG,
    )


def _method(ex: Explorer, fy: int) -> None:
    africa = ex.decomposed(fy, "Africa")
    with st.expander("How to read this, what to expect, and what it cannot tell you"):
        st.markdown(
            """
**What the bars are.** WHR fits one regression of national life evaluations on six
variables, pooled over countries and years. Each "explained by" value is that variable's
coefficient × (country's value − the value in *Dystopia*, a hypothetical country with the
world's lowest value on all six). The score is those six contributions plus *Dystopia +
residual*. So a gap between two scores splits **exactly** into a gap per part.

**What to expect if the numbers are right.**
- The per-part gaps add up to the score gap, within about ±0.01 (WHR rounds each country's
  parts).
- GDP per capita, social support and healthy life expectancy should carry most of any large
  gap. They have the largest coefficients and the widest spread between countries.
  Generosity and perceived corruption usually move the gap by less than 0.1.
- A share above 100% for one part is possible; then another part must be negative.

**What it cannot tell you.**
- **It is not causal.** The split follows WHR's regression weights. GDP and health are
  strongly correlated, so how credit is shared between them is a modelling choice.
  Anything that is not measured (conflict, institutions, climate) is either absorbed by
  the six coefficients or left in the residual. *Confidence:* high in the arithmetic,
  moderate in "these factors are associated with the gap", low in "raising X would close
  the gap by that much".
- **The residual is a mix.** It holds unmodelled national conditions, differences in how
  people answer the survey, and sampling noise. Scores carry 95% intervals of roughly
  ±0.1–0.2.
- **Means count countries, not people.** A continent mean treats Nigeria and Eswatini
  equally.
"""
        )
        if len(africa) >= 2:
            g = ex.gap(africa.iloc[0], africa.iloc[-1])
            parts = " + ".join(
                f"{charts.PART_LABELS[p]} {v:+.2f}" for p, v in zip(g.part, g.gap, strict=True)
            )
            st.markdown(
                f"**Worked example ({fy}, real data).** {africa.country.iat[0]} "
                f"{africa[TARGET].iat[0]:.2f} vs {africa.country.iat[-1]} "
                f"{africa[TARGET].iat[-1]:.2f}: gap {g.gap.sum():.2f} = {parts}. "
                f"Each term is the first country's contribution minus the second's."
            )
