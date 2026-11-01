"""Plain-language readings of a gap decomposition (pure functions, unit-tested)."""

from __future__ import annotations

import pandas as pd

from .config import RESIDUAL

LABELS = {
    "gdp_per_capita": "GDP per capita",
    "social_support": "social support",
    "healthy_life_expectancy": "healthy life expectancy",
    "freedom": "freedom to make life choices",
    "generosity": "generosity",
    "perceptions_of_corruption": "low perceived corruption",
    RESIDUAL: "the residual",
}


def explain_gap(gap: pd.DataFrame, high: str, low: str) -> list[str]:
    """Three to five sentences describing a :meth:`Explorer.gap` table.

    The sentences only restate the arithmetic of WHR's decomposition (which parts are large,
    how much the residual carries); they make no causal claim.
    """
    total = float(gap["gap"].sum())
    named = gap[gap["part"] != RESIDUAL].sort_values("gap", ascending=False)
    resid = float(gap.loc[gap["part"] == RESIDUAL, "gap"].iat[0])
    measured = float(named["gap"].sum())
    top = named.head(3)
    biggest = ", ".join(
        f"**{LABELS[p]}** ({g:+.2f}, {g / total:.0%})"
        for p, g in zip(top["part"], top["gap"], strict=True)
    )
    out = [
        f"**{high}** scores **{total:.2f} points** higher than **{low}** "
        f"({gap['high'].sum():.2f} vs {gap['low'].sum():.2f}).",
        f"The largest parts of that gap are {biggest}.",
        f"Together the six measured factors account for **{measured:.2f} points "
        f"({measured / total:.0%})**; the remaining **{resid:+.2f}** is the residual, the part "
        "WHR's model does not explain.",
    ]
    behind = named[named["gap"] < -0.02]
    if len(behind):
        names = " and ".join(LABELS[p] for p in behind["part"])
        out.append(f"{low} is actually *ahead* on {names}, which narrows the gap slightly.")
    if total and abs(resid / total) > 0.25:
        out.append(
            "A residual this large means much of the difference is **not** captured by the six "
            "factors: unmeasured conditions, culture of answering, or noise in the survey."
        )
    elif total and abs(resid / total) < 0.1:
        out.append("The residual is small, so the six factors account for nearly all of it.")
    return out
