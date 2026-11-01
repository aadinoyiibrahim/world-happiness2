"""Descriptive analysis: rankings, continent comparisons, trends, correlations."""

from __future__ import annotations

import pandas as pd

from .config import FACTORS, TARGET
from .countries import CONTINENTS

WORLD = "World"


class Explorer:
    """Read-only questions over the panel. Every method returns a DataFrame or Series."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def latest_year(self) -> int:
        return int(self.df["year"].max())

    @property
    def years(self) -> list[int]:
        return sorted(int(y) for y in self.df["year"].unique())

    def summary(self) -> pd.DataFrame:
        num = self.df.select_dtypes("number")
        return num.describe().T.assign(missing=num.isna().sum())

    def column_stats(self, col: str) -> pd.Series:
        s = self.df[col]
        if pd.api.types.is_numeric_dtype(s):
            return s.describe().rename(col)
        return s.value_counts().rename(col)

    # ------------------------------------------------------------------ rankings
    def scope(self, scope: str = WORLD, year: int | None = None) -> pd.DataFrame:
        """Rows for one year in the world or one continent, ranked within that scope.

        ``scope_rank`` is 1 for the happiest country *inside the scope*; ``rank`` stays the
        WHR world rank, so a continent's best country can have ``rank`` 1 and 20 elsewhere.
        """
        if scope != WORLD and scope not in CONTINENTS:
            raise ValueError(f"unknown scope {scope!r}; use {WORLD!r} or one of {CONTINENTS}")
        d = self.df[self.df["year"] == (year or self.latest_year)]
        if scope != WORLD:
            d = d[d["continent"] == scope]
        d = d.dropna(subset=[TARGET]).sort_values(TARGET, ascending=False)
        return d.assign(scope_rank=range(1, len(d) + 1)).reset_index(drop=True)

    def top(
        self,
        n: int = 5,
        year: int | None = None,
        bottom: bool = False,
        scope: str = WORLD,
    ) -> pd.DataFrame:
        """Happiest (or, with ``bottom``, least happy) ``n`` countries in a scope and year.

        Both directions are ordered best-first, so the last row of ``bottom=True`` is the
        least happy country. Asking for more countries than the scope holds returns them all.
        """
        d = self.scope(scope, year)
        return d.tail(n) if bottom else d.head(n)

    # ------------------------------------------------------- continent comparison
    def continent_summary(self, year: int | None = None) -> pd.DataFrame:
        """One row per continent: size, mean/median/spread of the score, best and worst."""
        d = self.scope(WORLD, year)
        g = d.groupby("continent", observed=True)
        best = d.loc[g[TARGET].idxmax(), ["continent", "country", TARGET]]
        worst = d.loc[g[TARGET].idxmin(), ["continent", "country", TARGET]]
        out = g[TARGET].agg(countries="count", mean="mean", median="median", std="std")
        out["gap"] = g[TARGET].max() - g[TARGET].min()
        out = out.join(best.set_index("continent").set_axis(["best", "best_score"], axis=1))
        out = out.join(worst.set_index("continent").set_axis(["worst", "worst_score"], axis=1))
        order = [c for c in CONTINENTS if c in out.index]
        return out.loc[order].reset_index()

    def continent_trend(self) -> pd.DataFrame:
        """Mean score per continent and year (over the countries surveyed that year)."""
        g = self.df.groupby(["continent", "year"], observed=True)[TARGET]
        return g.agg(["mean", "median", "count"]).reset_index()

    # ------------------------------------------------------------------- legacy
    def regional_trend(self) -> pd.DataFrame:
        g = self.df.groupby(["region", "year"])[TARGET]
        return g.agg(["mean", "std", "count"]).reset_index()

    def rank_changes(self, start: int, end: int) -> pd.DataFrame:
        cols = ["iso3", "country", "region", "rank", TARGET]
        a, b = (self.df.loc[self.df["year"] == y, cols] for y in (start, end))
        m = a.merge(b, on=["iso3", "country", "region"], suffixes=(f"_{start}", f"_{end}"))
        m["rank_change"] = m[f"rank_{start}"] - m[f"rank_{end}"]  # positive = improved
        m["score_change"] = m[f"{TARGET}_{end}"] - m[f"{TARGET}_{start}"]
        return m.sort_values("rank_change", ascending=False, ignore_index=True)

    def correlations(self, year: int | None = None) -> pd.DataFrame:
        d = self.df if year is None else self.df[self.df["year"] == year]
        return d[[TARGET, *FACTORS]].corr(method="spearman")

    def country(self, name: str) -> pd.DataFrame:
        key = "iso3" if len(name) == 3 and name.isupper() else "country"
        return self.df[self.df[key] == name].sort_values("year")
