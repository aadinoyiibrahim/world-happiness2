
from __future__ import annotations

import pandas as pd

from .config import FACTORS, RESIDUAL, TARGET
from .countries import CONTINENTS

WORLD = "World"
PARTS = [*FACTORS, RESIDUAL]  

class Explorer:
    """
    Read-only questions over the panel. 
    Every method returns a DataFrame or Series.
    """

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

    def scope(self, scope: str = WORLD, year: int | None = None) -> pd.DataFrame:
        """
        Rows for one year in the world or one continent, 
        ranked within that scope.

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
        """
        Happiest (or, with ``bottom``, least happy) ``n`` 
        countries in a scope and year.

        Both directions are ordered best-first, so the last row of ``bottom=True`` is the
        least happy country. Asking for more countries than the scope holds returns them all.
        """
        d = self.scope(scope, year)
        return d.tail(n) if bottom else d.head(n)

    def continent_summary(self, year: int | None = None) -> pd.DataFrame:
        """
        One row per continent: size, mean/median/spread of the score, 
        best and worst.
        """
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
        """
        Mean score per continent and year 
        (over the countries surveyed that year).
        """
        g = self.df.groupby(["continent", "year"], observed=True)[TARGET]
        return g.agg(["mean", "median", "count"]).reset_index()

    @property
    def factor_years(self) -> list[int]:
        """
        Years in which WHR published the six-factor 
        decomposition (2019 onwards).
        """
        full = self.df.dropna(subset=list(FACTORS))
        return sorted(int(y) for y in full["year"].unique())

    def factor_year(self, year: int) -> int:
        """
        ``year`` if it has the decomposition, 
        else the nearest year that does.
        """
        years = self.factor_years
        if not years:
            raise ValueError("no year with the factor decomposition")
        return min(years, key=lambda y: (abs(y - year), -y))

    def decomposed(self, year: int, scope: str = WORLD) -> pd.DataFrame:
        """
        Country rows with all six factors, 
        for one year and scope, best first.
        """
        d = self.scope(scope, year).dropna(subset=list(FACTORS))
        return d.assign(scope_rank=range(1, len(d) + 1)).reset_index(drop=True)

    def factor_means(self, year: int) -> pd.DataFrame:
        """
        Mean score and mean contribution of every factor 
        per continent (plus the world).

        Means are over countries with the full decomposition. Because each country's score is
        the sum of its parts, the mean score is the sum of the mean parts (up to WHR's
        rounding of about 0.003 per country).
        """
        d = self.decomposed(year)
        cols = [TARGET, *PARTS]
        g = d.groupby("continent", observed=True)[cols].mean()
        g["countries"] = d.groupby("continent", observed=True).size()
        world = d[cols].mean().to_frame(WORLD).T.assign(countries=len(d))
        order = [c for c in CONTINENTS if c in g.index]
        return pd.concat([g.loc[order], world]).rename_axis("group").reset_index()

    @staticmethod
    def gap(high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Split ``high[score] - low[score]`` into the gap 
        in each part of the decomposition.

        ``gap`` sums to the score gap; ``share`` is each part's fraction of it (can be
        negative when the lower group is ahead on that part).
        """
        total = float(high[TARGET] - low[TARGET])
        out = pd.DataFrame(
            {
                "part": PARTS,
                "high": [float(high[p]) for p in PARTS],
                "low": [float(low[p]) for p in PARTS],
            }
        )
        out["gap"] = out["high"] - out["low"]
        out["share"] = out["gap"] / total if total else float("nan")
        return out

    def continent_gap_trend(self, high: str, low: str) -> pd.DataFrame:
        """
        Per-part gap between two continents' means, 
        for every decomposition year.
        """
        rows = []
        for y in self.factor_years:
            m = self.factor_means(y).set_index("group")
            if high in m.index and low in m.index:
                g = self.gap(m.loc[high], m.loc[low]).assign(year=y)
                rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

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
