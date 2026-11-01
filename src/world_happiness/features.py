"""Panel-aware feature engineering (lags respect gaps in the yearly series)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .config import FACTORS, TARGET


class FeatureBuilder:
    """Chainable builder: ``FeatureBuilder(df).lags().shares().build()``."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    def lags(self, cols: Sequence[str] = (TARGET, "gdp_per_capita")) -> FeatureBuilder:
        """Previous-year value and change; NaN when the previous year is missing (e.g. 2013)."""
        g = self.df.groupby("iso3")
        consecutive = self.df["year"].sub(g["year"].shift()).eq(1)
        for c in cols:
            self.df[f"{c}_lag1"] = g[c].shift().where(consecutive)
            self.df[f"{c}_delta"] = self.df[c] - self.df[f"{c}_lag1"]
        return self

    def shares(self) -> FeatureBuilder:
        """Share of the *explained* part per factor. Descriptive only, never a model input."""
        explained = self.df[list(FACTORS)].sum(axis=1, min_count=len(FACTORS))
        self.df["explained"] = explained
        for f in FACTORS:
            self.df[f"{f}_share"] = self.df[f] / explained
        return self

    def build(self) -> pd.DataFrame:
        return self.df
