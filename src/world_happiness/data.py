
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from .config import FACTORS, RAW_COLUMNS, RESIDUAL, TARGET
from .countries import CONTINENTS, REGIONS, country_table

log = logging.getLogger(__name__)

SCHEMA = pa.DataFrameSchema(
    {
        "year": pa.Column(int, pa.Check.in_range(2005, 2100)),
        "rank": pa.Column(int, pa.Check.ge(1)),
        "country": pa.Column(str),
        "iso3": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$")),
        "continent": pa.Column(str, pa.Check.isin(CONTINENTS)),
        "region": pa.Column(str, pa.Check.isin(REGIONS)),
        TARGET: pa.Column(float, pa.Check.in_range(0, 10)),
        "ci_lower": pa.Column(float, nullable=True),
        "ci_upper": pa.Column(float, nullable=True),
        **{f: pa.Column(float, pa.Check.ge(0), nullable=True) for f in FACTORS},
        RESIDUAL: pa.Column(float, nullable=True),
    },
    checks=pa.Check(
        lambda d: d.ci_lower.isna() | d.ci_lower.le(d[TARGET]) & d[TARGET].le(d.ci_upper),
        error="score must lie inside its confidence interval",
    ),
    unique=["iso3", "year"],
    strict=True,
    coerce=True,
)


class WHRDataset:
    """
    A validated country-year panel. Build it with :meth:`from_excel` or :meth:`load`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = SCHEMA.validate(df).sort_values(["year", "rank"]).reset_index(drop=True)

    @classmethod
    def from_excel(cls, path: Path | str) -> WHRDataset:
        raw = pd.read_excel(path).rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
        raw = raw.dropna(subset=["country", TARGET])
        meta = country_table(raw["country"]).rename(columns={"country": "canonical"})
        df = raw.merge(meta, left_on="country", right_on="raw_name", validate="many_to_one")
        df = df.assign(country=df.pop("canonical")).drop(columns="raw_name")
        dupes = df.duplicated(["iso3", "year"])
        if dupes.any():
            log.warning("Dropping %d duplicate country-years after renaming", dupes.sum())
        return cls(df[~dupes])

    @classmethod
    def load(cls, path: Path | str) -> WHRDataset:
        return cls(pd.read_parquet(path))

    def save(self, path: Path | str) -> Path:
        self.df.to_parquet(path, index=False)
        return Path(path)

    @property
    def years(self) -> list[int]:
        return sorted(self.df["year"].unique().tolist())

    @property
    def with_factors(self) -> pd.DataFrame:
        """
        Rows where WHR published the factor decomposition (2019 onwards).
        """
        return self.df.dropna(subset=list(FACTORS))

    def filter(
        self,
        years: Sequence[int] | None = None,
        regions: Sequence[str] | None = None,
        countries: Sequence[str] | None = None,
        continents: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        pairs = (
            ("year", years),
            ("region", regions),
            ("country", countries),
            ("continent", continents),
        )
        for col, values in pairs:
            if values:
                mask &= self.df[col].isin(values)
        return self.df[mask]
