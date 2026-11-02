from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from world_happiness.config import FACTORS, RAW_COLUMNS, Settings
from world_happiness.data import WHRDataset

REAL = Path(__file__).parents[1] / "data" / "raw" / Settings().raw_file
NAMES = ["Finland", "Germany", "Nigeria", "Kosovo", "North Cyprus", "Brazil", "Japan", "India"]


def synthetic_raw(seed: int = 0) -> pd.DataFrame:
    """
    Figure-2.1-shaped data; factors missing before 2019, a gap in 2022, Swaziland->Eswatini.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2018, 2019, 2020, 2021, 2023):
        for i, name in enumerate([*NAMES, "Swaziland" if year < 2021 else "Eswatini"]):
            f = rng.uniform(0.05, 1.5, len(FACTORS)) if year >= 2019 else [np.nan] * len(FACTORS)
            resid = rng.uniform(0.5, 2.5)
            score = float(np.nansum(f) + resid) if year >= 2019 else rng.uniform(3, 8)
            rows.append(
                [
                    year,
                    i + 1,
                    name,
                    score,
                    score - 0.1,
                    score + 0.1,
                    *f,
                    resid if year >= 2019 else np.nan,
                ]
            )
    df = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    df["Rank"] = df.groupby("Year")[df.columns[3]].rank(ascending=False).astype(int)
    return df


@pytest.fixture
def raw_xlsx(tmp_path: Path) -> Path:
    path = tmp_path / "raw.xlsx"
    synthetic_raw().to_excel(path, index=False)
    return path


@pytest.fixture
def dataset(raw_xlsx: Path) -> WHRDataset:
    return WHRDataset.from_excel(raw_xlsx)


@pytest.fixture
def settings(tmp_path: Path, raw_xlsx: Path) -> Settings:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw_xlsx.rename(tmp_path / "data" / "raw" / "raw.xlsx")
    return Settings(root=tmp_path, raw_file="raw.xlsx")
