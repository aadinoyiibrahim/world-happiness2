"""Paths, column names and constants shared by the whole package."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TARGET = "happiness_score"
FACTORS = (
    "gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "perceptions_of_corruption",
)
RESIDUAL = "dystopia_residual"

# WHR "Data for Figure 2.1" header -> tidy name
RAW_COLUMNS = {
    "Year": "year",
    "Rank": "rank",
    "Country name": "country",
    "Life evaluation (3-year average)": TARGET,
    "Lower whisker": "ci_lower",
    "Upper whisker": "ci_upper",
    "Explained by: Log GDP per capita": "gdp_per_capita",
    "Explained by: Social support": "social_support",
    "Explained by: Healthy life expectancy": "healthy_life_expectancy",
    "Explained by: Freedom to make life choices": "freedom",
    "Explained by: Generosity": "generosity",
    "Explained by: Perceptions of corruption": "perceptions_of_corruption",
    "Dystopia + residual": RESIDUAL,
}


@dataclass(frozen=True)
class Settings:
    """Where things live. Override with ``WH_ROOT`` / ``WH_RAW_FILE`` env vars."""

    root: Path = field(default_factory=lambda: Path(os.getenv("WH_ROOT", ".")).resolve())
    raw_file: str = field(
        default_factory=lambda: os.getenv("WH_RAW_FILE", "WHR26_Data_Figure_2.1.xlsx")
    )
    seed: int = 42

    @property
    def raw_path(self) -> Path:
        return self.root / "data" / "raw" / self.raw_file

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def output(self, name: str) -> Path:
        self.outputs.mkdir(parents=True, exist_ok=True)
        return self.outputs / name
