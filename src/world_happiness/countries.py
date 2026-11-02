from __future__ import annotations

from collections.abc import Iterable

import country_converter as coco
import pandas as pd

ALIASES = {"Swaziland": "Eswatini"}  # historical names folded into the current one
TERRITORIES = {
    "North Cyprus": ("CYN", "Europe", "Europe"),
    "Somaliland Region": ("SOL", "Africa", "Africa"),
}

CONTINENTS = ("Africa", "Asia", "Europe", "North America", "South America", "Oceania")

REGIONS = (
    "Western Europe",
    "Central and Eastern Europe",
    "Commonwealth of Independent States",
    "North America and ANZ",
    "Latin America and Caribbean",
    "Middle East and North Africa",
    "Sub-Saharan Africa",
    "South Asia",
    "Southeast Asia",
    "East Asia",
)
_CIS = {"ARM", "AZE", "BLR", "GEO", "KAZ", "KGZ", "MDA", "RUS", "TJK", "TKM", "UKR", "UZB"}
_CEE = {"EST", "LVA", "LTU", "ALB", "BIH", "HRV", "XKX", "MNE", "MKD", "SRB", "SVN"}
_WEST = {"CYP", "CYN"}
_ANZ = {"USA", "CAN", "AUS", "NZL"}
_SSA = {"SDN"}  # UN puts Sudan in Northern Africa; WHR in Sub-Saharan Africa


def whr_region(iso3: str, subregion: str | None, continent: str) -> str:
    """
    Map a country to the ten regions used in the World Happiness Report.
    """
    if iso3 in _ANZ:
        return "North America and ANZ"
    if iso3 in _CIS:
        return "Commonwealth of Independent States"
    if iso3 in _CEE or subregion == "Eastern Europe":
        return "Central and Eastern Europe"
    if iso3 in _WEST or continent == "Europe":
        return "Western Europe"
    if iso3 in _SSA:
        return "Sub-Saharan Africa"
    if continent == "America":
        return "Latin America and Caribbean"
    if iso3 == "IRN" or subregion in {"Northern Africa", "Western Asia"}:
        return "Middle East and North Africa"
    if continent == "Africa":
        return "Sub-Saharan Africa"
    return {
        "Southern Asia": "South Asia",
        "South-eastern Asia": "Southeast Asia",
        "Eastern Asia": "East Asia",
    }.get(subregion or "", "Unmapped")


def _as_list(x: str | list[str]) -> list[str]:
    """
    ``coco.convert`` returns a bare string for a single input; always get a list.
    """
    return [x] if isinstance(x, str) else list(x)


def country_table(names: Iterable[str]) -> pd.DataFrame:
    """
    One row per raw name with canonical ``country``, ``iso3``, ``continent`` and ``region``.

    ``continent`` is one of :data:`CONTINENTS` (the Americas are split into North and South);
    ``region`` is one of the ten WHR regions.
    """
    raw = sorted(set(names))
    canon = [ALIASES.get(n, n) for n in raw]
    known = [c for c in canon if c not in TERRITORIES]
    cc = coco.CountryConverter()
    iso = _as_list(cc.convert(known, to="ISO3", not_found=None))
    meta: dict[str, tuple[str, str | None, str, str]] = {
        c: (i, sub, cont, cont7)
        for c, i, sub, cont, cont7 in zip(
            known,
            iso,
            _as_list(cc.convert(known, to="UNregion", not_found=None)),
            _as_list(cc.convert(known, to="continent", not_found=None)),
            _as_list(cc.convert(iso, src="ISO3", to="Continent_7", not_found=None)),
            strict=True,
        )
    }
    meta |= {c: (i, None, cont, cont7) for c, (i, cont, cont7) in TERRITORIES.items()}
    rows = [(r, c, *meta[c]) for r, c in zip(raw, canon, strict=True)]
    cols = ["raw_name", "country", "iso3", "subregion", "un_continent", "continent"]
    df = pd.DataFrame(rows, columns=cols)
    df["region"] = [
        whr_region(*t) for t in zip(df.iso3, df.subregion, df.un_continent, strict=True)
    ]
    return df.drop(columns=["subregion", "un_continent"])
