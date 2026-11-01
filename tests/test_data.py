import pandas as pd
import pandera.errors as pe
import pytest

from world_happiness.config import FACTORS, TARGET
from world_happiness.countries import REGIONS, country_table, whr_region
from world_happiness.data import WHRDataset

from .conftest import REAL


def test_loads_and_harmonises(dataset):
    df = dataset.df
    assert set(df.columns) >= {"iso3", "region", "continent", TARGET, *FACTORS}
    assert df.region.isin(REGIONS).all()
    assert not df.duplicated(["iso3", "year"]).any()
    assert set(df.loc[df.iso3 == "SWZ", "country"]) == {"Eswatini"}  # alias merged
    assert df.loc[df.country == "North Cyprus", "iso3"].iat[0] == "CYN"


def test_with_factors_drops_pre_2019(dataset):
    assert dataset.with_factors.year.min() == 2019


def test_filter(dataset):
    out = dataset.filter(years=[2020], regions=["Western Europe"])
    assert set(out.year) == {2020} and set(out.region) == {"Western Europe"}


def test_roundtrip(dataset, tmp_path):
    again = WHRDataset.load(dataset.save(tmp_path / "p.parquet"))
    assert again.df.equals(dataset.df)


def test_schema_rejects_bad_scores(dataset):
    bad = dataset.df.copy()
    bad.loc[0, TARGET] = 42
    with pytest.raises(pe.SchemaError):
        WHRDataset(bad)


def test_schema_rejects_duplicates(dataset):
    df = dataset.df
    with pytest.raises(pe.SchemaError):
        WHRDataset(pd.concat([df, df.head(1)], ignore_index=True))


@pytest.mark.parametrize(
    ("iso", "sub", "cont", "expected"),
    [
        ("GRC", "Southern Europe", "Europe", "Western Europe"),
        ("HRV", "Southern Europe", "Europe", "Central and Eastern Europe"),
        ("GEO", "Western Asia", "Asia", "Commonwealth of Independent States"),
        ("TUR", "Western Asia", "Asia", "Middle East and North Africa"),
        ("IRN", "Southern Asia", "Asia", "Middle East and North Africa"),
        ("CYP", "Western Asia", "Asia", "Western Europe"),
        ("MEX", "Central America", "America", "Latin America and Caribbean"),
        ("SDN", "Northern Africa", "Africa", "Sub-Saharan Africa"),
        ("AUS", "Australia and New Zealand", "Oceania", "North America and ANZ"),
    ],
)
def test_whr_region(iso, sub, cont, expected):
    assert whr_region(iso, sub, cont) == expected


@pytest.mark.skipif(not REAL.exists(), reason="raw WHR file not present")
def test_real_file_every_country_mapped():
    ds = WHRDataset.from_excel(REAL)
    assert ds.df.iso3.nunique() >= 160
    assert len(country_table(["Swaziland", "Eswatini"]).iso3.unique()) == 1
