import numpy as np
import plotly.graph_objects as go
import pytest

from world_happiness.analysis import Explorer
from world_happiness.config import FACTORS, TARGET
from world_happiness.features import FeatureBuilder
from world_happiness.plots import Figures as F


@pytest.fixture
def features(dataset):
    return FeatureBuilder(dataset.df).lags().shares().build()


def test_lags_respect_year_gaps(features):
    fin = features[features.country == "Finland"].set_index("year")
    assert np.isnan(fin.loc[2018, f"{TARGET}_lag1"])
    assert fin.loc[2020, f"{TARGET}_lag1"] == fin.loc[2019, TARGET]
    assert np.isnan(fin.loc[2023, f"{TARGET}_lag1"])  # 2022 missing -> no lag


def test_shares_sum_to_one(features):
    shares = features.dropna(subset=list(FACTORS))[[f"{f}_share" for f in FACTORS]].sum(axis=1)
    assert np.allclose(shares, 1)


def test_explorer(dataset):
    ex = Explorer(dataset.df)
    assert ex.latest_year == 2023
    assert ex.top(3)[TARGET].is_monotonic_decreasing
    assert len(ex.country("FIN")) == len(ex.country("Finland")) == 5
    changes = ex.rank_changes(2019, 2023)
    assert (changes.rank_change == changes.rank_2019 - changes.rank_2023).all()
    assert ex.correlations().shape == (7, 7)
    assert "missing" in ex.summary()
    assert ex.column_stats("region").sum() == len(dataset.df)


def test_figures(dataset, features):
    ex, df = Explorer(dataset.df), dataset.with_factors
    figs = [
        F.world_map(df),
        F.regional_trends(ex.regional_trend()),
        F.factor_breakdown(df),
        F.country_trend(df),
        F.scatter(df, "freedom"),
        F.histogram(df, TARGET),
        F.correlation(ex.correlations()),
        F.rank_changes(ex.rank_changes(2019, 2023)),
    ]
    assert all(isinstance(f, go.Figure) and f.data for f in figs)
