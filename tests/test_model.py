import importlib.util

import numpy as np
import pytest

from world_happiness.config import FACTORS
from world_happiness.model import AdditiveBaseline, HappinessModel
from world_happiness.plots import Figures as F


@pytest.fixture
def model(dataset):
    return HappinessModel(dataset.with_factors, seed=0)


def test_additive_baseline_is_exact_on_identity(dataset):
    df = dataset.with_factors
    x, y = df[list(FACTORS)], df[list(FACTORS)].sum(axis=1) + 1.0
    assert np.allclose(AdditiveBaseline().fit(x, y).predict(x), y)


def test_evaluate_has_both_schemes(model):
    m = model.evaluate()
    assert set(m.scheme) == {"temporal", "grouped"}
    assert set(m.model) == {"xgboost", "linear", "additive_identity"}
    assert m[["r2", "rmse", "mae"]].notna().all().all()


def test_grouped_split_never_leaks_countries(model):
    for scheme, (tr, te) in model.splits():
        if scheme == "grouped":
            assert not set(model.df.iso3.iloc[tr]) & set(model.df.iso3.iloc[te])


def test_shap_adds_up_to_prediction(model, tmp_path):
    shap = model.fit().explain()
    assert len(shap) == len(model.df) * len(FACTORS)
    pred = model.model.predict(model.x)
    per_row = shap.groupby(["iso3", "year"], sort=False).shap.sum().to_numpy()
    bias = pred - per_row
    assert np.allclose(bias, bias[0], atol=1e-3)  # prediction = bias + sum(SHAP)
    assert model.save(tmp_path / "m.json").exists()
    assert F.shap_importance(shap).data and F.shap_dependence(shap, "freedom").data


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("dowhy") is None, reason="dowhy extra not installed")
def test_causal(dataset):
    from world_happiness.causal import CausalAnalysis

    out = CausalAnalysis(dataset.with_factors).run(["freedom"])
    row = out.iloc[0]
    assert row.ci_lower <= row.ate <= row.ci_upper
    assert "gdp_per_capita" in row.adjustment_set
    assert row.direct_ci_lower <= row.direct_effect <= row.direct_ci_upper
    assert len(F.causal_effects(out).data) == 2
