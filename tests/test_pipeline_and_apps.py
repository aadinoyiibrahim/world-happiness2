import pandas as pd
import pytest

from world_happiness import cli
from world_happiness.apps.results import create_app
from world_happiness.pipeline import Pipeline


@pytest.fixture
def ran(settings):
    Pipeline(settings).run(causal=False)
    return settings


def test_pipeline_writes_artifacts(ran):
    names = {p.name for p in ran.outputs.iterdir()}
    assert {
        "panel.parquet",
        "features.parquet",
        "metrics.csv",
        "shap.parquet",
        "xgboost.json",
        "manifest.json",
    } <= names
    assert len(pd.read_csv(ran.outputs / "metrics.csv")) == 6


def test_dash_app_serves(ran):
    client = create_app(ran).server.test_client()
    assert client.get("/").status_code == 200
    assert b"Model performance" in client.get("/_dash-layout").data


def test_dash_app_without_results(settings):
    assert b"No results yet" in create_app(settings).server.test_client().get("/_dash-layout").data


def test_cli_prepare(settings, monkeypatch):
    monkeypatch.setenv("WH_ROOT", str(settings.root))
    monkeypatch.setenv("WH_RAW_FILE", settings.raw_file)
    cli.main(["prepare"])
    assert (settings.outputs / "panel.parquet").exists()


@pytest.mark.slow
def test_pipeline_causal_then_skip_removes_stale(settings):
    pytest.importorskip("dowhy")
    Pipeline(settings).run(causal=True)
    effects = pd.read_csv(settings.outputs / "causal_effects.csv")
    assert {"ate", "direct_effect", "direct_ci_lower"} <= set(effects.columns)
    Pipeline(settings).run(causal=False)
    assert not (settings.outputs / "causal_effects.csv").exists()
