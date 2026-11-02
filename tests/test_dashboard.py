import json
import re

import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from world_happiness import charts
from world_happiness.analysis import WORLD, Explorer
from world_happiness.cli import DASHBOARD
from world_happiness.config import TARGET
from world_happiness.countries import CONTINENTS, country_table
from world_happiness.livemap import VIEWS, build_frames, render_html


@pytest.fixture
def ex(dataset):
    return Explorer(dataset.df)


def test_continents_split_americas_and_cover_territories():
    t = country_table(["Brazil", "Mexico", "North Cyprus", "Somaliland Region", "Australia"])
    got = dict(zip(t.country, t.continent, strict=True))
    assert got == {
        "Australia": "Oceania",
        "Brazil": "South America",
        "Mexico": "North America",
        "North Cyprus": "Europe",
        "Somaliland Region": "Africa",
    }
    assert set(t.continent) <= set(CONTINENTS)


def test_single_name_country_table():
    assert country_table(["Japan"]).continent.tolist() == ["Asia"]


@pytest.mark.parametrize("scope", [WORLD, "Europe", "Africa"])
def test_top_and_bottom_are_ordered_and_scoped(ex, scope):
    top, bottom = ex.top(2, 2023, scope=scope), ex.top(2, 2023, bottom=True, scope=scope)
    assert top[TARGET].is_monotonic_decreasing and bottom[TARGET].is_monotonic_decreasing
    assert top.scope_rank.tolist() == [1, 2]
    full = ex.scope(scope, 2023)
    assert bottom.scope_rank.iat[-1] == len(full)
    assert top[TARGET].iat[0] == full[TARGET].max()
    if scope != WORLD:
        assert set(top.continent) == set(bottom.continent) == {scope}


def test_top_asks_for_more_than_exist(ex):
    assert len(ex.top(50, 2023, scope="South America")) == 1  # only Brazil


def test_unknown_scope_raises(ex):
    with pytest.raises(ValueError, match="unknown scope"):
        ex.scope("Atlantis")


def test_continent_summary_matches_scope(ex):
    s = ex.continent_summary(2023).set_index("continent")
    eu = ex.scope("Europe", 2023)
    assert s.loc["Europe", "countries"] == len(eu)
    assert s.loc["Europe", "best"] == eu.country.iat[0]
    assert s.loc["Europe", "worst"] == eu.country.iat[-1]
    assert s.loc["Europe", "mean"] == pytest.approx(eu[TARGET].mean())
    assert s["countries"].sum() == len(ex.scope(WORLD, 2023))
    assert list(s.index) == [c for c in CONTINENTS if c in s.index]


def test_continent_trend(ex):
    t = ex.continent_trend()
    assert set(t.columns) >= {"continent", "year", "mean", "count"}
    assert t["count"].sum() == len(ex.df)


def test_frames_rings_and_constant_locations(dataset):
    locs, frames = build_frames(dataset.df, "Europe", n=1)
    assert [f.year for f in frames] == sorted({2018, 2019, 2020, 2021, 2023})
    for f in frames:
        assert len(f.z) == len(f.customdata) == len(f.line_color) == len(locs)
        assert f.line_color.count("#0ca30c") == 1 and f.line_color.count("#d03b3b") == 1
        best = max(v for v in f.z if v is not None)
        assert f.top[0][1] == pytest.approx(best, abs=0.006)


def test_small_scope_never_rings_a_country_twice(dataset):
    _, frames = build_frames(dataset.df, "South America", n=5)
    assert all(f.bottom == [] and len(f.top) == 1 for f in frames)


@pytest.mark.parametrize("mode", ["timeline", "globe"])
def test_render_html_is_self_contained(dataset, mode):
    html = render_html(dataset.df, "Europe", 2, 2021, mode)
    payload = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    assert payload["mode"] == mode and payload["frames"][payload["start"]]["year"] == 2021
    proj = payload["fig"]["layout"]["geo"]["projection"]["type"]
    assert proj == ("orthographic" if mode == "globe" else "natural earth")
    assert "__" not in re.sub(r"const D = .*?;\n", "", html, flags=re.S)  # all filled in


def test_render_html_rejects_bad_scope(dataset):
    with pytest.raises(ValueError):
        render_html(dataset.df, "Mars")
    assert set(VIEWS) == {WORLD, *CONTINENTS}


def test_charts(ex):
    figs = [
        charts.ranking_bars(ex.top(3, 2023), False, "t"),
        charts.ranking_bars(ex.top(3, 2023, bottom=True), True, "t"),
        charts.continent_range(ex.continent_summary(2023)),
        charts.continent_distribution(ex.scope(WORLD, 2023), list(CONTINENTS)),
        charts.continent_trends(ex.continent_trend(), list(CONTINENTS)),
    ]
    assert all(isinstance(f, go.Figure) and f.data for f in figs)
    bars = figs[0].data[0]
    assert list(bars.y)[-1].startswith("1. ")  # best country drawn at the top


@pytest.fixture
def app(settings, monkeypatch):
    monkeypatch.setenv("WH_ROOT", str(settings.root))
    monkeypatch.setenv("WH_RAW_FILE", settings.raw_file)
    return AppTest.from_file(str(DASHBOARD), default_timeout=120)


def test_dashboard_buttons(app):
    at = app.run()
    assert not at.exception
    assert at.metric[0].value == "9"  # countries surveyed in 2023 (world)
    at.button_group(key="scope").set_value("Europe").run()
    assert not at.exception
    assert at.query_params["scope"] == ["Europe"]
    at.button_group(key="show").set_value("worst").run()
    at.button_group(key="map").set_value("globe").run()
    at.button_group(key="n").set_value(3).run()
    assert not at.exception
    assert at.query_params["show"] == ["worst"] and at.query_params["map"] == ["globe"]


def test_dashboard_reads_url(app):
    app.query_params.update({"scope": "Africa", "show": "worst", "year": "2020"})
    at = app.run()
    assert not at.exception
    assert at.button_group(key="scope").value == "Africa"
    assert at.select_slider(key="year").value == 2020


def test_dashboard_ignores_bad_url(app):
    app.query_params.update({"scope": "Narnia", "n": "999", "year": "1066"})
    at = app.run()
    assert not at.exception and at.button_group(key="scope").value == WORLD


from world_happiness import theme as T  # noqa: E402
from world_happiness.analysis import PARTS  # noqa: E402
from world_happiness.insights import explain_gap  # noqa: E402


def test_factor_means_add_up_and_include_world(ex):
    m = ex.factor_means(2023).set_index("group")
    assert WORLD in m.index and set(m.index) - {WORLD} <= set(CONTINENTS)
    # synthetic scores are built as sum(parts), so the identity is exact here
    assert m[PARTS].sum(axis=1).to_numpy() == pytest.approx(m[TARGET].to_numpy())
    assert m.loc[WORLD, "countries"] == m.drop(index=WORLD)["countries"].sum()


def test_gap_splits_exactly(ex):
    m = ex.factor_means(2023).set_index("group")
    g = ex.gap(m.loc["Europe"], m.loc["Africa"])
    assert list(g.part) == PARTS
    assert g.gap.sum() == pytest.approx(m.loc["Europe", TARGET] - m.loc["Africa", TARGET])
    assert g.share.sum() == pytest.approx(1)


def test_factor_year_falls_back_to_nearest_decomposed_year(ex):
    assert ex.factor_years == [2019, 2020, 2021, 2023]
    assert ex.factor_year(2018) == 2019 and ex.factor_year(2022) == 2023
    assert ex.factor_year(2021) == 2021


def test_gap_trend_has_every_part_every_year(ex):
    t = ex.continent_gap_trend("Europe", "Africa")
    assert len(t) == len(PARTS) * len(ex.factor_years)


def test_explain_gap_mentions_groups_and_residual(ex):
    m = ex.factor_means(2023).set_index("group")
    lines = explain_gap(ex.gap(m.loc["Europe"], m.loc["Africa"]), "Europe", "Africa")
    text = " ".join(lines)
    assert "**Europe**" in text and "**Africa**" in text and "residual" in text
    assert 3 <= len(lines) <= 5


def test_why_charts(ex):
    means = ex.factor_means(2023)
    m = means.set_index("group")
    gap = ex.gap(m.loc["Europe"], m.loc["Africa"])
    figs = [
        charts.decomposition_bars(means, ["Europe", "Africa"], T.LIGHT),
        charts.gap_waterfall(gap, "Europe", "Africa", T.LIGHT),
        charts.gap_trend(ex.continent_gap_trend("Europe", "Africa"), "Europe", "Africa"),
        charts.factor_profile(ex.decomposed(2023), ["Europe", "Africa"]),
        charts.country_trends(ex.df, ["Finland", "Nigeria"], ex.continent_trend()),
    ]
    assert all(isinstance(f, go.Figure) and f.data for f in figs)
    wf = figs[1].data[0]
    assert sum(wf.x[1:-1]) == pytest.approx(gap.gap.sum())  # steps reach the high score


def test_palettes_differ_and_map_follows_theme(dataset):
    assert T.palette("light") is T.LIGHT and T.palette(None) is T.DARK
    assert T.LIGHT.page != T.DARK.page
    light = render_html(dataset.df, "Europe", 2, 2021, "timeline", theme="light")
    assert "color-scheme: light" in light and T.LIGHT.ocean in light


def test_dashboard_why_and_data_tabs(app):
    at = app.run()
    assert not at.exception
    assert len(at.tabs) == 4 and at.tabs[2].label.startswith("🔍")
    assert any("higher than" in m.value for m in at.markdown)  # narrative rendered
    at.selectbox(key="why_low").set_value("Asia").run()
    assert not at.exception
    at.button_group(key="why_continent").set_value("Europe").run()
    assert not at.exception
    at.button_group(key="scope").set_value("Africa").run()
    assert not at.exception
    # Data tab follows the scope + year and hides empty columns
    data = at.tabs[3].dataframe[0].value
    assert set(data.continent) == {"Africa"} and set(data.year) == {2023}
    assert not data.isna().all().any()


from world_happiness.export import frame_figure  # noqa: E402
from world_happiness.livemap import color_range  # noqa: E402


def test_export_frame_has_year_and_ranking_card(dataset):
    locs, frames = build_frames(dataset.df, WORLD, 2)
    fig = frame_figure(locs, frames[-1], WORLD, "globe", color_range(dataset.df, WORLD), T.DARK, 40)
    texts = [a.text for a in fig.layout.annotations]
    assert texts[0] == "<b>2023</b>" and "HAPPIEST" in texts[2] and "LEAST HAPPY" in texts[2]
    assert fig.layout.geo.projection.rotation.lon == 40
