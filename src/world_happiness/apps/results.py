"""Dash results dashboard: model performance, SHAP explanations and causal estimates."""

from __future__ import annotations

import dash_ag_grid as dag
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from dash.development.base_component import Component

from world_happiness.config import FACTORS, Settings
from world_happiness.plots import Figures as F

NOTE = (
    "Features are WHR 'explained-by' contributions, i.e. WHR's own regression split into parts. "
    "Predictive scores are real, but causal effects are only as good as the assumed DAG. "
    "If the decomposition held exactly, every *direct* effect would be 1."
)


def _load(settings: Settings) -> dict[str, pd.DataFrame]:
    out, files = settings.outputs, {"metrics": "metrics.csv", "causal": "causal_effects.csv"}
    data = {k: pd.read_csv(out / f) for k, f in files.items() if (out / f).exists()}
    if (out / "shap.parquet").exists():
        data["shap"] = pd.read_parquet(out / "shap.parquet")
    return data


def _table(df: pd.DataFrame) -> dag.AgGrid:
    return dag.AgGrid(
        rowData=df.round(3).to_dict("records"),
        columnDefs=[{"field": c, "sortable": True, "filter": True} for c in df.columns],
        columnSize="responsiveSizeToFit",
        dashGridOptions={"domLayout": "autoHeight"},
    )


def _card(title: str, *children: Component) -> html.Div:
    return html.Div(
        [html.H3(title), *children],
        style={
            "padding": "12px 16px",
            "border": "1px solid #e5e4df",
            "borderRadius": 8,
            "marginBottom": 16,
        },
    )


def create_app(settings: Settings | None = None) -> Dash:
    data = _load(settings or Settings())
    app = Dash(__name__, title="World Happiness Results")
    if "metrics" not in data:
        app.layout = html.Div([html.H2("No results yet"), html.P("Run `wh run` first.")])
        return app

    sections = [
        html.H1("World Happiness: model & causal results"),
        html.P(NOTE, style={"color": "#52514e"}),
        _card(
            "Model performance",
            dcc.Graph(figure=F.metrics(data["metrics"])),
            _table(data["metrics"]),
        ),
    ]
    if "shap" in data:
        sections += [
            _card(
                "Explanations (TreeSHAP)",
                dcc.Graph(figure=F.shap_importance(data["shap"])),
                dcc.Dropdown(list(FACTORS), FACTORS[1], id="feature", clearable=False),
                dcc.Graph(id="dependence"),
            ),
        ]
    if "causal" in data:
        sections += [
            _card(
                "Causal estimates (DoWhy, back-door)",
                dcc.Graph(figure=F.causal_effects(data["causal"])),
                _table(data["causal"]),
            )
        ]
    app.layout = html.Div(
        sections,
        style={"maxWidth": 1100, "margin": "auto", "padding": "0 16px", "fontFamily": "system-ui"},
    )
    if "shap" in data:

        @app.callback(Output("dependence", "figure"), Input("feature", "value"))
        def dependence(feature: str) -> object:
            return F.shap_dependence(data["shap"], feature)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8050)
