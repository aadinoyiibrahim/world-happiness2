"""Plotly figure factory shared by both dashboards (one look, one palette)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import FACTORS, RESIDUAL, TARGET

# Validated categorical order (fixed, never cycled) + one-hue sequential + blue/red diverging.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#d03b3b", "#f0efec", "#2a78d6"]
FACTOR_COLORS = dict(zip([*FACTORS, RESIDUAL], CATEGORICAL, strict=True))
LABELS = {c: c.replace("_", " ").capitalize() for c in [TARGET, *FACTORS, RESIDUAL]}
TEMPLATE = "plotly_white"


def _style(fig: go.Figure, title: str) -> go.Figure:
    return fig.update_layout(
        template=TEMPLATE,
        title=title,
        hovermode="closest",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )


class Figures:
    """Every chart takes a tidy DataFrame and returns a ``go.Figure``."""

    @staticmethod
    def world_map(df: pd.DataFrame, col: str = TARGET) -> go.Figure:
        d = df.dropna(subset=[col]).sort_values("year")
        fig = px.choropleth(
            d,
            locations="iso3",
            color=col,
            hover_name="country",
            animation_frame="year",
            color_continuous_scale=SEQUENTIAL,
            range_color=(d[col].min(), d[col].max()),
            labels=LABELS,
        )
        return _style(
            fig.update_geos(showframe=False, projection_type="natural earth"),
            f"{LABELS.get(col, col)} by country",
        )

    @staticmethod
    def regional_trends(trend: pd.DataFrame) -> go.Figure:
        """Small multiples (10 regions exceed a safe categorical palette)."""
        fig = px.line(
            trend,
            x="year",
            y="mean",
            facet_col="region",
            facet_col_wrap=5,
            markers=True,
            color_discrete_sequence=CATEGORICAL[:1],
            labels={"mean": "Mean score"},
            height=520,
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        return _style(fig, "Mean life evaluation by region")

    @staticmethod
    def factor_breakdown(df: pd.DataFrame) -> go.Figure:
        cols = [*FACTORS, RESIDUAL]
        d = df.sort_values(TARGET).melt(["country"], cols, var_name="factor", value_name="points")
        fig = px.bar(
            d,
            x="points",
            y="country",
            color="factor",
            orientation="h",
            color_discrete_map=FACTOR_COLORS,
            labels=LABELS,
            height=max(350, 28 * df["country"].nunique()),
        )
        return _style(
            fig.update_traces(marker_line_width=2, marker_line_color="white"),
            "What the score is made of (WHR decomposition)",
        )

    @staticmethod
    def country_trend(df: pd.DataFrame) -> go.Figure:
        fig = px.line(
            df,
            x="year",
            y=TARGET,
            color="country",
            markers=True,
            color_discrete_sequence=CATEGORICAL,
            labels=LABELS,
        )
        for _, g in df.groupby("country"):
            fig.add_scatter(
                x=g.year,
                y=g.ci_upper,
                mode="lines",
                line_width=0,
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_scatter(
                x=g.year,
                y=g.ci_lower,
                mode="lines",
                line_width=0,
                fill="tonexty",
                fillcolor="rgba(82,81,78,0.12)",
                showlegend=False,
                hoverinfo="skip",
            )
        return _style(fig, "Life evaluation over time (band = 95% CI)")

    @staticmethod
    def scatter(df: pd.DataFrame, x: str, y: str = TARGET) -> go.Figure:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            hover_name="country",
            hover_data=["region", "year"],
            trendline="ols",
            trendline_color_override="#52514e",
            color_discrete_sequence=CATEGORICAL[:1],
            labels=LABELS,
        )
        return _style(fig, f"{LABELS.get(y, y)} vs {LABELS.get(x, x)}")

    @staticmethod
    def histogram(df: pd.DataFrame, col: str) -> go.Figure:
        fig = px.histogram(
            df, x=col, nbins=40, color_discrete_sequence=CATEGORICAL[:1], labels=LABELS
        )
        return _style(
            fig.update_traces(marker_line_width=2, marker_line_color="white"),
            f"Distribution of {LABELS.get(col, col)}",
        )

    @staticmethod
    def correlation(corr: pd.DataFrame) -> go.Figure:
        fig = px.imshow(
            corr.rename(index=LABELS, columns=LABELS),
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale=DIVERGING,
            aspect="auto",
        )
        return _style(fig, "Spearman correlation")

    @staticmethod
    def rank_changes(changes: pd.DataFrame, n: int = 10) -> go.Figure:
        d = pd.concat([changes.head(n), changes.tail(n)]).sort_values("rank_change")
        fig = px.bar(
            d,
            x="rank_change",
            y="country",
            orientation="h",
            color="rank_change",
            color_continuous_scale=DIVERGING,
            color_continuous_midpoint=0,
            labels={"rank_change": "Ranks gained"},
            height=560,
        )
        return _style(fig, f"Biggest {n} risers and fallers")

    @staticmethod
    def metrics(metrics: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            metrics,
            x="model",
            y="r2",
            color="model",
            facet_col="scheme",
            color_discrete_sequence=CATEGORICAL,
            text_auto=".2f",
            labels={"r2": "R² (held-out)"},
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + " validation"))
        return _style(fig, "Out-of-sample R²: XGBoost vs baselines")

    @staticmethod
    def shap_importance(shap: pd.DataFrame) -> go.Figure:
        imp = shap.assign(abs=shap["shap"].abs()).groupby("feature", as_index=False)["abs"].mean()
        fig = px.bar(
            imp.sort_values("abs"),
            x="abs",
            y="feature",
            orientation="h",
            color_discrete_sequence=CATEGORICAL[:1],
            labels={"abs": "Mean |SHAP|"},
        )
        return _style(fig, "Global feature importance")

    @staticmethod
    def shap_dependence(shap: pd.DataFrame, feature: str) -> go.Figure:
        d = shap[shap["feature"] == feature]
        fig = px.scatter(
            d,
            x="value",
            y="shap",
            hover_name="country",
            hover_data=["year"],
            color_discrete_sequence=CATEGORICAL[:1],
            labels={"value": LABELS.get(feature, feature), "shap": "SHAP value"},
        )
        return _style(fig.add_hline(y=0, line_color="#b5b4ad"), f"SHAP dependence: {feature}")

    @staticmethod
    def causal_effects(effects: pd.DataFrame) -> go.Figure:
        """Total (back-door) and direct effects with 95% clustered CIs."""
        d, fig = effects.sort_values("ate"), go.Figure()
        for (name, col, lo, hi), color in zip(
            [
                ("Total effect", "ate", "ci_lower", "ci_upper"),
                ("Direct effect", "direct_effect", "direct_ci_lower", "direct_ci_upper"),
            ],
            CATEGORICAL[:2],
            strict=True,
        ):
            fig.add_scatter(
                x=d[col],
                y=d.treatment,
                name=name,
                mode="markers",
                orientation="h",
                offsetgroup=name,
                marker={"size": 10, "color": color, "line": {"width": 2, "color": "white"}},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": d[hi] - d[col],
                    "arrayminus": d[col] - d[lo],
                },
            )
        fig.add_vline(x=1, line_dash="dash", line_color="#52514e")
        fig.add_annotation(
            x=1,
            y=1.06,
            yref="paper",
            showarrow=False,
            font_color="#52514e",
            text="direct effect = 1 if the WHR decomposition holds",
        )
        fig.add_vline(x=0, line_color="#b5b4ad")
        return _style(
            fig.update_layout(scattermode="group"), "Effect of +1 explained point on the score"
        )
