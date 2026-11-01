"""Back-door causal estimates with DoWhy (optional extra: ``pip install .[causal]``)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from .config import FACTORS, TARGET

# Assumed DAG: income and health shape the other factors; every factor affects the score.
EDGES: tuple[tuple[str, str], ...] = (
    *((f, TARGET) for f in FACTORS),
    *(("gdp_per_capita", f) for f in FACTORS if f != "gdp_per_capita"),
    *(("healthy_life_expectancy", f) for f in ("social_support", "freedom")),
)


class CausalAnalysis:
    """Total (back-door) and direct effect of each factor on the score, plus a placebo test."""

    def __init__(self, df: pd.DataFrame, edges: Sequence[tuple[str, str]] = EDGES, seed: int = 42):
        self.df = df.dropna(subset=[*FACTORS, TARGET]).reset_index(drop=True)
        assert "iso3" in self.df, "needs iso3 for clustered errors"
        self.edges = list(edges)
        self.seed = seed

    @property
    def graph(self):  # noqa: ANN201 - networkx.DiGraph (networkx ships with dowhy)
        import networkx as nx

        return nx.DiGraph(self.edges)

    def estimate(self, treatment: str) -> dict[str, float | str]:
        from dowhy import CausalModel

        logging.getLogger("dowhy").setLevel(logging.ERROR)
        cm = CausalModel(
            self.df[list(self.graph.nodes)], treatment=treatment, outcome=TARGET, graph=self.graph
        )
        estimand = cm.identify_effect(proceed_when_unidentifiable=True)
        est = cm.estimate_effect(
            estimand, method_name="backdoor.linear_regression", effect_modifiers=[]
        )  # additive model: one ATE, no interactions
        placebo = cm.refute_estimate(
            estimand,
            est,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=10,
            random_seed=self.seed,
        )
        adjust = sorted(estimand.get_backdoor_variables())
        others = [f for f in FACTORS if f != treatment]
        _, lo, hi = self._ols(treatment, adjust)  # clustered CI; DoWhy's bootstrap is very slow
        direct, dlo, dhi = self._ols(treatment, others)
        return {
            "treatment": treatment,
            "ate": float(est.value),  # total effect (includes paths via other factors)
            "ci_lower": lo,
            "ci_upper": hi,
            "direct_effect": direct,  # all other factors held fixed
            "direct_ci_lower": dlo,
            "direct_ci_upper": dhi,
            "placebo_effect": float(placebo.new_effect),
            "adjustment_set": ", ".join(adjust),
        }

    def _ols(self, treatment: str, adjust: Sequence[str]) -> tuple[float, float, float]:
        """OLS coefficient and 95% CI with errors clustered by country (panel rows overlap)."""
        import statsmodels.formula.api as smf

        groups = pd.factorize(self.df["iso3"])[0]
        fit = smf.ols(f"{TARGET} ~ {' + '.join([treatment, *adjust])}", data=self.df).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        lo, hi = fit.conf_int().loc[treatment]
        return float(fit.params[treatment]), float(lo), float(hi)

    def run(self, treatments: Sequence[str] = FACTORS) -> pd.DataFrame:
        return pd.DataFrame([self.estimate(t) for t in treatments])
