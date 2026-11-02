from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from .config import Settings
from .data import WHRDataset
from .features import FeatureBuilder

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    settings: Settings = field(default_factory=Settings)

    def prepare(self) -> WHRDataset:
        ds = WHRDataset.from_excel(self.settings.raw_path)
        ds.save(self.settings.output("panel.parquet"))
        FeatureBuilder(ds.df).lags().shares().build().to_parquet(
            self.settings.output("features.parquet"), index=False
        )
        log.info("Panel: %d rows, %d countries", len(ds.df), ds.df.iso3.nunique())
        return ds

    def train(self, ds: WHRDataset) -> pd.DataFrame:
        from .model import HappinessModel

        model = HappinessModel(ds.with_factors, seed=self.settings.seed)
        metrics = model.evaluate()
        metrics.to_csv(self.settings.output("metrics.csv"), index=False)
        model.fit().save(self.settings.output("xgboost.json"))
        model.explain().to_parquet(self.settings.output("shap.parquet"), index=False)
        log.info("Metrics:\n%s", metrics.round(3).to_string(index=False))
        return metrics

    def causal(self, ds: WHRDataset) -> pd.DataFrame | None:
        self.settings.output("causal_effects.csv").unlink(missing_ok=True)
        try:
            from .causal import CausalAnalysis

            effects = CausalAnalysis(ds.with_factors, seed=self.settings.seed).run()
        except ImportError:
            log.warning("dowhy not installed; skipping causal step (pip install '.[causal]')")
            return None
        effects.to_csv(self.settings.output("causal_effects.csv"), index=False)
        return effects

    def run(self, causal: bool = True) -> None:
        ds = self.prepare()
        self.train(ds)
        if causal:
            self.causal(ds)
        else:
            self.settings.output("causal_effects.csv").unlink(missing_ok=True)
        manifest = sorted(p.name for p in self.settings.outputs.iterdir())
        self.settings.output("manifest.json").write_text(json.dumps(manifest, indent=2))
