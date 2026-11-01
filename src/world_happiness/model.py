"""XGBoost model of the happiness score, baselines and SHAP explanations (TreeSHAP)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from .config import FACTORS, TARGET

Split = tuple[np.ndarray, np.ndarray]


class AdditiveBaseline:
    """score ~ sum(factors) + mean residual: the WHR identity itself, no learning."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> AdditiveBaseline:
        self.offset_ = float((y - x.sum(axis=1)).mean())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (x.sum(axis=1) + self.offset_).to_numpy()


class HappinessModel:
    """Fits XGBoost on the six WHR factors and compares it with two baselines.

    Validation uses two schemes: *temporal* (train on earlier years, test on the last)
    and *grouped* (5-fold by country, so no country appears in train and test).
    """

    def __init__(self, df: pd.DataFrame, features: Sequence[str] = FACTORS, seed: int = 42):
        self.df = df.dropna(subset=[*features, TARGET]).reset_index(drop=True)
        self.features = list(features)
        self.seed = seed
        self.model: xgb.XGBRegressor | None = None

    @property
    def x(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def y(self) -> pd.Series:
        return self.df[TARGET]

    def candidates(self) -> dict[str, Callable[[], object]]:
        return {
            "xgboost": lambda: xgb.XGBRegressor(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.seed,
            ),
            "linear": LinearRegression,
            "additive_identity": AdditiveBaseline,
        }

    def splits(self) -> Iterator[tuple[str, Split]]:
        last = self.df["year"].max()
        yield (
            "temporal",
            (np.flatnonzero(self.df.year < last), np.flatnonzero(self.df.year == last)),
        )
        for tr, te in GroupKFold(n_splits=5).split(self.x, groups=self.df["iso3"]):
            yield "grouped", (tr, te)

    def evaluate(self) -> pd.DataFrame:
        rows = []
        for scheme, (tr, te) in self.splits():
            for name, make in self.candidates().items():
                est = make().fit(self.x.iloc[tr], self.y.iloc[tr])  # type: ignore[attr-defined]
                pred, true = est.predict(self.x.iloc[te]), self.y.iloc[te]
                rows.append(
                    {
                        "scheme": scheme,
                        "model": name,
                        "r2": r2_score(true, pred),
                        "rmse": root_mean_squared_error(true, pred),
                        "mae": mean_absolute_error(true, pred),
                    }
                )
        return pd.DataFrame(rows).groupby(["scheme", "model"], as_index=False).mean()

    def fit(self) -> HappinessModel:
        self.model = self.candidates()["xgboost"]().fit(self.x, self.y)  # type: ignore[attr-defined]
        return self

    def explain(self) -> pd.DataFrame:
        """SHAP values in long format: one row per country-year-feature."""
        model = self.model or self.fit().model
        assert model is not None
        # XGBoost's built-in exact TreeSHAP; last column is the bias term.
        contribs = model.get_booster().predict(xgb.DMatrix(self.x), pred_contribs=True)
        wide = pd.DataFrame(contribs[:, :-1], columns=self.features)
        ids = self.df[["iso3", "country", "region", "year"]]
        long = pd.concat([ids, wide], axis=1).melt(
            ids.columns, var_name="feature", value_name="shap"
        )
        feat = self.x.melt(var_name="feature", value_name="value")["value"]
        return long.assign(value=feat.to_numpy())

    def save(self, path: Path | str) -> Path:
        assert self.model is not None, "call fit() first"
        self.model.save_model(path)
        return Path(path)
