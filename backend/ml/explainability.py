"""
SHAP explainability module for Drishti Predictive Command Console.

Fulfils the manual's "Explainable AI (SHAP)" deliverable (Vijay's role).

Design decision (per PROJECT_MEMORY dev principles - explain decisions):
- We explain the RandomForest crime-rate model (target = crime_rate_per_100k),
  because that is the genuine ML signal behind the platform's risk score.
- The risk_score itself is a post-hoc heuristic blend, so SHAP cannot explain it
  directly. Explaining the ML-predicted crime rate gives an intuitive,
  plain-English breakdown of *why* a district looks risky.
- We train SHAP's own RandomForest on the RAW (unscaled) features so the
  contributions are expressed in real units (e.g. "+Crime Rate / 100k"), which is
  far more readable than StandardScaler-scaled values. This model mirrors the
  hyperparameters of the pipeline's RandomForest (n_estimators=100, RANDOM_STATE)
  for consistency, while prioritising interpretability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ModelTrainingError
from backend.etl.config import (
    FEATURE_STORE_FILE,
    SHAP_EXPLANATIONS_FILE,
    RANDOM_STATE,
)

logger = get_logger(__name__)

# Friendly labels for raw feature keys (used in plain-English output).
FEATURE_LABELS = {
    "population_density": "Population Density",
    "crime_rate_per_100k": "Crime Rate / 100k",
    "literacy_pct": "Literacy %",
    "female_ratio": "Female Ratio",
    "urban_pct": "Urban %",
    "crime_growth_rate": "Crime Growth Rate",
    "economic_activity_index": "Economic Activity Index",
    "crime_severity_score": "Crime Severity Score",
    "normalized_crime_index": "Normalized Crime Index",
}

# Must stay in sync with backend/ml/pipeline.py MLPipeline.feature_cols.
EXPLAIN_FEATURES = [
    "population_density",
    "crime_rate_per_100k",
    "literacy_pct",
    "female_ratio",
    "urban_pct",
    "crime_growth_rate",
    "economic_activity_index",
    "crime_severity_score",
    "normalized_crime_index",
]

TARGET_COL = "crime_rate_per_100k"


class ShapExplainer:
    """
    Wraps a RandomForest crime-rate model with a SHAP TreeExplainer to produce
    per-district and global explanations.

    The model is fitted lazily on the feature store (28 rows -> instant). This
    keeps the explainer self-contained and lets the API serve explanations at
    runtime without a separately persisted model file.
    """

    def __init__(
        self,
        feature_store_path: Path = FEATURE_STORE_FILE,
        target_col: str = TARGET_COL,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.feature_store_path = feature_store_path
        self.target_col = target_col
        self.feature_cols = feature_cols or list(EXPLAIN_FEATURES)
        self.model = None
        self.explainer = None
        self.base_value = 0.0
        self._df: pd.DataFrame | None = None
        self._fit()

    def _fit(self) -> None:
        if not self.feature_store_path.exists():
            raise ModelTrainingError(
                f"Feature store not found: {self.feature_store_path}"
            )
        df = load_csv(self.feature_store_path)
        if df.empty:
            raise ModelTrainingError("Feature store is empty.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ModelTrainingError(f"Missing features in store: {missing}")
        if self.target_col not in df.columns:
            raise ModelTrainingError(f"Target column missing: {self.target_col}")

        X = df[self.feature_cols].astype(float)
        y = df[self.target_col].astype(float)

        self.model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        self.model.fit(X, y)

        self.explainer = shap.TreeExplainer(self.model)
        ev = np.asarray(self.explainer.expected_value).flatten()
        self.base_value = float(ev[0])

        self._df = df
        logger.info("SHAP explainer fitted. base_value=%.3f", self.base_value)

    def explain_district(self, district: str) -> dict[str, Any] | None:
        """
        Return a plain-English SHAP explanation for one district.

        Returns None if the district is not present in the feature store.
        """
        if self._df is None:
            return None
        row = self._df[self._df["district"] == district]
        if row.empty:
            return None
        row = row.iloc[0]

        feat_names = list(self.feature_cols)
        feat_vals = row[feat_names].astype(float).values.reshape(1, -1)
        X_row = pd.DataFrame(feat_vals, columns=feat_names)

        sv = np.asarray(self.explainer.shap_values(X_row)).reshape(-1)
        model_output = float(self.model.predict(X_row)[0])

        contributions: list[dict[str, Any]] = []
        for name, val, sh in zip(feat_names, feat_vals.reshape(-1), sv):
            contributions.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": round(float(val), 4),
                    "shap_value": round(float(sh), 4),
                    "direction": (
                        "increases_risk"
                        if sh > 0
                        else ("reduces_risk" if sh < 0 else "neutral")
                    ),
                }
            )

        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        top_increase = [c for c in contributions if c["shap_value"] > 0][:3]
        top_decrease = [c for c in contributions if c["shap_value"] < 0][:3]

        plain_english = []
        for c in top_increase + top_decrease:
            sign = "+" if c["shap_value"] >= 0 else "-"
            plain_english.append(f"{c['label']} ({sign}{abs(c['shap_value']):.2f})")

        return {
            "district": district,
            "base_value": round(self.base_value, 4),
            "model_output": round(model_output, 4),
            "contributions": contributions,
            "top_increase": [c["label"] for c in top_increase],
            "top_decrease": [c["label"] for c in top_decrease],
            "plain_english": plain_english,
        }

    def global_importance(self) -> list[dict[str, Any]]:
        """Mean absolute SHAP value per feature across all districts."""
        if self._df is None:
            return []
        X = self._df[self.feature_cols].astype(float)
        sv = np.asarray(self.explainer.shap_values(X))
        if sv.ndim == 3:
            sv = sv[0]
        mean_abs = np.abs(sv).mean(axis=0)

        out = []
        for name, m in zip(self.feature_cols, mean_abs):
            out.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "mean_abs_shap": round(float(m), 4),
                }
            )
        out.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
        return out

    def generate_all(self) -> dict[str, Any]:
        explanations: dict[str, Any] = {}
        if self._df is not None:
            for d in self._df["district"].tolist():
                exp = self.explain_district(d)
                if exp:
                    explanations[d] = exp
        return {
            "target": self.target_col,
            "model": "RandomForestRegressor",
            "base_value": round(self.base_value, 4),
            "global_importance": self.global_importance(),
            "explanations": explanations,
        }

    def save(self, out_path: Path = SHAP_EXPLANATIONS_FILE) -> Path:
        data = self.generate_all()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info("Saved SHAP explanations to %s", out_path.name)
        return out_path


def main() -> None:
    explainer = ShapExplainer()
    path = explainer.save()
    print(f"SHAP explanations saved to {path}")


if __name__ == "__main__":
    main()
