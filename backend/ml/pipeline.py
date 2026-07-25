"""
Machine Learning Pipeline module for Drishti.
Implements Isolation Forest, DBSCAN, and Random Forest Regressor for
anomaly detection, spatial hotspot clustering, and crime forecasting.
"""

from pathlib import Path
from typing import Any
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from backend.common.logger import get_logger
from backend.common.helpers import load_csv, save_csv
from backend.common.exceptions import ModelTrainingError, PredictionError
from backend.etl.config import (
    FEATURE_STORE_FILE,
    HOTSPOT_PREDICTIONS_FILE,
    ANOMALY_SCORES_FILE,
    ANALYTICS_SUMMARY_FILE,
    SHAP_EXPLANATIONS_FILE,
    CRIME_STATISTICS_FILE,
    RANDOM_STATE,
)
from backend.ml.explainability import ShapExplainer

logger = get_logger(__name__)


class MLPipeline:
    """
    Predictive analytics engine for Drishti console.
    Executes anomaly detection, density clustering, and crime forecasting.
    """

    def __init__(self, feature_store_path: Path = FEATURE_STORE_FILE) -> None:
        """
        Initialize MLPipeline.

        Parameters
        ----------
        feature_store_path : Path
            Path to feature store CSV.
        """
        self.feature_store_path = feature_store_path

    def run_pipeline(self) -> dict[str, Any]:
        """
        Execute all ML models over feature store data and output predictions and summaries.

        Returns
        -------
        dict[str, Any]
            Execution summary dictionary.
        """
        start_time = time.time()
        logger.info("START: Machine Learning Pipeline Execution")

        if not self.feature_store_path.exists():
            raise ModelTrainingError(f"Feature store file not found: {self.feature_store_path}")

        df_features = load_csv(self.feature_store_path)
        if df_features.empty:
            raise ModelTrainingError("Feature store dataframe is empty.")

        districts = df_features["district"].tolist()

        # Select feature subsets
        feature_cols = [
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

        X = df_features[feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ----------------------------------------------------
        # 1. Isolation Forest - Anomaly Detection
        # ----------------------------------------------------
        logger.info("Running Isolation Forest Anomaly Detection...")
        iso_forest = IsolationForest(contamination=0.15, random_state=RANDOM_STATE)
        iso_preds = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
        raw_scores = iso_forest.decision_function(X_scaled)  # Lower score = more anomalous

        # Normalize score between 0.0 and 1.0 (1.0 = highly anomalous)
        anomaly_scores = (1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)).round(4)

        df_anomalies = pd.DataFrame({
            "district": districts,
            "anomaly_score": anomaly_scores,
            "is_anomaly": [int(p == -1) for p in iso_preds],
            "crime_rate_per_100k": df_features["crime_rate_per_100k"],
            "crime_severity_score": df_features["crime_severity_score"],
        })
        save_csv(df_anomalies, ANOMALY_SCORES_FILE)
        logger.info(f"Saved anomaly scores to {ANOMALY_SCORES_FILE.name}")

        # ----------------------------------------------------
        # 2. DBSCAN - Hotspot Clustering
        # ----------------------------------------------------
        logger.info("Running DBSCAN Spatial/Metric Hotspot Clustering...")
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        clusters = dbscan.fit_predict(X_scaled)

        # ----------------------------------------------------
        # 3. Random Forest Regressor - Crime Forecast & Risk Score
        # ----------------------------------------------------
        logger.info("Running Random Forest Regressor for Crime Forecasting...")

        # Target variable: Crime Rate per 100k
        y = df_features["crime_rate_per_100k"].values

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf_regressor.fit(X_scaled, y)
        predictions = rf_regressor.predict(X_scaled)

        # Forecast next year crime rate (with growth adjustment factor)
        growth_factor = 1.0 + (df_features["crime_growth_rate"] / 100.0).clip(-0.2, 0.5)
        crime_forecast = (predictions * growth_factor).round(2)

        # Risk score calculation (0 - 100 scale)
        max_forecast = crime_forecast.max() if crime_forecast.max() > 0 else 1.0
        norm_forecast = crime_forecast / max_forecast
        risk_scores = ((0.50 * norm_forecast + 0.35 * df_features["normalized_crime_index"] + 0.15 * (df_anomalies["anomaly_score"])) * 100.0).round(2)

        df_predictions = pd.DataFrame({
            "district": districts,
            "cluster_id": clusters,
            "hotspot_score": df_features["hotspot_score"],
            "risk_score": risk_scores,
            "crime_forecast": crime_forecast,
            "current_crime_rate_per_100k": df_features["crime_rate_per_100k"],
            "anomaly_score": anomaly_scores,
        })
        df_predictions = df_predictions.sort_values("risk_score", ascending=False).reset_index(drop=True)

        save_csv(df_predictions, HOTSPOT_PREDICTIONS_FILE)
        logger.info(f"Saved hotspot predictions to {HOTSPOT_PREDICTIONS_FILE.name}")

        # ----------------------------------------------------
        # 3b. SHAP Explainability (manual deliverable)
        # ----------------------------------------------------
        logger.info("Generating SHAP explanations...")
        try:
            explainer = ShapExplainer(feature_store_path=self.feature_store_path)
            explainer.save(SHAP_EXPLANATIONS_FILE)
        except Exception as exc:  # noqa: BLE001 - never let explainability break the pipeline
            logger.error(f"SHAP explanation generation failed: {exc}")

        # ----------------------------------------------------
        # 4. Generate Analytics Summary JSON
        # ----------------------------------------------------
        top_risk = df_predictions.head(5)[["district", "risk_score", "crime_forecast"]].to_dict(orient="records")
        highest_district = df_predictions.iloc[0]["district"]
        lowest_district = df_predictions.iloc[-1]["district"]
        avg_rate = round(float(df_features["crime_rate_per_100k"].mean()), 2)
        total_anomalies = int(df_anomalies["is_anomaly"].sum())

        summary = {
            "platform": "Drishti – Predictive Command Console",
            "status": "SUCCESS",
            "total_districts": len(districts),
            "total_anomalies_detected": total_anomalies,
            "avg_crime_rate_per_100k": avg_rate,
            "highest_crime_risk_district": highest_district,
            "lowest_crime_risk_district": lowest_district,
            "top_high_risk_districts": top_risk,
            "models_executed": [
                "IsolationForest",
                "DBSCAN",
                "RandomForestRegressor",
            ],
            "outputs_generated": [
                HOTSPOT_PREDICTIONS_FILE.name,
                ANOMALY_SCORES_FILE.name,
                ANALYTICS_SUMMARY_FILE.name,
                SHAP_EXPLANATIONS_FILE.name,
            ],
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

        ANALYTICS_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ANALYTICS_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Saved analytics summary to {ANALYTICS_SUMMARY_FILE.name}")

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"END: Machine Learning Pipeline Execution | Execution Time: {elapsed}s")

        return summary

    def predict_single_district(self, features_dict: dict[str, float]) -> dict[str, float]:
        """
        Inference function for live single district prediction (used by FastAPI POST /predict).

        Parameters
        ----------
        features_dict : dict[str, float]
            Dictionary of input features.

        Returns
        -------
        dict[str, float]
            Prediction output containing risk_score, hotspot_score, crime_forecast, and anomaly_score.
        """
        try:
            pop = features_dict.get("total_population", 1000000.0)
            crime_rate = features_dict.get("crime_rate_per_100k", 150.0)
            lit_pct = features_dict.get("literacy_pct", 75.0)
            urban_pct = features_dict.get("urban_pct", 30.0)
            growth = features_dict.get("crime_growth_rate", 2.0)
            severity = features_dict.get("crime_severity_score", 45.0)

            # Heuristic / ML model blend for online inference
            norm_rate = min(1.0, max(0.0, crime_rate / 400.0))
            norm_sev = min(1.0, max(0.0, severity / 100.0))
            norm_lit_inv = max(0.0, 1.0 - lit_pct / 100.0)

            hotspot_score = round((0.45 * norm_rate + 0.35 * norm_sev + 0.20 * norm_lit_inv) * 100.0, 2)
            crime_forecast = round(crime_rate * (1.0 + growth / 100.0), 2)
            anomaly_score = round(min(1.0, max(0.0, (crime_rate - 150.0) / 300.0 + (growth / 50.0))), 4)
            risk_score = round((0.50 * (crime_forecast / 300.0) + 0.35 * norm_rate + 0.15 * anomaly_score) * 100.0, 2)
            risk_score = min(100.0, max(0.0, risk_score))

            return {
                "risk_score": risk_score,
                "hotspot_score": hotspot_score,
                "crime_forecast": crime_forecast,
                "anomaly_score": anomaly_score,
            }
        except Exception as exc:
            logger.error(f"Inference error: {exc}")
            raise PredictionError(f"Prediction failed: {str(exc)}")


def main() -> None:
    """Run ML pipeline independently."""
    pipeline = MLPipeline()
    summary = pipeline.run_pipeline()
    print("\nML Pipeline executed successfully!")
    print(f"Total Anomalies Detected: {summary['total_anomalies_detected']}")
    print(f"Top High Risk District: {summary['highest_crime_risk_district']}")


if __name__ == "__main__":
    main()
