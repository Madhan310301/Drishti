"""
Database seed module for populating SQLAlchemy database from processed ETL & ML output CSVs.
"""

from pathlib import Path
from typing import Any
import pandas as pd
from sqlalchemy.orm import Session

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.database.connection import init_db, SessionLocal
from backend.database.models import (
    DistrictProfile,
    CrimeStatistic,
    FeatureStore,
    PredictionResult,
    Alert,
)
from backend.etl.config import (
    DISTRICT_PROFILE_FILE,
    CRIME_STATISTICS_FILE,
    FEATURE_STORE_FILE,
    HOTSPOT_PREDICTIONS_FILE,
    ANOMALY_SCORES_FILE,
)

logger = get_logger(__name__)


def seed_district_profiles(db: Session) -> int:
    """Seed district_profiles table from CSV."""
    if not DISTRICT_PROFILE_FILE.exists():
        logger.warning(f"Seed skipped: {DISTRICT_PROFILE_FILE.name} not found.")
        return 0

    df = load_csv(DISTRICT_PROFILE_FILE)
    count = 0
    for _, row in df.iterrows():
        existing = db.query(DistrictProfile).filter_by(district=row["district"]).first()
        if not existing:
            profile = DistrictProfile(
                district=str(row["district"]),
                total_population=int(row["total_population"]),
                male_population=int(row["male_population"]),
                female_population=int(row["female_population"]),
                literate_population=int(row["literate_population"]),
                households=int(row["households"]),
                total_workers=int(row["total_workers"]),
                main_workers=int(row["main_workers"]),
                non_workers=int(row["non_workers"]),
                urban_population=int(row["urban_population"]),
                rural_population=int(row["rural_population"]),
                literacy_rate=float(row["literacy_rate"]),
                sex_ratio=float(row["sex_ratio"]),
                female_ratio=float(row["female_ratio"]),
                urban_pct=float(row["urban_pct"]),
                work_participation_rate=float(row["work_participation_rate"]),
            )
            db.add(profile)
            count += 1
    db.commit()
    logger.info(f"Seeded {count} district profiles.")
    return count


def seed_crime_statistics(db: Session) -> int:
    """Seed crime_statistics table from CSV."""
    if not CRIME_STATISTICS_FILE.exists():
        logger.warning(f"Seed skipped: {CRIME_STATISTICS_FILE.name} not found.")
        return 0

    df = load_csv(CRIME_STATISTICS_FILE)
    count = 0
    for _, row in df.iterrows():
        existing = (
            db.query(CrimeStatistic)
            .filter_by(district=row["district"], year=int(row["year"]))
            .first()
        )
        if not existing:
            stat = CrimeStatistic(
                district=str(row["district"]),
                year=int(row["year"]),
                murder=float(row.get("murder", 0.0)),
                attempt_to_murder=float(row.get("attempt_to_murder", 0.0)),
                rape=float(row.get("rape", 0.0)),
                kidnapping=float(row.get("kidnapping", 0.0)),
                dacoity=float(row.get("dacoity", 0.0)),
                robbery=float(row.get("robbery", 0.0)),
                burglary=float(row.get("burglary", 0.0)),
                theft=float(row.get("theft", 0.0)),
                riots=float(row.get("riots", 0.0)),
                cheating=float(row.get("cheating", 0.0)),
                dowry_deaths=float(row.get("dowry_deaths", 0.0)),
                total_crimes=float(row.get("total_crimes", 0.0)),
                crime_rate_per_100k=float(row.get("crime_rate_per_100k", 0.0)),
            )
            db.add(stat)
            count += 1
    db.commit()
    logger.info(f"Seeded {count} crime statistics records.")
    return count


def seed_feature_store(db: Session) -> int:
    """Seed feature_store table from CSV."""
    if not FEATURE_STORE_FILE.exists():
        logger.warning(f"Seed skipped: {FEATURE_STORE_FILE.name} not found.")
        return 0

    df = load_csv(FEATURE_STORE_FILE)
    count = 0
    for _, row in df.iterrows():
        existing = db.query(FeatureStore).filter_by(district=row["district"]).first()
        if not existing:
            feature = FeatureStore(
                district=str(row["district"]),
                total_population=int(row["total_population"]),
                population_density=float(row["population_density"]),
                crime_rate_per_100k=float(row["crime_rate_per_100k"]),
                literacy_pct=float(row["literacy_pct"]),
                female_ratio=float(row["female_ratio"]),
                urban_pct=float(row["urban_pct"]),
                population_growth=float(row["population_growth"]),
                crime_growth_rate=float(row["crime_growth_rate"]),
                economic_activity_index=float(row["economic_activity_index"]),
                crime_severity_score=float(row["crime_severity_score"]),
                normalized_crime_index=float(row["normalized_crime_index"]),
                hotspot_score=float(row["hotspot_score"]),
            )
            db.add(feature)
            count += 1
    db.commit()
    logger.info(f"Seeded {count} feature store records.")
    return count


def seed_predictions_and_alerts(db: Session) -> int:
    """Seed prediction_results and initial alerts from ML output CSVs."""
    if not HOTSPOT_PREDICTIONS_FILE.exists():
        logger.warning(f"Seed skipped: {HOTSPOT_PREDICTIONS_FILE.name} not found.")
        return 0

    df = load_csv(HOTSPOT_PREDICTIONS_FILE)
    count = 0
    alert_count = 0

    for _, row in df.iterrows():
        district = str(row["district"])
        risk = float(row["risk_score"])
        anomaly = float(row.get("anomaly_score", 0.0))

        pred = PredictionResult(
            district=district,
            cluster_id=int(row.get("cluster_id", 0)),
            hotspot_score=float(row["hotspot_score"]),
            risk_score=risk,
            crime_forecast=float(row["crime_forecast"]),
            current_crime_rate_per_100k=float(row["current_crime_rate_per_100k"]),
            anomaly_score=anomaly,
        )
        db.add(pred)
        count += 1

        # Seed automated alerts for high risk or high anomaly districts
        if risk > 65.0:
            alert = Alert(
                district=district,
                alert_type="CRIME_HOTSPOT",
                severity="HIGH" if risk < 80.0 else "CRITICAL",
                message=f"High crime risk detected in {district} (Risk Score: {risk}, Forecast: {row['crime_forecast']}).",
            )
            db.add(alert)
            alert_count += 1
        elif anomaly > 0.70:
            alert = Alert(
                district=district,
                alert_type="ANOMALY_DETECTED",
                severity="HIGH",
                message=f"Statistical crime anomaly detected in {district} (Anomaly Score: {anomaly}).",
            )
            db.add(alert)
            alert_count += 1

    db.commit()
    logger.info(f"Seeded {count} prediction results and {alert_count} alerts.")
    return count


def seed_database() -> None:
    """Master database seed runner."""
    logger.info("START: Database Seeding")
    init_db()

    db = SessionLocal()
    try:
        p_count = seed_district_profiles(db)
        c_count = seed_crime_statistics(db)
        f_count = seed_feature_store(db)
        pred_count = seed_predictions_and_alerts(db)

        logger.info(
            f"END: Database Seeding Complete | Profiles: {p_count} | Crime Records: {c_count} | "
            f"Features: {f_count} | Predictions: {pred_count}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
