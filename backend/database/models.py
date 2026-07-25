"""
SQLAlchemy ORM models for Drishti Predictive Command Console database layer.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, BigInteger, Float, String, DateTime, Index
from backend.database.connection import Base


class DistrictProfile(Base):
    """
    District demographic profiles table.
    """

    __tablename__ = "district_profiles"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String(100), unique=True, index=True, nullable=False)
    total_population = Column(BigInteger, default=0)
    male_population = Column(BigInteger, default=0)
    female_population = Column(BigInteger, default=0)
    literate_population = Column(BigInteger, default=0)
    households = Column(BigInteger, default=0)
    total_workers = Column(BigInteger, default=0)
    main_workers = Column(BigInteger, default=0)
    non_workers = Column(BigInteger, default=0)
    urban_population = Column(BigInteger, default=0)
    rural_population = Column(BigInteger, default=0)
    literacy_rate = Column(Float, default=0.0)
    sex_ratio = Column(Float, default=0.0)
    female_ratio = Column(Float, default=0.0)
    urban_pct = Column(Float, default=0.0)
    work_participation_rate = Column(Float, default=0.0)


class CrimeStatistic(Base):
    """
    Yearly crime statistics by district.
    """

    __tablename__ = "crime_statistics"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String(100), index=True, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    murder = Column(Float, default=0.0)
    attempt_to_murder = Column(Float, default=0.0)
    rape = Column(Float, default=0.0)
    kidnapping = Column(Float, default=0.0)
    dacoity = Column(Float, default=0.0)
    robbery = Column(Float, default=0.0)
    burglary = Column(Float, default=0.0)
    theft = Column(Float, default=0.0)
    riots = Column(Float, default=0.0)
    cheating = Column(Float, default=0.0)
    dowry_deaths = Column(Float, default=0.0)
    total_crimes = Column(Float, default=0.0)
    crime_rate_per_100k = Column(Float, default=0.0)

    __table_args__ = (
        Index("idx_district_year", "district", "year", unique=True),
    )


class FeatureStore(Base):
    """
    Engineered feature store metrics for predictive ML models.
    """

    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String(100), unique=True, index=True, nullable=False)
    total_population = Column(BigInteger, default=0)
    population_density = Column(Float, default=0.0)
    crime_rate_per_100k = Column(Float, default=0.0)
    literacy_pct = Column(Float, default=0.0)
    female_ratio = Column(Float, default=0.0)
    urban_pct = Column(Float, default=0.0)
    population_growth = Column(Float, default=0.0)
    crime_growth_rate = Column(Float, default=0.0)
    economic_activity_index = Column(Float, default=0.0)
    crime_severity_score = Column(Float, default=0.0)
    normalized_crime_index = Column(Float, default=0.0)
    hotspot_score = Column(Float, default=0.0)


class PredictionResult(Base):
    """
    Model output predictions (Risk score, Hotspot score, Crime forecast, Anomaly score).
    """

    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String(100), index=True, nullable=False)
    cluster_id = Column(Integer, default=0)
    hotspot_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    crime_forecast = Column(Float, default=0.0)
    current_crime_rate_per_100k = Column(Float, default=0.0)
    anomaly_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    """
    System generated crime alerts for command console.
    """

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String(100), index=True, nullable=False)
    alert_type = Column(String(50), nullable=False)  # e.g., CRIME_HOTSPOT, ANOMALY_DETECTED
    severity = Column(String(20), default="HIGH")    # HIGH, CRITICAL, MEDIUM
    message = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
