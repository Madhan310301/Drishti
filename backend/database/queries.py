"""
Database query helpers for Drishti FastAPI endpoints and data handlers.
"""

from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.database.models import (
    DistrictProfile,
    CrimeStatistic,
    FeatureStore,
    PredictionResult,
    Alert,
)


def get_all_districts(db: Session) -> list[dict[str, Any]]:
    """
    Fetch all district profiles.

    Parameters
    ----------
    db : Session
        Database session.

    Returns
    -------
    list[dict[str, Any]]
        List of district profile dictionaries.
    """
    profiles = db.query(DistrictProfile).all()
    return [
        {
            "district": p.district,
            "total_population": p.total_population,
            "male_population": p.male_population,
            "female_population": p.female_population,
            "literacy_rate": p.literacy_rate,
            "urban_pct": p.urban_pct,
            "work_participation_rate": p.work_participation_rate,
        }
        for p in profiles
    ]


def get_district_by_name(db: Session, district_name: str) -> dict[str, Any] | None:
    """
    Fetch comprehensive details for a specific district.

    Parameters
    ----------
    db : Session
        Database session.
    district_name : str
        Target district name.

    Returns
    -------
    dict[str, Any] | None
        District details dict or None if not found.
    """
    profile = (
        db.query(DistrictProfile)
        .filter(func.lower(DistrictProfile.district) == district_name.strip().lower())
        .first()
    )
    if not profile:
        return None

    crime_stats = (
        db.query(CrimeStatistic)
        .filter(func.lower(CrimeStatistic.district) == district_name.strip().lower())
        .order_by(CrimeStatistic.year.asc())
        .all()
    )

    feature = (
        db.query(FeatureStore)
        .filter(func.lower(FeatureStore.district) == district_name.strip().lower())
        .first()
    )

    prediction = (
        db.query(PredictionResult)
        .filter(func.lower(PredictionResult.district) == district_name.strip().lower())
        .order_by(PredictionResult.id.desc())
        .first()
    )

    return {
        "district": profile.district,
        "demographics": {
            "total_population": profile.total_population,
            "male_population": profile.male_population,
            "female_population": profile.female_population,
            "literacy_rate": profile.literacy_rate,
            "sex_ratio": profile.sex_ratio,
            "urban_pct": profile.urban_pct,
            "work_participation_rate": profile.work_participation_rate,
        },
        "crime_history": [
            {
                "year": c.year,
                "total_crimes": c.total_crimes,
                "murder": c.murder,
                "rape": c.rape,
                "robbery": c.robbery,
                "crime_rate_per_100k": c.crime_rate_per_100k,
            }
            for c in crime_stats
        ],
        "feature_metrics": {
            "population_density": feature.population_density if feature else 0.0,
            "crime_severity_score": feature.crime_severity_score if feature else 0.0,
            "normalized_crime_index": feature.normalized_crime_index if feature else 0.0,
            "hotspot_score": feature.hotspot_score if feature else 0.0,
        }
        if feature
        else {},
        "predictions": {
            "risk_score": prediction.risk_score if prediction else 0.0,
            "hotspot_score": prediction.hotspot_score if prediction else 0.0,
            "crime_forecast": prediction.crime_forecast if prediction else 0.0,
            "anomaly_score": prediction.anomaly_score if prediction else 0.0,
        }
        if prediction
        else {},
    }


def get_crime_trends(db: Session) -> dict[str, Any]:
    """
    Fetch annual crime totals and category breakdown across Karnataka.

    Parameters
    ----------
    db : Session
        Database session.

    Returns
    -------
    dict[str, Any]
        Yearly trends dictionary.
    """
    yearly_totals = (
        db.query(
            CrimeStatistic.year,
            func.sum(CrimeStatistic.total_crimes).label("total_crimes"),
            func.sum(CrimeStatistic.murder).label("murder"),
            func.sum(CrimeStatistic.rape).label("rape"),
            func.sum(CrimeStatistic.robbery).label("robbery"),
            func.sum(CrimeStatistic.kidnapping).label("kidnapping"),
        )
        .group_by(CrimeStatistic.year)
        .order_by(CrimeStatistic.year.asc())
        .all()
    )

    trends = [
        {
            "year": int(row.year),
            "total_crimes": float(row.total_crimes or 0.0),
            "murder": float(row.murder or 0.0),
            "rape": float(row.rape or 0.0),
            "robbery": float(row.robbery or 0.0),
            "kidnapping": float(row.kidnapping or 0.0),
        }
        for row in yearly_totals
    ]

    return {
        "state": "Karnataka",
        "years_count": len(trends),
        "yearly_trends": trends,
    }


def get_hotspots(db: Session, limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetch top hotspot districts ordered by risk score and hotspot score.

    Parameters
    ----------
    db : Session
        Database session.
    limit : int
        Maximum number of hotspots to return.

    Returns
    -------
    list[dict[str, Any]]
        Hotspot districts list.
    """
    preds = (
        db.query(PredictionResult)
        .order_by(desc(PredictionResult.risk_score), desc(PredictionResult.hotspot_score))
        .limit(limit)
        .all()
    )

    return [
        {
            "district": p.district,
            "risk_score": p.risk_score,
            "hotspot_score": p.hotspot_score,
            "crime_forecast": p.crime_forecast,
            "current_crime_rate_per_100k": p.current_crime_rate_per_100k,
            "anomaly_score": p.anomaly_score,
            "cluster_id": p.cluster_id,
        }
        for p in preds
    ]
