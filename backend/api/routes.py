"""
FastAPI route handlers for Drishti REST API.
"""

from typing import Any
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.database.connection import get_db
from backend.database.queries import (
    get_all_districts,
    get_district_by_name,
    get_crime_trends,
    get_hotspots,
)
from backend.ml.pipeline import MLPipeline
from backend.ml.explainability import ShapExplainer
from backend.etl.config import ANALYTICS_SUMMARY_FILE
from backend.api.schemas import (
    HealthResponse,
    DistrictProfileSummary,
    DistrictDetailResponse,
    CrimeTrendsResponse,
    HotspotResponse,
    PredictionRequest,
    PredictionResponse,
    ShapContribution,
    ShapDistrictResponse,
    ShapGlobalResponse,
)

router = APIRouter()
ml_pipeline = MLPipeline()


@router.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Check API service health and database connectivity.
    """
    try:
        db.execute(text("SELECT 1"))
        db_status = "Connected"
    except Exception:
        db_status = "Disconnected"

    return HealthResponse(status="OK", database=db_status)


@router.get("/districts", response_model=list[DistrictProfileSummary], summary="List All Districts")
def list_districts(db: Session = Depends(get_db)) -> list[DistrictProfileSummary]:
    """
    Retrieve list of all processed district profiles.
    """
    districts = get_all_districts(db)
    return [DistrictProfileSummary(**d) for d in districts]


@router.get("/district/{district}", response_model=DistrictDetailResponse, summary="Get District Details")
def get_district_details(district: str, db: Session = Depends(get_db)) -> DistrictDetailResponse:
    """
    Retrieve comprehensive details, demographics, crime history, and predictions for a specific district.
    """
    detail = get_district_by_name(db, district)
    if not detail:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found in database.")
    return DistrictDetailResponse(**detail)


@router.get("/crime/trends", response_model=CrimeTrendsResponse, summary="Statewide Crime Trends")
def crime_trends(db: Session = Depends(get_db)) -> CrimeTrendsResponse:
    """
    Retrieve yearly crime statistics and category trends across Karnataka.
    """
    trends = get_crime_trends(db)
    return CrimeTrendsResponse(**trends)


@router.get("/crime/hotspots", response_model=list[HotspotResponse], summary="Top Crime Hotspots")
def crime_hotspots(
    limit: int = Query(default=10, ge=1, le=50, description="Max number of hotspots"),
    db: Session = Depends(get_db),
) -> list[HotspotResponse]:
    """
    Retrieve top high-risk crime hotspots predicted by ML models.
    """
    hotspots = get_hotspots(db, limit=limit)
    return [HotspotResponse(**h) for h in hotspots]


@router.get("/analytics/summary", summary="Analytics Summary")
def analytics_summary() -> dict[str, Any]:
    """
    Retrieve high-level predictive analytics summary.
    """
    if ANALYTICS_SUMMARY_FILE.exists():
        try:
            with open(ANALYTICS_SUMMARY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error reading summary file: {exc}")
    
    return {
        "platform": "Drishti – Predictive Command Console",
        "status": "NO_SUMMARY_FILE",
        "message": "Run master ETL to generate analytics summary.",
    }


@router.post("/predict", response_model=PredictionResponse, summary="Live ML Prediction")
def predict_crime(request: PredictionRequest) -> PredictionResponse:
    """
    Live prediction endpoint returning risk score, hotspot score, crime forecast, and anomaly score.
    """
    feat_dict = request.model_dump()
    preds = ml_pipeline.predict_single_district(feat_dict)

    risk_score = preds["risk_score"]
    if risk_score > 75.0:
        risk_level = "CRITICAL"
    elif risk_score > 50.0:
        risk_level = "HIGH"
    elif risk_score > 25.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        district=request.district,
        risk_score=preds["risk_score"],
        hotspot_score=preds["hotspot_score"],
        crime_forecast=preds["crime_forecast"],
        anomaly_score=preds["anomaly_score"],
        risk_level=risk_level,
    )


@router.get("/explain/global", response_model=ShapGlobalResponse, summary="SHAP Global Importance")
def explain_global() -> ShapGlobalResponse:
    """
    Return mean-absolute SHAP feature importance across all districts.
    Registered before /explain/{district} so 'global' is not captured as a district name.
    """
    try:
        explainer = ShapExplainer()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP explainer failed: {exc}")

    importance = [ShapContribution(**{"feature": i["feature"], "label": i["label"],
                                      "value": 0.0, "shap_value": i["mean_abs_shap"],
                                      "direction": "neutral"}) for i in explainer.global_importance()]
    return ShapGlobalResponse(
        model="RandomForestRegressor",
        target="crime_rate_per_100k",
        base_value=explainer.base_value,
        importance=importance,
    )


@router.get("/explain/{district}", response_model=ShapDistrictResponse, summary="SHAP District Explanation")
def explain_district(district: str) -> ShapDistrictResponse:
    """
    Explain a district's predicted crime rate with SHAP feature contributions
    (the manual's Explainable AI deliverable).
    """
    try:
        explainer = ShapExplainer()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP explainer failed: {exc}")

    explanation = explainer.explain_district(district)
    if not explanation:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found in feature store.")

    contributions = [ShapContribution(**c) for c in explanation["contributions"]]
    return ShapDistrictResponse(
        district=explanation["district"],
        base_value=explanation["base_value"],
        model_output=explanation["model_output"],
        top_increase=explanation["top_increase"],
        top_decrease=explanation["top_decrease"],
        plain_english=explanation["plain_english"],
        contributions=contributions,
    )
