"""
FastAPI route handlers for Drishti REST API.
"""

from typing import Any
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.common.exceptions import ProcessingError
from backend.database.connection import get_db
from backend.database.queries import (
    get_all_districts,
    get_district_by_name,
    get_crime_trends,
    get_hotspots,
)
from backend.ml.pipeline import MLPipeline
from backend.ml.explainability import ShapExplainer
from backend.ml.patrol_optimizer import PatrolOptimizer
from backend.ml.network_graph import NetworkGraphBuilder
from backend.ml.hotspot_map import HotspotMapBuilder
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
    PatrolAssignment,
    PatrolPlanResponse,
    NetworkGraphResponse,
    NetworkTopConnected,
    HotspotMapResponse,
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


@router.get("/explain/districts", summary="Districts available for SHAP explanation")
def explain_districts() -> list[str]:
    """Return the district names the SHAP explainer can explain."""
    try:
        explainer = ShapExplainer()
        if explainer._df is not None:
            return sorted(explainer._df["district"].tolist())
    except Exception:
        pass
    return []


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


@router.post("/patrol/optimize", response_model=PatrolPlanResponse, summary="Patrol Deployment Plan")
def patrol_optimize(
    total_units: int = Query(default=20, ge=1, le=500, description="Available patrol units tonight"),
    max_radius_km: float = Query(default=3.0, gt=0, le=50, description="Patrol coverage radius (km)"),
) -> PatrolPlanResponse:
    """
    Allocate limited patrol units across Bengaluru hotspots to maximise risk
    coverage (PuLP linear program). The manual's Patrol Recommendation Engine.
    """
    try:
        optimizer = PatrolOptimizer()
        plan = optimizer.optimize(total_units=total_units, max_radius_km=max_radius_km)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Patrol optimization failed: {exc}")

    assignments = [PatrolAssignment(**a) for a in plan["assignments"]]
    return PatrolPlanResponse(
        total_units=plan["total_units"],
        max_radius_km=plan["max_radius_km"],
        districts_considered=plan["districts_considered"],
        risk_reduced=plan["risk_reduced"],
        risk_reduction_pct=plan["risk_reduction_pct"],
        residual_risk=plan["residual_risk"],
        covered_pct=plan["covered_pct"],
        uncovered_count=plan["uncovered_count"],
        solver_status=plan["solver_status"],
        assignments=assignments,
    )


@router.get("/hotspots/centers", summary="Hotspot Cluster Centers (DBSCAN)")
def hotspot_centers() -> list[dict]:
    """
    Return DBSCAN hotspot cluster centers with coordinates, for the map and the
    patrol optimizer. The manual's Task 1 output (hotspot_centers.csv).
    """
    try:
        from backend.etl.config import HOTSPOT_CENTERS_FILE
        from backend.common.helpers import load_csv
        if not HOTSPOT_CENTERS_FILE.exists():
            raise ProcessingError("Hotspot centers not generated. Run the ML pipeline.")
        df = load_csv(HOTSPOT_CENTERS_FILE)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hotspot centers failed: {exc}")


@router.get("/network/graph", response_model=NetworkGraphResponse, summary="Criminal Network Graph")
def network_graph(district: str | None = Query(default=None, description="Optional district filter")) -> NetworkGraphResponse:
    """
    Build an interactive PyVis HTML graph of the offender co-offending network.
    The manual's Criminal Network Analysis (Jenifa's role).
    """
    try:
        builder = NetworkGraphBuilder()
        meta = builder.build(district=district)
        builder.save_metadata(meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Network graph failed: {exc}")

    return NetworkGraphResponse(
        html_path=meta["html_path"],
        node_count=meta["node_count"],
        edge_count=meta["edge_count"],
        district_filter=meta["district_filter"],
        gangs=meta["gangs"],
        top_connected=[NetworkTopConnected(**t) for t in meta["top_connected"]],
    )


@router.get("/map/hotspots", response_model=HotspotMapResponse, summary="Hotspot Map")
def map_hotspots(
    overlay_patrols: bool = Query(default=False, description="Overlay latest patrol plan if available"),
) -> HotspotMapResponse:
    """
    Build an interactive Folium map of district crime hotspots (OpenStreetMap).
    The manual's Hotspot Map (Jenifa's role). Optionally overlays patrol units
    from the most recent patrol plan file.
    """
    try:
        deployments = None
        if overlay_patrols:
            from backend.etl.config import PATROL_PLAN_FILE
            if PATROL_PLAN_FILE.exists():
                import json
                with open(PATROL_PLAN_FILE) as f:
                    plan = json.load(f)
                deployments = plan.get("assignments")
        builder = HotspotMapBuilder()
        meta = builder.build(deployments=deployments)
        builder.save_metadata(meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hotspot map failed: {exc}")

    return HotspotMapResponse(
        html_path=meta["html_path"],
        plotted_count=meta["plotted_count"],
        missing_coords=meta["missing_coords"],
        bands=meta["bands"],
        deployments_overlaid=meta["deployments_overlaid"],
    )