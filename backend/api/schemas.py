"""
Pydantic schema definitions for Drishti FastAPI REST API.
"""

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str = "1.0.0"


class DistrictProfileSummary(BaseModel):
    district: str
    total_population: int
    male_population: int
    female_population: int
    literacy_rate: float
    urban_pct: float
    work_participation_rate: float


class DistrictDetailResponse(BaseModel):
    district: str
    demographics: dict[str, Any]
    crime_history: list[dict[str, Any]]
    feature_metrics: dict[str, Any]
    predictions: dict[str, Any]


class CrimeTrendsResponse(BaseModel):
    state: str
    years_count: int
    yearly_trends: list[dict[str, Any]]


class HotspotResponse(BaseModel):
    district: str
    risk_score: float
    hotspot_score: float
    crime_forecast: float
    current_crime_rate_per_100k: float
    anomaly_score: float
    cluster_id: int


class PredictionRequest(BaseModel):
    district: str = Field(default="Sample District")
    total_population: float = Field(default=1000000.0, description="Total district population")
    crime_rate_per_100k: float = Field(default=150.0, description="Current crime rate per 100k")
    literacy_pct: float = Field(default=75.0, description="Literacy rate percentage")
    urban_pct: float = Field(default=35.0, description="Urbanization percentage")
    crime_growth_rate: float = Field(default=2.5, description="Annual crime growth rate percentage")
    crime_severity_score: float = Field(default=45.0, description="Crime severity score")


class PredictionResponse(BaseModel):
    district: str
    risk_score: float
    hotspot_score: float
    crime_forecast: float
    anomaly_score: float
    risk_level: str


class ShapContribution(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float
    direction: str


class ShapDistrictResponse(BaseModel):
    district: str
    base_value: float
    model_output: float
    top_increase: list[str]
    top_decrease: list[str]
    plain_english: list[str]
    contributions: list[ShapContribution]


class ShapGlobalResponse(BaseModel):
    model: str
    target: str
    base_value: float
    importance: list[ShapContribution]


class PatrolAssignment(BaseModel):
    district: str
    units_assigned: int
    risk_score: float
    coverage_fraction: float
    risk_reduced: float


class PatrolPlanResponse(BaseModel):
    total_units: int
    max_radius_km: float
    districts_considered: int
    risk_reduced: float
    risk_reduction_pct: float
    residual_risk: float
    covered_pct: float
    uncovered_count: int
    solver_status: str
    assignments: list[PatrolAssignment]


class NetworkTopConnected(BaseModel):
    suspect_id: str
    links: int


class NetworkGraphResponse(BaseModel):
    html_path: str
    node_count: int
    edge_count: int
    district_filter: str | None
    gangs: list[str]
    top_connected: list[NetworkTopConnected]


class HotspotMapResponse(BaseModel):
    html_path: str
    plotted_count: int
    missing_coords: list[str]
    bands: dict[str, int]
    deployments_overlaid: int
