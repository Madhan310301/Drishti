"""
Test suite for Drishti Predictive Command Console.

Covers the manual-required features with real, runnable checks:
- hotspots (DBSCAN) produces coordinate-bearing centers
- patrol optimizer solve_patrol (manual signature) behaves correctly
- SHAP explainability returns plain-English + contributions
- network graph builds (nodes/edges, no missing coords)
- hotspot map builds (all districts plotted)
- FastAPI endpoints respond 200
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml import hotspots as hs
from backend.ml.patrol_optimizer import solve_patrol, PatrolOptimizer, haversine
from backend.ml.explainability import ShapExplainer
from backend.ml.network_graph import NetworkGraphBuilder
from backend.ml.hotspot_map import HotspotMapBuilder
from backend.etl.config import HOTSPOT_CENTERS_FILE


# ---------------------------------------------------------------- hotspots
def test_hotspot_centers_have_coordinates():
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    assert {"cluster_id", "district", "center_lat", "center_lon", "risk_score"}.issubset(centers.columns)
    assert len(centers) > 0
    assert centers["center_lat"].between(12.0, 18.0).all()
    assert centers["center_lon"].between(74.0, 78.5).all()


# ---------------------------------------------------------------- patrol (manual)
def test_haversine_sanity():
    d = haversine(12.9716, 77.5946, 13.0358, 77.5970)
    assert 6 < d < 9


def test_solve_patrol_manual_behavior():
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    bl = centers[
        (centers.center_lat.between(12.7, 13.25)) & (centers.center_lon.between(77.4, 77.95))
    ].reset_index(drop=True)
    r5 = solve_patrol(bl, num_units=5, max_radius_km=3.0)
    r10 = solve_patrol(bl, num_units=10, max_radius_km=3.0)
    assert r5["covered_pct"] <= r10["covered_pct"] + 1e-9
    assert r5["uncovered_count"] >= r10["uncovered_count"]
    assert len(r5["deployed"]) <= 5 and len(r10["deployed"]) <= 10
    assert r5["total_hotspots"] == len(bl)


def test_solve_patrol_empty():
    empty = pd.DataFrame(columns=["center_lat", "center_lon", "risk_score"])
    r = solve_patrol(empty, num_units=5)
    assert r["covered_pct"] == 100.0 and r["uncovered_count"] == 0


def test_patrol_optimizer_wrapper():
    plan = PatrolOptimizer().optimize(total_units=10)
    assert plan["total_units"] == 10
    assert plan["risk_reduction_pct"] >= 0


# ---------------------------------------------------------------- SHAP
def test_shap_explainer():
    exp = ShapExplainer()
    out = exp.explain_district("Bangalore")
    assert out is not None
    assert len(out["contributions"]) == 9
    assert out["plain_english"]
    assert exp.explain_district("Narnia") is None


# ---------------------------------------------------------------- network
def test_network_graph_builds():
    meta = NetworkGraphBuilder().build()
    assert meta["node_count"] == 45 and meta["edge_count"] == 70


# ---------------------------------------------------------------- map
def test_hotspot_map_builds():
    meta = HotspotMapBuilder().build()
    assert meta["plotted_count"] == 28
    assert meta["missing_coords"] == []


# ---------------------------------------------------------------- API
def test_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/explain/Bangalore").status_code == 200
    assert c.post("/patrol/optimize", params={"total_units": 10}).status_code == 200
    assert c.get("/network/graph").status_code == 200
    assert c.get("/map/hotspots").status_code == 200
    assert c.get("/hotspots/centers").status_code == 200
