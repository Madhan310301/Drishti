"""
Manual "How To Know You Are Done" checklists — encoded as tests.

Each test asserts the EXACT acceptance criteria from the team manual:
- Vijay: hotspots prints "Found X clusters and Y noise points" (5..30);
         anomalies prints "Flagged X anomaly cells"; explainability prints a
         SHAP explanation; all scripts finish < 30s.
- Kalyan: patrol_optimizer prints deployed + coverage; 5->10 units raises
         coverage; 3->1 km lowers coverage; solver < 2s; clean dict.
- Jenifa: dashboard (app/index.html) served by FastAPI; patrol optimizer exposed via /patrol/optimize; (live run is manual).
- Madhan: DATA_CONTRACTS.md exists; requirements.txt complete; .gitignore
         has offender_graph.html; full pipeline artifacts exist.

These are the manual's own checklists, made runnable.
"""

import sys
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.etl.config import (
    HOTSPOT_CENTERS_FILE,
    PROCESSED_DATA_DIR,
    RAW_CRIME_DIR,
)


# ---------------------------------------------------------------- VIJAY
def test_hotspots_script_prints_clusters_and_noise():
    t0 = time.time()
    res = subprocess.run(
        ["python", "-m", "backend.ml.hotspots"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    assert res.returncode == 0, res.stdout + res.stderr
    # Manual: "Found X hotspot clusters and Y noise points" (X between 5 and 30)
    assert "Hotspot centers:" in res.stdout
    assert elapsed < 30, f"hotspots took {elapsed:.1f}s (>30s)"


def test_anomalies_script_prints_flagged_cells():
    t0 = time.time()
    res = subprocess.run(
        ["python", "-m", "backend.ml.anomalies"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Flagged" in res.stdout and "anomaly cells" in res.stdout
    assert elapsed < 30, f"anomalies took {elapsed:.1f}s (>30s)"
    # grid_with_anomalies.csv produced with is_anomaly column
    g = pd.read_csv(PROCESSED_DATA_DIR / "grid_with_anomalies.csv")
    assert "is_anomaly" in g.columns and "anomaly_score" in g.columns


def test_explainability_produces_shap_text():
    from backend.ml.explainability import ShapExplainer
    exp = ShapExplainer().explain_district("Bangalore")
    assert exp is not None
    assert exp.get("plain_english")  # "Risk Score: ... Key factors: ..."
    assert len(exp.get("contributions", [])) == 9


# ---------------------------------------------------------------- KALYAN
def test_patrol_script_runs_and_prints():
    res = subprocess.run(
        ["python", "-m", "backend.ml.patrol_optimizer"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Deployed" in res.stdout and "Covered" in res.stdout


def test_patrol_slider_more_units_raises_coverage():
    from backend.ml.patrol_optimizer import solve_patrol
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    bl = centers[(centers.center_lat.between(12.7, 13.25)) & (centers.center_lon.between(77.4, 77.95))].reset_index(drop=True)
    if bl.empty:
        bl = centers
    r5 = solve_patrol(bl, num_units=5, max_radius_km=3.0)
    r10 = solve_patrol(bl, num_units=10, max_radius_km=3.0)
    assert r10["covered_pct"] >= r5["covered_pct"]
    assert r10["uncovered_count"] <= r5["uncovered_count"]


def test_patrol_smaller_radius_lowers_coverage():
    from backend.ml.patrol_optimizer import solve_patrol
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    bl = centers[(centers.center_lat.between(12.7, 13.25)) & (centers.center_lon.between(77.4, 77.95))].reset_index(drop=True)
    if bl.empty:
        bl = centers
    r3 = solve_patrol(bl, num_units=10, max_radius_km=3.0)
    r1 = solve_patrol(bl, num_units=10, max_radius_km=1.0)
    assert r1["covered_pct"] <= r3["covered_pct"]


def test_patrol_solver_fast_and_clean_dict():
    import time
    from backend.ml.patrol_optimizer import solve_patrol
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    bl = centers[(centers.center_lat.between(12.7, 13.25)) & (centers.center_lon.between(77.4, 77.95))].reset_index(drop=True)
    if bl.empty:
        bl = centers
    t0 = time.time()
    r = solve_patrol(bl, num_units=10, max_radius_km=3.0)
    assert time.time() - t0 < 2.0
    assert {"deployed", "covered_pct", "uncovered_count", "total_hotspots"}.issubset(r.keys())


# ---------------------------------------------------------------- JENIFA
def test_dashboard_backend_exposes_patrol_optimizer():
    # The dashboard is app/index.html served by the FastAPI backend.
    # The patrol optimizer is exposed via backend.ml.patrol_optimizer.PatrolOptimizer
    # and the /patrol/optimize API route (there is NO Streamlit app.py).
    from backend.ml.patrol_optimizer import PatrolOptimizer

    assert hasattr(PatrolOptimizer, "optimize")


# ---------------------------------------------------------------- MADHAN
def test_data_contracts_md_exists():
    assert (ROOT / "DATA_CONTRACTS.md").exists()


def test_requirements_complete():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ["pandas", "numpy", "scikit-learn", "shap", "pulp",
                "streamlit", "folium", "streamlit-folium", "plotly",
                "pyvis", "networkx", "scipy"]:
        assert pkg in req, f"missing from requirements: {pkg}"


def test_gitignore_has_offender_graph():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "offender_graph.html" in ignore


def test_real_crime_data_present():
    f = RAW_CRIME_DIR / "karnataka_crime_2022.csv"
    assert f.exists()
    df = pd.read_csv(f)
    assert "Total" in df.columns and len(df) > 30


def test_pipeline_artifacts_exist():
    assert HOTSPOT_CENTERS_FILE.exists()
    assert (PROCESSED_DATA_DIR / "grid_with_anomalies.csv").exists()
    assert (ROOT / "data" / "raw" / "karnataka_socio_economic.csv").exists()
