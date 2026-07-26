"""
Drishti Predictive Command Console - Streamlit Dashboard (Jenifa's territory).

Manual-required deliverable (line 44, 62): a Streamlit app that consumes the
team's ML outputs. It imports Kalyan's solve_patrol directly (manual line 454)
and presents the Folium hotspot map, PyVis network graph, and SHAP charts.

Run:  streamlit run app/app.py
(Assumes the FastAPI backend is NOT required - this app reads generated artifacts
 and imports ml modules directly, matching the manual's import pattern.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing backend modules (manual: sys.path.append('..'))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.patrol_optimizer import solve_patrol  # Kalyan's function
from backend.etl.config import (
    HOTSPOT_CENTERS_FILE,
    HOTSPOT_MAP_HTML,
    NETWORK_GRAPH_HTML,
    SHAP_EXPLANATIONS_FILE,
    RAW_CRIME_DIR,
)

st.set_page_config(page_title="Drishti | Predictive Command Console", layout="wide")

# ---------------------------------------------------------------- header
st.title("DRISHTI - Predictive Command Console")
st.caption("Karnataka State Police - Datathon 2026 | SHAP + PuLP + Folium + PyVis")

# ---------------------------------------------------------------- KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Districts Analyzed", 28)
with col2:
    try:
        crime = pd.read_csv(RAW_CRIME_DIR / "karnataka_crime_2022.csv")
        total_crimes = int(crime["Total"].sum())
        st.metric("Crimes (2022, real)", f"{total_crimes:,}")
    except Exception:
        st.metric("Crimes (2022, real)", "n/a")
with col3:
    st.metric("Highest Risk Zone", "Bengaluru Urban")
with col4:
    st.metric("ML Anomalies", 5)

# ---------------------------------------------------------------- patrol simulator
st.header("Patrol Deployment Optimizer (PuLP)")
st.write("Allocate limited Bengaluru patrol units to maximise risk coverage.")

try:
    centers = pd.read_csv(HOTSPOT_CENTERS_FILE)
    bengaluru = centers[
        (centers["center_lat"].between(12.7, 13.25)) & (centers["center_lon"].between(77.4, 77.95))
    ].reset_index(drop=True)
    if bengaluru.empty:
        bengaluru = centers
except Exception:
    st.error("Hotspot centers not found. Run: python -m backend.ml.hotspots")
    bengaluru = pd.DataFrame(columns=["center_lat", "center_lon", "risk_score"])

units = st.slider("Patrol units available", 1, 15, 5, key="patrol_units")
radius = st.slider("Coverage radius (km)", 1.0, 10.0, 3.0, 0.5, key="patrol_radius")

if not bengaluru.empty:
    result = solve_patrol(bengaluru, num_units=units, max_radius_km=radius)
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Covered", f"{result['covered_pct']}%")
    c2.metric("Hotspots Uncovered", result["uncovered_count"])
    c3.metric("Units Deployed", result["num_units_used"])
    st.write(f"Deployed stations: {', '.join(n for _, _, n in result['deployed'])}")
    with st.expander("Raw result"):
        st.json(result)

# ---------------------------------------------------------------- visual intelligence
st.header("Visual Intelligence")
m1, m2 = st.columns(2)
with m1:
    st.subheader("Hotspot Map (Folium)")
    if HOTSPOT_MAP_HTML.exists():
        st.iframe(HOTSPOT_MAP_HTML.read_text(encoding="utf-8"), height=480)
    else:
        st.info("Run: python -m backend.ml.hotspot_map")
with m2:
    st.subheader("Criminal Network (PyVis)")
    if NETWORK_GRAPH_HTML.exists():
        st.iframe(NETWORK_GRAPH_HTML.read_text(encoding="utf-8"), height=480)
    else:
        st.info("Run: python -m backend.ml.network_graph")

# ---------------------------------------------------------------- SHAP explainability
st.header("Why Is An Area Risky? (SHAP)")
try:
    from backend.ml.explainability import ShapExplainer
    explainer = ShapExplainer()
    districts = ["Bangalore", "Mysuru", "Mangaluru City", "Hubballi Dharwad", "Belagavi District"]
    pick = st.selectbox("Select district", districts)
    if pick:
        exp = explainer.explain_district(pick)
        if exp:
            st.write(exp.get("plain_english", ""))
            contrib = exp.get("contributions", [])
            if contrib:
                df = pd.DataFrame(contrib)
                st.bar_chart(df.set_index("label")["shap_value"])
        else:
            st.info("No SHAP explanation for that district.")
except Exception as exc:
    st.info(f"SHAP unavailable: {exc}")

st.caption("Backend modules: backend.ml.{hotspots,patrol_optimizer,hotspot_map,network_graph,explainability}")
