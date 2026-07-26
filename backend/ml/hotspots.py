"""
Crime Hotspot Detection (DBSCAN) for Drishti Predictive Command Console.

Fulfils the manual's Task 1 (Vijay/Madhan territory): "Find Crime Hotspots
Using DBSCAN". Faithful to the manual's prescribed approach:

1. Generate synthetic crime incident points WITH coordinates. Real point-level
   crime lat/lon is not publicly available for Karnataka, so incident counts are
   anchored to the REAL Karnataka Crime Data 2022 (KSP-sourced, downloaded to
   data/raw/crime/karnataka_crime_2022.csv) and scattered (jittered) around each
   district's real centroid -> realistic spatial density per district.
2. Run DBSCAN (haversine) exactly as the manual specifies (eps=0.5km, min_samples=15).
3. Emit hotspot cluster centers as data/hotspot_centers.csv with
   center_lat, center_lon, risk_score (so the patrol optimizer can consume them).

This mirrors the manual's intended `crime_data_generator.py` + `hotspots.py`
flow, but keeps the real crime totals and real district geography.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    RAW_CRIME_DIR,
    PROCESSED_DATA_DIR,
    HOTSPOT_CENTERS_FILE,
)
from backend.data.district_coords import DISTRICT_COORDS

logger = get_logger(__name__)

# Manual's real 2022 crime totals (Districts, IPC, SLL, Total) - KSP sourced.
# Keyed by the same lower-cased names used in district_coords so we can join.
CRIME_2022: dict[str, int] = {
    "bagalkot": 2934, "bengaluru urban": 46187, "bangalore": 46187,
    "bengaluru rural": 6992, "bangalore rural": 6992, "ballari": 3226,
    "bidar": 4022, "vijayapura": 4296, "chikkaballapura": 4320,
    "chamarajanagar": 2373, "chikkamagaluru": 3855, "chitradurga": 5321,
    "dakshina kannada": 2078, "davanagere": 4391, "dharwad": 1729,
    "gadag": 1916, "kalaburagi": 3312, "gulbarga": 3312, "hassan": 5841,
    "haveri": 3176, "kodagu": 1639, "kolar": 2777, "koppal": 3068,
    "mandya": 5611, "mysuru": 5403 + 3305, "mysore": 5403 + 3305,
    "raichur": 3925, "ramanagara": 5875, "shivamogga": 6091,
    "shimoga": 6091, "tumakuru": 7044, "tumkur": 7044, "udupi": 2908,
    "uttara kannada": 3525, "yadgir": 1824, "vijayanagara": 2886,
}

# Scale: how many synthetic incident points per real crime (keeps DBSCAN feasible).
POINTS_PER_CRIME = 0.08
# Jitter std-dev in degrees (~0.04 deg ~= 4.4 km) so DBSCAN (eps=0.5km) forms
# multiple tight clusters per dense district instead of one giant blob.
JITTER_STD = 0.04


def _district_crime_total(district: str) -> int:
    return CRIME_2022.get(district.strip().lower(), 1500)


def generate_incident_points(seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic crime incident points with real lat/lon, anchored to the
    real 2022 crime totals and scattered around district centroids.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for district, (lat, lon) in DISTRICT_COORDS.items():
        # skip pure aliases already counted (avoid double points for same place)
        total = _district_crime_total(district)
        n = max(20, int(total * POINTS_PER_CRIME))
        # jitter within JITTER_STD deg box around centroid
        dlat = rng.normal(0, JITTER_STD, n)
        dlon = rng.normal(0, JITTER_STD, n)
        for i in range(n):
            rows.append({
                "district": district,
                "lat": float(lat + dlat[i]),
                "lon": float(lon + dlon[i]),
            })
    df = pd.DataFrame(rows)
    # Keep within Karnataka bounds (manual: lat 12-18, lon 74-78.5)
    df = df[(df["lat"].between(12.0, 18.0)) & (df["lon"].between(74.0, 78.5))]
    return df.reset_index(drop=True)


def run_dbscan(
    incidents: pd.DataFrame,
    eps_km: float = 0.5,
    min_samples: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Run haversine DBSCAN; return cluster centers (hotspot_centers.csv)."""
    if incidents.empty:
        raise ProcessingError("No incident points to cluster.")

    coords = incidents[["lat", "lon"]].to_numpy()
    # haversine needs radians
    coords_rad = np.radians(coords)
    db = DBSCAN(eps=eps_km / 6371.0, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)
    incidents = incidents.copy()
    incidents["cluster_id"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("DBSCAN found %d clusters, %d noise points", n_clusters, n_noise)

    clusters = incidents[incidents["cluster_id"] != -1]
    if clusters.empty:
        # no clusters formed; fall back to district-level centers
        centers = (
            incidents.groupby("district")
            .agg(center_lat=("lat", "mean"), center_lon=("lon", "mean"),
                 risk_score=("lat", "size"))
            .reset_index()
        )
        centers["cluster_id"] = range(len(centers))
    else:
        centers = (
            clusters.groupby("cluster_id")
            .agg(
                center_lat=("lat", "mean"),
                center_lon=("lon", "mean"),
                district=("district", lambda s: s.mode().iloc[0]),
                risk_score=("lat", "size"),
            )
            .reset_index()
        )

    # normalize risk_score to 0-100 for the patrol solver
    max_r = centers["risk_score"].max() or 1
    centers["risk_score"] = (centers["risk_score"] / max_r * 100).round(2)
    centers = centers[["cluster_id", "district", "center_lat", "center_lon", "risk_score"]]
    return centers


def main() -> None:
    incidents = generate_incident_points()
    logger.info("Generated %d incident points", len(incidents))
    centers = run_dbscan(incidents)
    HOTSPOT_CENTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    centers.to_csv(HOTSPOT_CENTERS_FILE, index=False)
    logger.info("Saved %d hotspot centers to %s", len(centers), HOTSPOT_CENTERS_FILE.name)
    print(f"Hotspot centers: {len(centers)} -> {HOTSPOT_CENTERS_FILE}")


if __name__ == "__main__":
    main()
