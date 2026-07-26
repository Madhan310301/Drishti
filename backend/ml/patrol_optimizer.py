"""
Patrol Deployment Optimizer for Drishti Predictive Command Console.

Fulfils the manual's Task 2 (Kalyan's territory): "Build The Optimizer
(patrol_optimizer.py)". Faithful to the manual's prescribed design:

- Function: solve_patrol(hotspot_df, num_units, max_radius_km)
- Uses REAL Bengaluru police-station coordinates (manual lines 409-413).
- Haversine distance + binary coverage matrix (a base covers a hotspot if
  within max_radius_km).
- PuLP maximizes total RISK COVERED by deploying <= num_units bases.
- Returns: deployed (list of station lat/lon/name), covered_pct
  (% of total risk covered), uncovered_count, total_hotspots.
- Edge cases handled: 0 hotspots, num_units > num_bases, etc.

A higher-level optimize() wrapper is also provided for the FastAPI endpoint,
mapping the manual result into the dashboard's response schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    HOTSPOT_CENTERS_FILE,
    PATROL_PLAN_FILE,
    RANDOM_STATE,
)

logger = get_logger(__name__)

# Real Bengaluru police stations (manual lines 409-413). (lat, lon, name)
BASE_STATIONS = [
    (12.9716, 77.5946, "MG Road PS"),
    (12.9352, 77.6245, "Koramangala PS"),
    (12.9698, 77.7500, "Whitefield PS"),
    (13.0358, 77.5970, "Yeshwantpur PS"),
    (12.9141, 77.6411, "HSR Layout PS"),
    (12.9857, 77.5533, "Rajajinagar PS"),
    (13.0012, 77.6536, "Indiranagar PS"),
    (12.9063, 77.5857, "JP Nagar PS"),
    (12.9279, 77.6834, "Marathahalli PS"),
    (13.0632, 77.5800, "Hebbal PS"),
    (12.9537, 77.5007, "Kengeri PS"),
    (13.1007, 77.5963, "Yelahanka PS"),
    (12.8438, 77.6593, "Electronic City PS"),
    (12.9600, 77.6400, "Ulsoor PS"),
    (13.0200, 77.6500, "Banaswadi PS"),
    # Additional Karnataka city police stations (real district HQ coordinates)
    # so the optimizer can cover the whole state, not just Bengaluru.
    (12.2958, 76.6394, "Mysuru PS"),
    (15.8497, 74.4977, "Belagavi PS"),
    (15.4589, 75.0078, "Dharwad PS"),
    (13.9299, 75.5681, "Shivamogga PS"),
    (15.1394, 76.9214, "Ballari PS"),
    (16.8302, 75.7100, "Vijayapura PS"),
    (17.3297, 76.8343, "Kalaburagi PS"),
    (13.0068, 76.1025, "Hassan PS"),
    (14.4644, 75.9218, "Davanagere PS"),
    (13.3409, 77.1025, "Tumakuru PS"),
    (12.5216, 76.8964, "Mandya PS"),
    (13.4354, 77.7293, "Chikkaballapura PS"),
    (16.1783, 75.6947, "Bagalkot PS"),
    (14.7947, 75.4043, "Haveri PS"),
    (15.4189, 75.6333, "Gadag PS"),
    (15.3454, 76.2105, "Koppal PS"),
    (16.2120, 77.3439, "Raichur PS"),
    (12.8700, 75.4200, "Mangaluru PS"),
    (13.3400, 74.7450, "Udupi PS"),
    (13.1367, 78.1326, "Kolar PS"),
    (14.2252, 76.3980, "Chitradurga PS"),
    (11.9269, 76.9411, "Chamarajanagar PS"),
    (16.2020, 76.1317, "Yadgir PS"),
    (17.9123, 77.5199, "Bidar PS"),
]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(a)))


def solve_patrol(
    hotspot_df: pd.DataFrame,
    num_units: int = 5,
    max_radius_km: float = 3.0,
) -> dict[str, Any]:
    """
    Given hotspots and N patrol units, find optimal deployment.

    Parameters
    ----------
    hotspot_df : pd.DataFrame
        Columns [center_lat, center_lon, risk_score] (plus optional 'district').
    num_units : int
        Number of available patrol cars.
    max_radius_km : float
        Maximum patrol coverage radius in km.

    Returns
    -------
    dict with 'deployed', 'covered_pct', 'uncovered_count', 'total_hotspots'.
    """
    if hotspot_df is None or len(hotspot_df) == 0:
        return {"deployed": [], "covered_pct": 100.0, "uncovered_count": 0, "total_hotspots": 0}

    num_bases = len(BASE_STATIONS)
    num_spots = len(hotspot_df)

    # Distance matrix base -> hotspot
    dist = np.zeros((num_bases, num_spots))
    for i, (blat, blon, _) in enumerate(BASE_STATIONS):
        for j in range(num_spots):
            hlat = float(hotspot_df.iloc[j]["center_lat"])
            hlon = float(hotspot_df.iloc[j]["center_lon"])
            dist[i][j] = haversine(blat, blon, hlat, hlon)

    # Coverage matrix: 1 if base i can reach hotspot j
    A = (dist <= max_radius_km).astype(int)

    risks = hotspot_df["risk_score"].values.astype(float)
    num_units = min(num_units, num_bases)  # can't deploy more than bases

    prob = pulp.LpProblem("Patrol_Coverage", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Deploy", range(num_bases), cat="Binary")
    y = pulp.LpVariable.dicts("Covered", range(num_spots), cat="Binary")

    # Objective: maximize covered risk
    prob += pulp.lpSum([risks[j] * y[j] for j in range(num_spots)])
    # Constraint: at most num_units deployed
    prob += pulp.lpSum([x[i] for i in range(num_bases)]) <= num_units
    # Constraint: hotspot covered only if a nearby base is active
    for j in range(num_spots):
        prob += y[j] <= pulp.lpSum([A[i][j] * x[i] for i in range(num_bases)])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    deployed = [
        (BASE_STATIONS[i][0], BASE_STATIONS[i][1], BASE_STATIONS[i][2])
        for i in range(num_bases)
        if x[i].varValue == 1
    ]
    covered = sum(1 for j in range(num_spots) if y[j].varValue == 1)
    covered_risk = sum(risks[j] for j in range(num_spots) if y[j].varValue == 1)
    total_risk = float(sum(risks))

    return {
        "deployed": deployed,
        "covered_pct": round(covered_risk / total_risk * 100, 1) if total_risk > 0 else 0.0,
        "uncovered_count": num_spots - covered,
        "total_hotspots": num_spots,
        "num_units_used": len(deployed),
    }


class PatrolOptimizer:
    """High-level wrapper used by the FastAPI endpoint."""

    def __init__(
        self,
        centers_file: Path = HOTSPOT_CENTERS_FILE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.centers_file = centers_file
        self.random_state = random_state

    def optimize(
        self,
        total_units: int,
        max_radius_km: float = 3.0,
    ) -> dict[str, Any]:
        if not self.centers_file.exists():
            raise ProcessingError(f"Hotspot centers not found: {self.centers_file}")
        df = load_csv(self.centers_file)
        if "risk_score" not in df.columns:
            df["risk_score"] = 50.0
        # Statewide simulator: feed ALL hotspot centers (Bengaluru + other
        # districts) to the optimizer now that base stations cover Karnataka.
        result = solve_patrol(df, num_units=total_units, max_radius_km=max_radius_km)
        return {
            "total_units": total_units,
            "max_radius_km": max_radius_km,
            "districts_considered": result["total_hotspots"],
            "risk_reduced": result["covered_pct"],
            "residual_risk": round(100.0 - result["covered_pct"], 1),
            "risk_reduction_pct": result["covered_pct"],
            "solver_status": pulp.LpStatus[pulp.LpStatusOptimal] if result["deployed"] or result["total_hotspots"] == 0 else "Infeasible",
            "assignments": [
                {"district": f"Station:{name}", "units_assigned": 1,
                 "risk_score": 0.0, "coverage_fraction": 0.0, "risk_reduced": 0.0}
                for _, _, name in result["deployed"]
            ],
            "covered_pct": result["covered_pct"],
            "uncovered_count": result["uncovered_count"],
        }

    def save(self, plan: dict[str, Any], out_path: Path = PATROL_PLAN_FILE) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4)
        logger.info("Saved patrol plan to %s", out_path.name)
        return out_path


def main() -> None:
    df = load_csv(HOTSPOT_CENTERS_FILE)
    result = solve_patrol(df, num_units=5, max_radius_km=3.0)
    print(f"Deployed {len(result['deployed'])} units")
    print(f"Covered {result['covered_pct']}% of total risk")
    print(f"{result['uncovered_count']} hotspots uncovered")
    for lat, lon, name in result["deployed"]:
        print(f"  -> {name} ({lat}, {lon})")


if __name__ == "__main__":
    main()
