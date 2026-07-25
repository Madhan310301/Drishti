"""
Patrol Deployment Simulator for Drishti Predictive Command Console.

Fulfils the manual's "Patrol Recommendation Engine" (Kalyan's role) -
the project's headline differentiator. Uses PuLP (linear programming)
to allocate a LIMITED pool of patrol units across districts so that
expected crime-risk reduction is maximised.

Design decisions (per PROJECT_MEMORY dev principles):
- Inputs are district risk_score + crime_forecast from the existing ML
  pipeline outputs (hotspot_predictions.csv). No new data source needed.
- We model coverage with diminishing returns: assigning the 1st unit to a
  district yields full effect; each additional unit in the same district
  yields less (a standard "saturation" assumption). This stops the solver
  from dumping all units in one district.
- Objective: maximize sum(district_risk * coverage_units_assigned).
- Constraints: total units == available; per-district units in [0, max_per_district].
- Pure, deterministic, fast (<1s on 28 districts). No randomness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pulp

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    HOTSPOT_PREDICTIONS_FILE,
    PATROL_PLAN_FILE,
    RANDOM_STATE,
)

logger = get_logger(__name__)

# A patrol unit covers this fraction of a district's risk on the 1st assignment,
# and each further unit in the same district adds this *multiplier* of the remainder
# (diminishing returns). Tune here if deployment doctrine changes.
FIRST_UNIT_COVERAGE = 0.45
EXTRA_UNIT_FACTOR = 0.55


class PatrolOptimizer:
    """
    Allocates patrol units across districts to maximise risk reduction.

    Districts are pulled from hotspot_predictions.csv (risk_score + crime_forecast).
    """

    def __init__(
        self,
        predictions_file: Path = HOTSPOT_PREDICTIONS_FILE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.predictions_file = predictions_file
        self.random_state = random_state
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if not self.predictions_file.exists():
            raise ProcessingError(f"Hotspot predictions not found: {self.predictions_file}")
        df = load_csv(self.predictions_file)
        if df.empty:
            raise ProcessingError("Hotspot predictions are empty.")
        needed = {"district", "risk_score", "crime_forecast"}
        missing = needed - set(df.columns)
        if missing:
            raise ProcessingError(f"Hotspot predictions missing columns: {missing}")
        # Normalise risk to a non-negative weight.
        df = df.copy()
        df["risk_weight"] = df["risk_score"].clip(lower=0.0)
        return df

    def optimize(
        self,
        total_units: int,
        max_per_district: int | None = None,
    ) -> dict[str, Any]:
        """
        Solve the patrol allocation.

        Parameters
        ----------
        total_units : int
            Total patrol units available tonight.
        max_per_district : int | None
            Cap on units assigned to a single district. Defaults to total_units.

        Returns
        -------
        dict[str, Any]
            Plan with assignment, per-district coverage, and totals.
        """
        df = self._load()
        self._df = df
        districts = df["district"].tolist()
        risks = df["risk_weight"].tolist()
        n = len(districts)

        if total_units <= 0:
            raise ProcessingError("total_units must be a positive integer.")
        if max_per_district is None:
            max_per_district = total_units
        max_per_district = max(1, min(max_per_district, total_units))

        # --- PuLP model ---
        prob = pulp.LpProblem("PatrolDeployment", pulp.LpMaximize)

        # x[i][k] = 1 if k-th unit (0-indexed) is assigned to district i
        x = {}
        for i in range(n):
            for k in range(total_units):
                x[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", cat="Binary")

        # Each unit assigned to exactly one district.
        for k in range(total_units):
            prob += pulp.lpSum(x[(i, k)] for i in range(n)) == 1, f"unit_{k}_one_district"

        # Per-district cap.
        units_in = {i: pulp.lpSum(x[(i, k)] for k in range(total_units)) for i in range(n)}
        for i in range(n):
            prob += units_in[i] <= max_per_district, f"cap_{i}"

        # Coverage with diminishing returns (linearised via ordered unit slots).
        coverage = []
        for i in range(n):
            cov = 0.0
            remaining = 1.0
            factor = 1.0
            for k in range(total_units):
                add = FIRST_UNIT_COVERAGE if k == 0 else (FIRST_UNIT_COVERAGE * factor)
                cov += remaining * add * x[(i, k)]
                remaining = remaining * (1.0 - add)
                factor *= EXTRA_UNIT_FACTOR
            coverage.append(cov)

        prob += pulp.lpSum(risks[i] * coverage[i] for i in range(n)), "total_risk_reduction"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            raise ProcessingError(f"Patrol optimizer failed (status={pulp.LpStatus[prob.status]}).")

        # --- Build assignment ---
        assignment: dict[str, int] = {d: 0 for d in districts}
        for i in range(n):
            for k in range(total_units):
                if x[(i, k)].value() is not None and x[(i, k)].value() > 0.5:
                    assignment[districts[i]] += 1

        baseline_risk = float(sum(risks))
        covered = [coverage[i].value() if coverage[i].value() is not None else 0.0 for i in range(n)]
        risk_reduction = float(sum(risks[i] * covered[i] for i in range(n)))
        residual_risk = baseline_risk - risk_reduction

        assignments = []
        for i, d in enumerate(districts):
            if assignment[d] > 0:
                assignments.append(
                    {
                        "district": d,
                        "units_assigned": assignment[d],
                        "risk_score": round(float(risks[i]), 2),
                        "coverage_fraction": round(float(covered[i]), 4),
                        "risk_reduced": round(float(risks[i] * covered[i]), 2),
                    }
                )
        assignments.sort(key=lambda a: a["units_assigned"], reverse=True)

        return {
            "total_units": total_units,
            "max_per_district": max_per_district,
            "districts_considered": n,
            "baseline_risk": round(baseline_risk, 2),
            "risk_reduced": round(risk_reduction, 2),
            "residual_risk": round(residual_risk, 2),
            "risk_reduction_pct": round(100.0 * risk_reduction / baseline_risk, 2) if baseline_risk else 0.0,
            "assignments": assignments,
            "solver_status": pulp.LpStatus[prob.status],
        }

    def save(self, plan: dict[str, Any], out_path: Path = PATROL_PLAN_FILE) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4)
        logger.info("Saved patrol plan to %s", out_path.name)
        return out_path


def main() -> None:
    optimizer = PatrolOptimizer()
    plan = optimizer.optimize(total_units=20, max_per_district=4)
    path = optimizer.save(plan)
    print(f"Patrol plan saved to {path}")
    print(f"Risk reduction: {plan['risk_reduction_pct']}% across {plan['districts_considered']} districts")


if __name__ == "__main__":
    main()
