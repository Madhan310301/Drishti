#!/usr/bin/env python
"""
Build a fully-static Drishti dashboard (no backend required).

Runs the real backend ML/ETL functions, snapshots every API endpoint the
dashboard uses into app/static-data/*.json, precomputes the patrol optimizer
grid, copies the generated map/network HTMLs next to the page, and writes a
static-data/manifest.json listing available SHAP districts.

Run from repo root:  python scripts/build_static_dashboard.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backend.api.routes import (
    hotspot_centers, network_graph, explain_global, explain_districts,
    analytics_summary,
)
from backend.ml.patrol_optimizer import PatrolOptimizer
import pandas as pd


def load_districts_from_csv():
    """Mirror get_all_districts() from the committed processed CSV."""
    p = ROOT / "data" / "processed" / "district_profiles.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["district", "total_population", "male_population", "female_population",
            "literacy_rate", "urban_pct", "work_participation_rate"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")


def load_crime_hotspots_from_csv(limit=50):
    """Mirror get_hotspots() ordered by risk/hotspot score.
    Source: data/output/hotspot_predictions.csv (generated ML output)."""
    fs = ROOT / "data" / "output" / "hotspot_predictions.csv"
    if not fs.exists():
        # fallback to feature_store if predictions absent
        fs = ROOT / "data" / "output" / "feature_store.csv"
    if not fs.exists():
        return []
    df = pd.read_csv(fs)
    score_cols = [c for c in ["risk_score", "hotspot_score", "crime_forecast",
                              "current_crime_rate_per_100k", "anomaly_score", "cluster_id"]
                  if c in df.columns]
    sort_cols = [c for c in ["risk_score", "hotspot_score"] if c in df.columns]
    df = df.sort_values(by=sort_cols, ascending=False) if sort_cols else df
    out = []
    for _, row in df.head(limit).iterrows():
        rec = {"district": row.get("district")}
        for c in score_cols:
            rec[c] = row.get(c)
        out.append(rec)
    return out

STATIC = ROOT / "app" / "static-data"
VIZ = ROOT / "app" / "static-viz"
STATIC.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)


def dump(name: str, data) -> None:
    (STATIC / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print("wrote", name)


# --- endpoints the dashboard fetches ---
try:
    dump("hotspot_centers.json", hotspot_centers())
except Exception as e:
    dump("hotspot_centers.json", {"error": str(e)})

try:
    dump("offender_network.json", network_graph())
except Exception as e:
    dump("offender_network.json", {"error": str(e)})

try:
    dump("analytics_summary.json", analytics_summary())
except Exception as e:
    dump("analytics_summary.json", {"error": str(e)})

try:
    dump("crime_hotspots.json", load_crime_hotspots_from_csv(limit=50))
except Exception as e:
    dump("crime_hotspots.json", {"error": str(e)})

try:
    dump("districts.json", load_districts_from_csv())
except Exception as e:
    dump("districts.json", {"error": str(e)})

try:
    dump("shap_global.json", explain_global())
except Exception as e:
    dump("shap_global.json", {"error": str(e)})

# SHAP per-district: snapshot every available district
try:
    dists = explain_districts()
except Exception:
    dists = []
shap_records = {}
for d in dists:
    try:
        from backend.api.routes import explain_district
        r = explain_district(d)
        if r:
            # r is a Pydantic model -> convert to plain dict so JSON is real,
            # not a str() dump.
            shap_records[d] = r.model_dump() if hasattr(r, "model_dump") else r
    except Exception as e:
        shap_records[d] = {"error": str(e)}
dump("shap_districts.json", shap_records)
dump("manifest.json", {"shap_districts": dists})

# --- patrol optimizer grid (precompute for slider range) ---
opt = PatrolOptimizer()
grid = []
try:
    for units in range(1, 31):
        for radius in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            try:
                plan = opt.optimize(total_units=units, max_radius_km=radius)
                grid.append({
                    "units": units, "radius": radius,
                    "covered_pct": plan["covered_pct"],
                    "uncovered_count": plan["uncovered_count"],
                    "assignments": [a["district"] for a in plan["assignments"]],
                })
            except Exception:
                pass
    dump("patrol_grid.json", grid)
    print(f"patrol grid: {len(grid)} cells")
except Exception as e:
    dump("patrol_grid.json", {"error": str(e)})

# --- copy generated viz HTML next to the page ---
for src in ["data/output/hotspot_map.html", "data/output/offender_network.html"]:
    p = ROOT / src
    if p.exists():
        import shutil
        shutil.copy(p, VIZ / p.name)
        print("copied", p.name)

print("\nSTATIC BUILD COMPLETE")
print("static-data files:", len(list(STATIC.glob('*.json'))))
