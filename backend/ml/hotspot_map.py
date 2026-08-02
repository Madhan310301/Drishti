"""
Hotspot Map for Drishti Predictive Command Console.

Fulfils the manual's "Folium / OpenStreetMap Hotspot Map" (Jenifa's role):
render DBSCAN crime hotspots on an interactive Karnataka map. Markers are
sized/colored by risk, and patrol deployments from the optimizer can be
overlaid so commanders see where coverage lands.

Design decisions (per PROJECT_MEMORY dev principles):
- Reads hotspot_predictions.csv (district risk + hotspot_score) and joins to
  embedded Karnataka district centroids (backend/data/district_coords.py).
- No lat/long exists in source data, so centroids are the placement strategy.
- Marker color = risk band (low/med/high/critical); radius scales with risk.
- Optional patrol overlay: pass deployments [{district, units}] to mark covered
  districts with a patrol icon. Keeps map + optimizer decoupled but composable.
- Output: standalone HTML (works offline in the existing dashboard - Option A).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import folium

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    HOTSPOT_PREDICTIONS_FILE,
    HOTSPOT_MAP_HTML,
)
from backend.data.district_coords import get_coords

logger = get_logger(__name__)

# risk_score -> color band
def _risk_color(score: float) -> str:
    if score >= 75:
        return "darkred"
    if score >= 50:
        return "red"
    if score >= 25:
        return "orange"
    return "green"


def _risk_band(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


class HotspotMapBuilder:
    """Builds a Folium HTML map of district crime hotspots."""

    def __init__(
        self,
        predictions_file: Path = HOTSPOT_PREDICTIONS_FILE,
        out_file: Path = HOTSPOT_MAP_HTML,
    ) -> None:
        self.predictions_file = predictions_file
        self.out_file = out_file

    def build(self, deployments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """
        Render the hotspot map.

        Parameters
        ----------
        deployments : list[dict] | None
            Optional list of {"district": str, "units_assigned": int} from the
            patrol optimizer, overlaid as patrol markers.

        Returns
        -------
        dict[str, Any]
            Metadata: plotted count, missing coords, html path, bands.
        """
        if not self.predictions_file.exists():
            raise ProcessingError(f"Hotspot predictions not found: {self.predictions_file}")
        df = load_csv(self.predictions_file)
        if df.empty:
            raise ProcessingError("Hotspot predictions are empty.")

        plotted, missing = [], []
        fmap = folium.Map(
            location=[15.0, 76.0],
            zoom_start=7,
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google Maps"
        )
        deploy_map = {}
        if deployments:
            for d in deployments:
                deploy_map[str(d.get("district", "")).strip().lower()] = int(d.get("units_assigned", 0))

        for _, row in df.iterrows():
            district = str(row["district"])
            coords = get_coords(district)
            if not coords:
                missing.append(district)
                continue
            risk = float(row["risk_score"])
            score = float(row["hotspot_score"])
            band = _risk_band(risk)
            radius = 6 + (risk / 100.0) * 22
            popup = (
                f"<b>{district}</b><br>"
                f"Risk score: {risk:.1f} ({band})<br>"
                f"Hotspot score: {score:.1f}<br>"
                f"Crime forecast: {float(row['crime_forecast']):.0f}<br>"
                f"Anomaly: {float(row['anomaly_score']):.3f}"
            )
            folium.CircleMarker(
                location=coords,
                radius=radius,
                color=_risk_color(risk),
                fill=True,
                fill_color=_risk_color(risk),
                fill_opacity=0.6,
                popup=folium.Popup(popup, max_width=260),
                tooltip=f"{district} ({band})",
            ).add_to(fmap)

            units = deploy_map.get(district.strip().lower())
            if units:
                folium.Marker(
                    location=coords,
                    icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
                    tooltip=f"Patrol: {units} unit(s)",
                ).add_to(fmap)

            plotted.append({"district": district, "risk_score": round(risk, 2), "band": band})

        if not plotted:
            raise ProcessingError("No districts could be placed on the map (missing coordinates).")

        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(self.out_file))
        logger.info("Wrote hotspot map to %s", self.out_file.name)

        bands = {}
        for p in plotted:
            bands[p["band"]] = bands.get(p["band"], 0) + 1

        return {
            "html_path": str(self.out_file),
            "plotted_count": len(plotted),
            "missing_coords": missing,
            "bands": bands,
            "deployments_overlaid": len(deploy_map),
        }

    def save_metadata(self, meta: dict[str, Any], out_path: Path | None = None) -> Path:
        out_path = out_path or (self.out_file.parent / "hotspot_map.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        return out_path


def main() -> None:
    builder = HotspotMapBuilder()
    meta = builder.build()
    builder.save_metadata(meta)
    print(f"Hotspot map -> {meta['html_path']} ({meta['plotted_count']} districts, missing={meta['missing_coords']})")


if __name__ == "__main__":
    main()
