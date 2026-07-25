"""
Criminal Network Visualization for Drishti Predictive Command Console.

Fulfils the manual's "Criminal Network Analysis" (Jenifa's role): render the
offender co-offending network as an interactive HTML graph using PyVis/NetworkX.

Design decisions (per PROJECT_MEMORY dev principles):
- Reads the existing ETL outputs: offender_nodes.csv + offender_edges.csv.
  No new data source; just visualizes what offender_network.py already built.
- Nodes colored by gang, sized by degree (number of links) so kingpins stand out.
- Edges colored by relationship_type (same_gang/associate/shared_mo) and
  weighted by link strength, so investigators see tight clusters fast.
- Optional district filter via build(district=...) so the dashboard can show
  one district's sub-network.
- Output is a standalone HTML file (opens in any browser, no server needed) -
  matches the existing vanilla-JS dashboard (Option A) integration plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pyvis.network import Network

from backend.common.logger import get_logger
from backend.common.helpers import load_csv
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    OFFENDER_NODES_FILE,
    OFFENDER_EDGES_FILE,
    NETWORK_GRAPH_HTML,
)

logger = get_logger(__name__)

# Relationship -> color (edges)
EDGE_COLORS = {
    "same_gang": "#ff4d6d",   # red: strongest tie
    "associate": "#ffd166",   # amber
    "shared_mo": "#4cc9f0",   # blue
}
DEFAULT_EDGE_COLOR = "#94a3b8"


class NetworkGraphBuilder:
    """Builds an interactive PyVis HTML graph from offender network CSVs."""

    def __init__(
        self,
        nodes_file: Path = OFFENDER_NODES_FILE,
        edges_file: Path = OFFENDER_EDGES_FILE,
        out_file: Path = NETWORK_GRAPH_HTML,
    ) -> None:
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.out_file = out_file

    def build(self, district: str | None = None) -> dict[str, Any]:
        """
        Render the network to HTML.

        Parameters
        ----------
        district : str | None
            If given, restrict to suspects in this district.

        Returns
        -------
        dict[str, Any]
            Metadata: node count, edge count, districts covered, html path.
        """
        if not self.nodes_file.exists():
            raise ProcessingError(f"Offender nodes not found: {self.nodes_file}")
        if not self.edges_file.exists():
            raise ProcessingError(f"Offender edges not found: {self.edges_file}")

        nodes = load_csv(self.nodes_file)
        edges = load_csv(self.edges_file)

        # Degree (link count) per suspect for node sizing.
        degree = {}
        for _, e in edges.iterrows():
            degree[e["suspect_1"]] = degree.get(e["suspect_1"], 0) + 1
            degree[e["suspect_2"]] = degree.get(e["suspect_2"], 0) + 1

        if district:
            district = district.strip().lower()
            wanted = set(nodes[nodes["district"].str.strip().str.lower() == district]["suspect_id"])
            if not wanted:
                raise ProcessingError(f"No suspects found in district '{district}'.")
            nodes = nodes[nodes["suspect_id"].isin(wanted)].copy()
            edge_set = set(nodes["suspect_id"])
            edges = edges[(edges["suspect_1"].isin(edge_set)) & (edges["suspect_2"].isin(edge_set))].copy()

        if nodes.empty:
            raise ProcessingError("No nodes to render after filtering.")

        # Stable color per gang.
        gangs = sorted(nodes["gang_name"].dropna().unique().tolist())
        palette = ["#00f2fe", "#f72585", "#7209b7", "#3a0ca3", "#4361ee",
                   "#4cc9f0", "#80ed99", "#ffd166", "#ff7b00", "#ff4d6d"]
        gang_color = {g: palette[i % len(palette)] for i, g in enumerate(gangs)}

        net = Network(height="720px", width="100%", notebook=False, directed=False,
                      bgcolor="#0b1020", font_color="#e2e8f0")
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.04)

        for _, n in nodes.iterrows():
            gid = n.get("suspect_id")
            deg = degree.get(gid, 0)
            color = gang_color.get(n.get("gang_name"), "#64748b")
            title = (
                f"<b>{n.get('alias')}</b><br>"
                f"ID: {gid}<br>"
                f"Crime: {n.get('primary_crime_type')}<br>"
                f"Gang: {n.get('gang_name')}<br>"
                f"District: {n.get('district')}<br>"
                f"Links: {deg}"
            )
            net.add_node(
                gid,
                label=str(n.get("alias")),
                title=title,
                color=color,
                size=12 + min(deg, 12) * 2,  # bigger = more connected
            )

        for _, e in edges.iterrows():
            s1, s2 = e["suspect_1"], e["suspect_2"]
            if s1 not in net.get_nodes() or s2 not in net.get_nodes():
                continue
            rtype = e.get("relationship_type", "")
            color = EDGE_COLORS.get(rtype, DEFAULT_EDGE_COLOR)
            w = float(e.get("weight", 1) or 1)
            net.add_edge(s1, s2, title=f"{rtype} (FIR {e.get('linked_fir_id')})", color=color, width=1 + w * 0.6)

        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        net.write_html(str(self.out_file))
        logger.info("Wrote network graph to %s", self.out_file.name)

        return {
            "html_path": str(self.out_file),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "district_filter": district,
            "gangs": gangs,
            "top_connected": sorted(
                [{"suspect_id": k, "links": v} for k, v in degree.items()],
                key=lambda d: d["links"], reverse=True,
            )[:5],
        }

    def save_metadata(self, meta: dict[str, Any], out_path: Path | None = None) -> Path:
        import json
        out_path = out_path or (self.out_file.parent / "offender_network.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        return out_path


def main() -> None:
    builder = NetworkGraphBuilder()
    meta = builder.build()
    builder.save_metadata(meta)
    print(f"Network graph -> {meta['html_path']} ({meta['node_count']} nodes, {meta['edge_count']} edges)")


if __name__ == "__main__":
    main()
