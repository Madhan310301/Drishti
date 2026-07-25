"""
Offender network generator module for Drishti ETL Pipeline.

Generates a synthetic suspect/offender relationship dataset used to power the
Criminal Network Analysis feature. No real identities are used -- suspects,
aliases, and FIR references are entirely synthetic and exist only to
demonstrate the network-graph visualization (co-accused, gang, and shared
modus-operandi links) described in the project manual.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any
import random

import pandas as pd

from backend.common.logger import get_logger
from backend.common.helpers import load_csv, save_csv
from backend.common.constants import CRIME_CATEGORIES
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    DISTRICT_PROFILE_FILE,
    OFFENDER_NODES_FILE,
    OFFENDER_EDGES_FILE,
    OFFENDER_GANG_NAMES,
    OFFENDER_RELATIONSHIP_TYPES,
    OFFENDER_SUSPECT_COUNT,
    OFFENDER_EDGE_COUNT,
    RANDOM_STATE,
)

logger = get_logger(__name__)

_ALIAS_PREFIXES = [
    "Bandit", "Shadow", "Cobra", "Iron", "Silent", "Wolf", "Ghost", "Tiger",
    "Bullet", "Raven", "Viper", "Steel", "Night", "Falcon", "Storm", "Fox",
    "Hawk", "Rusty", "Blade", "Scar", "Chota", "Bada", "Local", "Loose",
]

_ALIAS_SUFFIXES = [
    "Raju", "Manju", "Kumar", "Shetty", "Gowda", "Naik", "Rao", "Reddy",
    "Swamy", "Iyer", "Achar", "Bhai", "Anna", "Master", "Setty", "Naidu",
]


class OffenderNetworkGenerator:
    """
    Generates synthetic offender nodes (suspects) and edges (relationships)
    for the criminal network graph feature.

    Suspects are distributed across real, standardized Karnataka districts
    (pulled from the already-processed district profile file) so the network
    graph aligns with the rest of the platform's district-level intelligence.
    """

    def __init__(
        self,
        district_profile_file=DISTRICT_PROFILE_FILE,
        suspect_count: int = OFFENDER_SUSPECT_COUNT,
        edge_count: int = OFFENDER_EDGE_COUNT,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Initialize OffenderNetworkGenerator.

        Parameters
        ----------
        district_profile_file : Path
            Processed district profile CSV, used as the source of truth for
            standardized district names.
        suspect_count : int
            Number of synthetic suspect nodes to generate.
        edge_count : int
            Number of synthetic relationship edges to generate.
        random_state : int
            Seed for reproducible generation, matching the project-wide
            ML random state.
        """
        self.district_profile_file = district_profile_file
        self.suspect_count = suspect_count
        self.edge_count = edge_count
        self.rng = random.Random(random_state)

    def _load_districts(self) -> list[str]:
        """
        Load the canonical district list from the processed district profiles.

        Returns
        -------
        list[str]
            Standardized district names.

        Raises
        ------
        ProcessingError
            If the district profile file has not been generated yet.
        """
        try:
            df = load_csv(self.district_profile_file)
        except Exception as exc:
            raise ProcessingError(
                "Cannot generate offender network before district profiles "
                "exist. Run the census processor first."
            ) from exc

        return sorted(df["district"].dropna().unique().tolist())

    def _generate_alias(self, used_aliases: set[str]) -> str:
        """
        Generate a unique synthetic alias.

        Parameters
        ----------
        used_aliases : set[str]
            Aliases already assigned, to avoid duplicates.

        Returns
        -------
        str
            A unique alias string.
        """
        while True:
            alias = f"{self.rng.choice(_ALIAS_PREFIXES)} {self.rng.choice(_ALIAS_SUFFIXES)}"
            if alias not in used_aliases:
                used_aliases.add(alias)
                return alias

    def generate_nodes(self, districts: list[str]) -> pd.DataFrame:
        """
        Generate synthetic suspect node records.

        Parameters
        ----------
        districts : list[str]
            Standardized district names to distribute suspects across.

        Returns
        -------
        pd.DataFrame
            Suspect nodes with columns: suspect_id, alias, primary_crime_type,
            gang_name, district.
        """
        used_aliases: set[str] = set()
        records = []

        for i in range(1, self.suspect_count + 1):
            records.append(
                {
                    "suspect_id": f"SUS-{i:04d}",
                    "alias": self._generate_alias(used_aliases),
                    "primary_crime_type": self.rng.choice(CRIME_CATEGORIES),
                    "gang_name": self.rng.choice(OFFENDER_GANG_NAMES),
                    "district": self.rng.choice(districts),
                }
            )

        return pd.DataFrame(records)

    def generate_edges(self, nodes: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic relationship edges between suspects.

        Edges are biased towards suspects who already share a gang or
        district, so the resulting graph has realistic-looking clusters
        instead of pure random noise.

        Parameters
        ----------
        nodes : pd.DataFrame
            Suspect node records produced by ``generate_nodes``.

        Returns
        -------
        pd.DataFrame
            Relationship edges with columns: suspect_1, suspect_2,
            relationship_type, linked_fir_id, weight.
        """
        suspect_ids = nodes["suspect_id"].tolist()
        gang_lookup = dict(zip(nodes["suspect_id"], nodes["gang_name"]))
        district_lookup = dict(zip(nodes["suspect_id"], nodes["district"]))

        all_pairs = list(combinations(suspect_ids, 2))

        def _pair_weight(pair: tuple[str, str]) -> float:
            s1, s2 = pair
            score = 1.0
            if gang_lookup[s1] == gang_lookup[s2] and gang_lookup[s1] != "Independent":
                score += 4.0
            if district_lookup[s1] == district_lookup[s2]:
                score += 2.0
            return score

        weights = [_pair_weight(pair) for pair in all_pairs]

        chosen_pairs = set()
        target = min(self.edge_count, len(all_pairs))

        while len(chosen_pairs) < target:
            pair = self.rng.choices(all_pairs, weights=weights, k=1)[0]
            chosen_pairs.add(pair)

        records = []
        for idx, (s1, s2) in enumerate(sorted(chosen_pairs), start=1):
            same_gang = gang_lookup[s1] == gang_lookup[s2] and gang_lookup[s1] != "Independent"
            relationship_type = "same_gang" if same_gang else self.rng.choice(OFFENDER_RELATIONSHIP_TYPES)

            records.append(
                {
                    "suspect_1": s1,
                    "suspect_2": s2,
                    "relationship_type": relationship_type,
                    "linked_fir_id": f"FIR-{self.rng.randint(2020, 2026)}-{idx:05d}",
                    "weight": self.rng.randint(1, 5),
                }
            )

        return pd.DataFrame(records)

    def generate(self) -> dict[str, Any]:
        """
        Run full offender network generation and persist outputs.

        Returns
        -------
        dict[str, Any]
            Summary of generated node and edge counts.
        """
        logger.info("Generating synthetic offender network...")

        districts = self._load_districts()
        nodes = self.generate_nodes(districts)
        edges = self.generate_edges(nodes)

        save_csv(nodes, OFFENDER_NODES_FILE)
        save_csv(edges, OFFENDER_EDGES_FILE)

        logger.info(
            f"Offender network generated: {len(nodes)} suspects, {len(edges)} relationships."
        )

        return {
            "suspect_count": len(nodes),
            "edge_count": len(edges),
            "nodes_file": str(OFFENDER_NODES_FILE),
            "edges_file": str(OFFENDER_EDGES_FILE),
        }


def main() -> None:
    """Entry point for standalone offender network generation."""
    generator = OffenderNetworkGenerator()
    summary = generator.generate()
    print(
        f"Offender network generated: {summary['suspect_count']} suspects, "
        f"{summary['edge_count']} relationships."
    )
    print(f"  [OK] {summary['nodes_file']}")
    print(f"  [OK] {summary['edges_file']}")


if __name__ == "__main__":
    main()
