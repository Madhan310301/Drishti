"""
Census data processor module for Drishti ETL Pipeline.
"""

from pathlib import Path
from typing import Any
import time
import pandas as pd

from backend.common.logger import get_logger
from backend.common.helpers import (
    list_csv_files,
    clean_column_names,
    remove_duplicate_rows,
    save_csv,
    standardize_district_name,
)
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    CENSUS_DATASET_DIR,
    DISTRICT_PROFILE_FILE,
    NUMERIC_FILL_VALUE,
)

logger = get_logger(__name__)


class CensusProcessor:
    """
    Processor to load, merge, clean, and aggregate raw Census CSV files
    into a unified district demographic profiles dataset.
    """

    def __init__(self, census_dir: Path = CENSUS_DATASET_DIR) -> None:
        """
        Initialize CensusProcessor.

        Parameters
        ----------
        census_dir : Path
            Path to directory containing raw census CSV files.
        """
        self.census_dir = census_dir

    def process_single_census_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Extract aggregated demographic metrics for a single district census CSV.

        Parameters
        ----------
        file_path : Path
            Path to individual district census file.

        Returns
        -------
        dict[str, Any] | None
            Aggregated metrics for the district or None if invalid.
        """
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty or "District_Name" not in df.columns:
                return None

            raw_district = str(df["District_Name"].dropna().iloc[0])
            district = standardize_district_name(raw_district)

            # Filter rows for CD BLOCK totals
            cd_blocks = df[(df["Level"] == "CD BLOCK") & (df["Total/Rural/Urban"] == "Total")]
            cd_urbans = df[(df["Level"] == "CD BLOCK") & (df["Total/Rural/Urban"] == "Urban")]
            cd_rurals = df[(df["Level"] == "CD BLOCK") & (df["Total/Rural/Urban"] == "Rural")]

            if cd_blocks.empty:
                # Fallback to total rows if CD BLOCK not available
                cd_blocks = df[df["Total/Rural/Urban"] == "Total"]
                cd_urbans = df[df["Total/Rural/Urban"] == "Urban"]
                cd_rurals = df[df["Total/Rural/Urban"] == "Rural"]

            tot_pop = int(cd_blocks["Total Population Person"].sum())
            if tot_pop == 0:
                return None

            tot_male = int(cd_blocks["Total Population Male"].sum())
            tot_female = int(cd_blocks["Total Population Female"].sum())
            tot_lit = int(cd_blocks["Literates Population Person"].sum())
            tot_hh = int(cd_blocks["No of Households"].sum())
            tot_workers = int(cd_blocks["Total Worker Population Person"].sum()) if "Total Worker Population Person" in cd_blocks.columns else 0
            main_workers = int(cd_blocks["Main Working Population Person"].sum()) if "Main Working Population Person" in cd_blocks.columns else 0
            non_workers = int(cd_blocks["Non Working Population Person"].sum()) if "Non Working Population Person" in cd_blocks.columns else 0

            urban_pop = int(cd_urbans["Total Population Person"].sum()) if not cd_urbans.empty else 0
            rural_pop = int(cd_rurals["Total Population Person"].sum()) if not cd_rurals.empty else 0

            literacy_rate = round((tot_lit / tot_pop) * 100.0, 2) if tot_pop > 0 else 0.0
            sex_ratio = round((tot_female / tot_male) * 1000.0, 2) if tot_male > 0 else 0.0
            female_ratio = round((tot_female / tot_pop), 4) if tot_pop > 0 else 0.0
            urban_pct = round((urban_pop / tot_pop) * 100.0, 2) if tot_pop > 0 else 0.0
            work_part_rate = round((tot_workers / tot_pop) * 100.0, 2) if tot_pop > 0 else 0.0

            return {
                "district": district,
                "total_population": tot_pop,
                "male_population": tot_male,
                "female_population": tot_female,
                "literate_population": tot_lit,
                "households": tot_hh,
                "total_workers": tot_workers,
                "main_workers": main_workers,
                "non_workers": non_workers,
                "urban_population": urban_pop,
                "rural_population": rural_pop,
                "literacy_rate": literacy_rate,
                "sex_ratio": sex_ratio,
                "female_ratio": female_ratio,
                "urban_pct": urban_pct,
                "work_participation_rate": work_part_rate,
            }
        except Exception as exc:
            logger.error(f"Error processing census file {file_path.name}: {exc}")
            return None

    def process(self) -> pd.DataFrame:
        """
        Execute full census processing workflow.

        Returns
        -------
        pd.DataFrame
            Merged and cleaned district profiles dataframe.
        """
        start_time = time.time()
        logger.info("START: Census Processing")

        files = list_csv_files(self.census_dir)
        if not files:
            raise ProcessingError(f"No census CSV files found in {self.census_dir}")

        logger.info(f"Loaded {len(files)} raw census files for merging.")

        records = []
        for file_path in files:
            res = self.process_single_census_file(file_path)
            if res:
                records.append(res)

        df = pd.DataFrame(records)
        raw_rows = len(df)

        # Remove duplicate district rows if any
        initial_dups = int(df.duplicated(subset=["district"]).sum())
        df = df.drop_duplicates(subset=["district"], keep="first").reset_index(drop=True)
        cleaned_rows = len(df)

        # Fill missing values if any
        df = df.fillna(NUMERIC_FILL_VALUE)

        # Clean string whitespace
        df["district"] = df["district"].astype(str).str.strip()

        save_csv(df, DISTRICT_PROFILE_FILE)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"END: Census Processing | Rows Loaded: {raw_rows} | Rows Cleaned: {cleaned_rows} | "
            f"Duplicates Removed: {initial_dups} | File Generated: {DISTRICT_PROFILE_FILE.name} | Execution Time: {elapsed}s"
        )

        return df


def main() -> None:
    """Run census processor independently."""
    processor = CensusProcessor()
    df = processor.process()
    print(f"\nSuccessfully generated district profiles with {len(df)} rows.")
    print(f"Saved to: {DISTRICT_PROFILE_FILE}")


if __name__ == "__main__":
    main()
