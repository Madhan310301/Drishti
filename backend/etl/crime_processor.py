"""
Crime data processor module for Drishti ETL Pipeline.
"""

from pathlib import Path
from typing import Any
import time
import pandas as pd

from backend.common.logger import get_logger
from backend.common.helpers import (
    load_csv,
    save_csv,
    standardize_district_name,
    remove_duplicate_rows,
)
from backend.common.exceptions import ProcessingError
from backend.etl.config import (
    CRIME_DATASET_DIR,
    CRIME_STATISTICS_FILE,
    DISTRICT_PROFILE_FILE,
    NUMERIC_FILL_VALUE,
)

logger = get_logger(__name__)


class CrimeProcessor:
    """
    Processor to load, normalize, and aggregate the raw multi-year crime dataset
    into standard yearly and district crime statistics.
    """

    def __init__(self, crime_dir: Path = CRIME_DATASET_DIR) -> None:
        """
        Initialize CrimeProcessor.

        Parameters
        ----------
        crime_dir : Path
            Directory containing raw crime data CSV.
        """
        self.crime_dir = crime_dir
        self.crime_file = crime_dir / "Karnataka Crime Data 2016-22.csv"

    def _parse_raw_crime_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Parse non-standard multi-year raw crime CSV into structured records.

        Parameters
        ----------
        df : pd.DataFrame
            Raw CSV dataframe.

        Returns
        -------
        list[dict[str, Any]]
            Parsed crime records list.
        """
        records = []

        for idx, row in df.iterrows():
            col0 = str(row.iloc[0]).strip()
            col1 = str(row.iloc[1]).strip()

            # Format 1: 2013 data section (Col 1 is '2013')
            if col1 == "2013" and col0 not in ["nan", "DISTRICT", "ZZ TOTAL"]:
                district = standardize_district_name(col0)
                if district in ["Unknown", "Railways"]:
                    continue

                murder = float(row["MURDER"]) if pd.notna(row.get("MURDER")) else 0.0
                attempt_murder = float(row["ATTEMPT TO MURDER"]) if pd.notna(row.get("ATTEMPT TO MURDER")) else 0.0
                rape = float(row["RAPE"]) if pd.notna(row.get("RAPE")) else 0.0
                kidnapping = float(row["KIDNAPPING & ABDUCTION"]) if pd.notna(row.get("KIDNAPPING & ABDUCTION")) else 0.0
                dacoity = float(row["DACOITY"]) if pd.notna(row.get("DACOITY")) else 0.0
                robbery = float(row["ROBBERY"]) if pd.notna(row.get("ROBBERY")) else 0.0
                burglary = float(row["BURGLARY"]) if pd.notna(row.get("BURGLARY")) else 0.0
                theft = float(row["THEFT"]) if pd.notna(row.get("THEFT")) else 0.0
                riots = float(row["RIOTS"]) if pd.notna(row.get("RIOTS")) else 0.0
                cheating = float(row["CHEATING"]) if pd.notna(row.get("CHEATING")) else 0.0
                dowry = float(row["DOWRY DEATHS"]) if pd.notna(row.get("DOWRY DEATHS")) else 0.0
                total_ipc = float(row["TOTAL IPC CRIMES"]) if pd.notna(row.get("TOTAL IPC CRIMES")) else 0.0

                records.append({
                    "district": district,
                    "year": 2013,
                    "murder": murder,
                    "attempt_to_murder": attempt_murder,
                    "rape": rape,
                    "kidnapping": kidnapping,
                    "dacoity": dacoity,
                    "robbery": robbery,
                    "burglary": burglary,
                    "theft": theft,
                    "riots": riots,
                    "cheating": cheating,
                    "dowry_deaths": dowry,
                    "total_crimes": total_ipc if total_ipc > 0 else (murder + attempt_murder + rape + kidnapping + robbery + theft),
                })

            # Format 2: 2016-2022 data section (Col 1 is 'Karnataka')
            elif col1 == "Karnataka":
                year_str = col0
                district_raw = str(row.iloc[5]).strip()
                if year_str.isdigit() and district_raw not in ["nan", "District Name", "Total", "District"]:
                    district = standardize_district_name(district_raw)
                    if district in ["Unknown", "Railways"]:
                        continue

                    year = int(year_str)

                    # Extract crime counts starting at col index 6
                    c6 = float(row.iloc[6]) if pd.notna(row.iloc[6]) and str(row.iloc[6]).replace(".", "", 1).isdigit() else 0.0
                    c7 = float(row.iloc[7]) if pd.notna(row.iloc[7]) and str(row.iloc[7]).replace(".", "", 1).isdigit() else 0.0
                    c8 = float(row.iloc[8]) if pd.notna(row.iloc[8]) and str(row.iloc[8]).replace(".", "", 1).isdigit() else 0.0
                    c9 = float(row.iloc[9]) if pd.notna(row.iloc[9]) and str(row.iloc[9]).replace(".", "", 1).isdigit() else 0.0
                    c10 = float(row.iloc[10]) if pd.notna(row.iloc[10]) and str(row.iloc[10]).replace(".", "", 1).isdigit() else 0.0
                    c11 = float(row.iloc[11]) if pd.notna(row.iloc[11]) and str(row.iloc[11]).replace(".", "", 1).isdigit() else 0.0

                    # Calculate total crimes across row
                    row_vals = []
                    for c_idx in range(6, len(row)):
                        val = row.iloc[c_idx]
                        if pd.notna(val) and str(val).replace(".", "", 1).isdigit():
                            row_vals.append(float(val))

                    tot_crimes = sum(row_vals) if row_vals else 0.0

                    records.append({
                        "district": district,
                        "year": year,
                        "murder": c6,
                        "attempt_to_murder": c7,
                        "rape": c8,
                        "kidnapping": c9,
                        "dacoity": c10,
                        "robbery": c11,
                        "burglary": 0.0,
                        "theft": 0.0,
                        "riots": 0.0,
                        "cheating": 0.0,
                        "dowry_deaths": 0.0,
                        "total_crimes": tot_crimes,
                    })

        return records

    def process(self) -> pd.DataFrame:
        """
        Execute full crime dataset processing pipeline.

        Returns
        -------
        pd.DataFrame
            Normalized crime statistics dataframe.
        """
        start_time = time.time()
        logger.info("START: Crime Processing")

        if not self.crime_file.exists():
            raise ProcessingError(f"Crime dataset file not found: {self.crime_file}")

        df_raw = pd.read_csv(self.crime_file, low_memory=False)
        raw_rows = len(df_raw)
        logger.info(f"Loaded raw crime CSV with {raw_rows} rows.")

        records = self._parse_raw_crime_dataframe(df_raw)
        df_parsed = pd.DataFrame(records)

        if df_parsed.empty:
            raise ProcessingError("Failed to extract valid crime records from dataset.")

        # Group by district & year to aggregate multi-jurisdiction records (e.g. City + District)
        agg_funcs = {
            "murder": "sum",
            "attempt_to_murder": "sum",
            "rape": "sum",
            "kidnapping": "sum",
            "dacoity": "sum",
            "robbery": "sum",
            "burglary": "sum",
            "theft": "sum",
            "riots": "sum",
            "cheating": "sum",
            "dowry_deaths": "sum",
            "total_crimes": "sum",
        }

        df_grouped = df_parsed.groupby(["district", "year"], as_index=False).agg(agg_funcs)

        initial_count = len(df_parsed)
        cleaned_rows = len(df_grouped)
        dups_removed = initial_count - cleaned_rows

        # Attach census total population to compute crime_rate_per_100k
        if DISTRICT_PROFILE_FILE.exists():
            try:
                df_census = pd.read_csv(DISTRICT_PROFILE_FILE)
                if "district" in df_census.columns and "total_population" in df_census.columns:
                    pop_map = dict(zip(df_census["district"], df_census["total_population"]))
                    df_grouped["population"] = df_grouped["district"].map(pop_map).fillna(1000000)
                    df_grouped["crime_rate_per_100k"] = (
                        (df_grouped["total_crimes"] / df_grouped["population"]) * 100000.0
                    ).round(2)
                else:
                    df_grouped["crime_rate_per_100k"] = 0.0
            except Exception as exc:
                logger.warning(f"Could not calculate crime rate per 100k: {exc}")
                df_grouped["crime_rate_per_100k"] = 0.0
        else:
            df_grouped["crime_rate_per_100k"] = 0.0

        save_csv(df_grouped, CRIME_STATISTICS_FILE)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"END: Crime Processing | Rows Loaded: {raw_rows} | Rows Cleaned: {cleaned_rows} | "
            f"Duplicates Removed: {dups_removed} | File Generated: {CRIME_STATISTICS_FILE.name} | Execution Time: {elapsed}s"
        )

        return df_grouped


def main() -> None:
    """Run crime processor independently."""
    processor = CrimeProcessor()
    df = processor.process()
    print(f"\nSuccessfully generated crime statistics with {len(df)} records across {df['district'].nunique()} districts.")
    print(f"Saved to: {CRIME_STATISTICS_FILE}")


if __name__ == "__main__":
    main()
