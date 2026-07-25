"""
Feature engineering module for Drishti ETL Pipeline.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from backend.common.logger import get_logger
from backend.common.helpers import (
    load_csv,
    save_csv,
    remove_duplicate_rows,
)
from backend.common.exceptions import FeatureEngineeringError
from backend.etl.config import (
    DISTRICT_PROFILE_FILE,
    CRIME_STATISTICS_FILE,
    FEATURE_STORE_FILE,
    NUMERIC_FILL_VALUE,
)

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Combines demographic profiles and crime statistics to engineer
    ML features, normalized indices, and risk scores.
    """

    def __init__(
        self,
        district_file: Path = DISTRICT_PROFILE_FILE,
        crime_file: Path = CRIME_STATISTICS_FILE,
        output_file: Path = FEATURE_STORE_FILE,
    ) -> None:
        """
        Initialize FeatureEngineer with file paths.

        Parameters
        ----------
        district_file : Path
            Path to processed district_profiles.csv.
        crime_file : Path
            Path to processed crime_statistics.csv.
        output_file : Path
            Path to output feature_store.csv.
        """
        self.district_file = district_file
        self.crime_file = crime_file
        self.output_file = output_file

    def generate_features(self) -> pd.DataFrame:
        """
        Engineers ML features from demographic and crime data.

        Returns
        -------
        pd.DataFrame
            Engineered feature store dataframe.
        """
        start_time = time.time()
        logger.info("START: Feature Engineering")

        if not self.district_file.exists():
            raise FeatureEngineeringError(f"Missing required district profiles file: {self.district_file}")
        if not self.crime_file.exists():
            raise FeatureEngineeringError(f"Missing required crime statistics file: {self.crime_file}")

        df_district = load_csv(self.district_file)
        df_crime = load_csv(self.crime_file)

        raw_rows = len(df_district)
        logger.info(f"Loaded district profiles ({len(df_district)} rows) & crime stats ({len(df_crime)} rows).")

        # Aggregate crime statistics by district
        crime_agg_records = []

        for district, group in df_crime.groupby("district"):
            group_sorted = group.sort_values("year")
            earliest_crimes = group_sorted.iloc[0]["total_crimes"]
            latest_crimes = group_sorted.iloc[-1]["total_crimes"]

            if earliest_crimes > 0:
                crime_growth = round(((latest_crimes - earliest_crimes) / earliest_crimes) * 100.0, 2)
            else:
                crime_growth = 0.0

            mean_crimes = group["total_crimes"].mean()
            max_crimes = group["total_crimes"].max()

            violent_crimes = (
                group["murder"].sum()
                + group["attempt_to_murder"].sum()
                + group["rape"].sum()
                + group["kidnapping"].sum()
                + group["dacoity"].sum()
                + group["robbery"].sum()
            )

            property_crimes = (
                group["burglary"].sum()
                + group["theft"].sum()
                + group["cheating"].sum()
            )

            crime_agg_records.append({
                "district": district,
                "total_crimes_mean": round(mean_crimes, 2),
                "total_crimes_latest": latest_crimes,
                "total_crimes_max": max_crimes,
                "violent_crimes_sum": violent_crimes,
                "property_crimes_sum": property_crimes,
                "crime_growth": crime_growth,
            })

        df_crime_agg = pd.DataFrame(crime_agg_records)

        # Merge district profiles and aggregated crime data
        df_features = pd.merge(df_district, df_crime_agg, on="district", how="inner")

        if df_features.empty:
            logger.warning("Inner merge yielded 0 rows. Performing left join on district profiles.")
            df_features = pd.merge(df_district, df_crime_agg, on="district", how="left").fillna(0.0)

        # ----------------------------------------------------
        # Engineer ML Features
        # ----------------------------------------------------

        # 1. Population Density (Estimated area proxy or households)
        # Using households / 0.04 as area estimate
        df_features["population_density"] = (
            df_features["total_population"] / (df_features["households"] * 0.04 + 1.0)
        ).round(2)

        # 2. Crime Rate per 100k
        df_features["crime_rate_per_100k"] = (
            (df_features["total_crimes_latest"] / df_features["total_population"]) * 100000.0
        ).round(2)

        # 3. Literacy %
        df_features["literacy_pct"] = df_features["literacy_rate"].round(2)

        # 4. Female Ratio (Female pop / Total pop)
        df_features["female_ratio"] = df_features["female_ratio"].round(4)

        # 5. Urban %
        df_features["urban_pct"] = df_features["urban_pct"].round(2)

        # 6. Population Growth (Baseline demographic estimate)
        df_features["population_growth"] = 1.25  # 1.25% baseline per annum

        # 7. Crime Growth %
        df_features["crime_growth_rate"] = df_features["crime_growth"].round(2)

        # 8. Economic Indicator: Work Participation Rate %
        df_features["economic_activity_index"] = df_features["work_participation_rate"].round(2)

        # 9. Crime Severity Score
        # Weighted combination of violent crimes vs property crimes normalized per 10k pop
        df_features["crime_severity_score"] = (
            (df_features["violent_crimes_sum"] * 4.0 + df_features["property_crimes_sum"] * 1.5 + df_features["total_crimes_max"])
            / (df_features["total_population"] / 10000.0)
        ).round(2)

        # 10. Normalized Crime Index [0.0 - 1.0]
        max_rate = df_features["crime_rate_per_100k"].max()
        min_rate = df_features["crime_rate_per_100k"].min()
        rate_diff = max_rate - min_rate if max_rate > min_rate else 1.0
        df_features["normalized_crime_index"] = (
            (df_features["crime_rate_per_100k"] - min_rate) / rate_diff
        ).round(4)

        # 11. Hotspot Score [0 - 100]
        max_sev = df_features["crime_severity_score"].max()
        max_sev = max_sev if max_sev > 0 else 1.0
        norm_sev = df_features["crime_severity_score"] / max_sev
        norm_lit_inv = 1.0 - (df_features["literacy_pct"] / 100.0)

        df_features["hotspot_score"] = (
            (0.45 * df_features["normalized_crime_index"] + 0.35 * norm_sev + 0.20 * norm_lit_inv) * 100.0
        ).round(2)

        # Select key output features
        output_cols = [
            "district",
            "total_population",
            "population_density",
            "crime_rate_per_100k",
            "literacy_pct",
            "female_ratio",
            "urban_pct",
            "population_growth",
            "crime_growth_rate",
            "economic_activity_index",
            "crime_severity_score",
            "normalized_crime_index",
            "hotspot_score",
        ]

        df_final = df_features[output_cols].copy()
        df_final = remove_duplicate_rows(df_final)

        save_csv(df_final, self.output_file)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"END: Feature Engineering | Input Rows: {raw_rows} | Output Rows: {len(df_final)} | "
            f"File Generated: {self.output_file.name} | Execution Time: {elapsed}s"
        )

        return df_final


def main() -> None:
    """Run feature engineering independently."""
    engineer = FeatureEngineer()
    df = engineer.generate_features()
    print(f"\nSuccessfully generated feature store with {len(df)} rows and {df.shape[1]} features.")
    print(f"Saved to: {FEATURE_STORE_FILE}")


if __name__ == "__main__":
    main()
