"""
Dataset validation module for Drishti ETL Pipeline.
"""

from pathlib import Path
from typing import Any
import os
import pandas as pd

from backend.common.logger import get_logger
from backend.common.helpers import (
    load_csv,
    clean_column_names,
    dataframe_summary,
    ensure_directories,
    list_csv_files,
    standardize_district_name,
)
from backend.common.exceptions import (
    ValidationError,
    EmptyDatasetError,
    DatasetNotFoundError,
)
from backend.etl.config import (
    CENSUS_DATASET_DIR,
    CRIME_DATASET_DIR,
    VALIDATION_REPORT_FILE,
    REPORT_LINE,
)

logger = get_logger(__name__)


class DatasetValidator:
    """
    Validates input raw datasets (Census and Crime files) for schema integrity,
    missing values, duplicate records, empty files, and district name consistency.
    """

    def __init__(self) -> None:
        """Initialize validator and ensure output directories exist."""
        ensure_directories()

    def validate_file_existence(self, file_path: Path) -> bool:
        """
        Verify if target file exists and is non-empty.

        Parameters
        ----------
        file_path : Path
            Path to file.

        Returns
        -------
        bool
            True if file exists and has size > 0.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty (0 bytes): {file_path}")
            return False
        return True

    def validate_dataset(self, file_path: Path) -> dict[str, Any]:
        """
        Validate single CSV file for statistics, missing values, duplicates, and invalid numbers.

        Parameters
        ----------
        file_path : Path
            Path to raw CSV file.

        Returns
        -------
        dict[str, Any]
            Validation report dictionary for the dataset.
        """
        file_name = file_path.name
        logger.info(f"Validating dataset file: {file_name}")

        if not self.validate_file_existence(file_path):
            return {
                "file_name": file_name,
                "status": "FAILED",
                "error": "File missing or zero bytes",
            }

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            logger.error(f"Failed to parse CSV format for {file_name}: {exc}")
            return {
                "file_name": file_name,
                "status": "FAILED",
                "error": f"CSV parse error: {str(exc)}",
            }

        if df.empty:
            logger.warning(f"Empty dataset encountered: {file_name}")
            return {
                "file_name": file_name,
                "status": "FAILED",
                "error": "Empty dataset",
            }

        rows, cols = df.shape
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        # Check for invalid numeric values (NaN, Inf in numeric columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        invalid_numeric_count = 0
        if len(numeric_cols) > 0:
            invalid_numeric_count = int(df[numeric_cols].isin([float("inf"), float("-inf")]).sum().sum())

        logger.info(
            f"File: {file_name} | Rows: {rows} | Cols: {cols} | "
            f"Missing: {missing_count} | Duplicates: {duplicate_count} | Invalid Numeric: {invalid_numeric_count}"
        )

        return {
            "file_name": file_name,
            "status": "PASSED",
            "rows": rows,
            "columns": cols,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
            "invalid_numeric_values": invalid_numeric_count,
        }

    def validate_district_consistency(
        self, census_dir: Path, crime_file: Path
    ) -> dict[str, Any]:
        """
        Check consistency between Census district names and Crime district names.

        Parameters
        ----------
        census_dir : Path
            Directory containing raw census CSV files.
        crime_file : Path
            Path to crime dataset CSV file.

        Returns
        -------
        dict[str, Any]
            Consistency report.
        """
        logger.info("Checking district name consistency between Census and Crime datasets...")
        census_files = list_csv_files(census_dir)
        census_districts = set()

        for cfile in census_files:
            try:
                cdf = pd.read_csv(cfile, nrows=50, low_memory=False)
                if "District_Name" in cdf.columns:
                    dists = cdf["District_Name"].dropna().unique()
                    for d in dists:
                        census_districts.add(standardize_district_name(str(d)))
            except Exception as exc:
                logger.warning(f"Could not extract district names from {cfile.name}: {exc}")

        crime_districts = set()
        if crime_file.exists():
            try:
                cr_df = pd.read_csv(crime_file, low_memory=False)
                if "DISTRICT" in cr_df.columns:
                    raw_dists = cr_df["DISTRICT"].dropna().unique()
                    for d in raw_dists:
                        d_str = str(d).strip()
                        if not d_str.isdigit() and d_str not in ["year", "ZZ TOTAL"]:
                            crime_districts.add(standardize_district_name(d_str))
            except Exception as exc:
                logger.warning(f"Could not extract crime district names: {exc}")

        matched_districts = census_districts.intersection(crime_districts)
        unmatched_census = census_districts - crime_districts
        unmatched_crime = crime_districts - census_districts

        report = {
            "total_census_districts": len(census_districts),
            "total_crime_districts": len(crime_districts),
            "matched_districts_count": len(matched_districts),
            "unmatched_census": sorted(list(unmatched_census)),
            "unmatched_crime": sorted(list(unmatched_crime)),
        }

        logger.info(
            f"District Consistency Check: Matched {len(matched_districts)} districts "
            f"(Census: {len(census_districts)}, Crime: {len(crime_districts)})"
        )
        return report

    def run_full_validation(self) -> dict[str, Any]:
        """
        Execute full validation pipeline over Census & Crime raw data and save report.

        Returns
        -------
        dict[str, Any]
            Full validation execution report.
        """
        logger.info("START: Full Dataset Validation")
        census_files = list_csv_files(CENSUS_DATASET_DIR)
        crime_files = list_csv_files(CRIME_DATASET_DIR)

        census_reports = [self.validate_dataset(f) for f in census_files]
        crime_reports = [self.validate_dataset(f) for f in crime_files]

        crime_main_file = CRIME_DATASET_DIR / "Karnataka Crime Data 2016-22.csv"
        consistency_report = self.validate_district_consistency(
            CENSUS_DATASET_DIR, crime_main_file
        )

        overall_status = "PASSED"
        for r in census_reports + crime_reports:
            if r.get("status") == "FAILED":
                overall_status = "FAILED WITH WARNINGS"

        full_report = {
            "overall_status": overall_status,
            "census_files_validated": len(census_files),
            "crime_files_validated": len(crime_files),
            "census_details": census_reports,
            "crime_details": crime_reports,
            "district_consistency": consistency_report,
        }

        self.save_validation_report(full_report)
        logger.info("END: Full Dataset Validation")
        return full_report

    def save_validation_report(self, report: dict[str, Any]) -> None:
        """
        Save validation report as formatted text file.

        Parameters
        ----------
        report : dict[str, Any]
            Validation statistics.
        """
        lines = [
            REPORT_LINE,
            "DRISHTI PREDICTIVE COMMAND CONSOLE - DATASET VALIDATION REPORT",
            REPORT_LINE,
            f"Overall Status          : {report['overall_status']}",
            f"Census Files Validated  : {report['census_files_validated']}",
            f"Crime Files Validated   : {report['crime_files_validated']}",
            "",
            "DISTRICT CONSISTENCY REPORT",
            "-" * 40,
            f"Unique Census Districts : {report['district_consistency']['total_census_districts']}",
            f"Unique Crime Districts  : {report['district_consistency']['total_crime_districts']}",
            f"Matched Districts       : {report['district_consistency']['matched_districts_count']}",
            f"Unmatched Census        : {', '.join(report['district_consistency']['unmatched_census']) or 'None'}",
            f"Unmatched Crime         : {', '.join(report['district_consistency']['unmatched_crime']) or 'None'}",
            "",
            "CENSUS FILES SUMMARY",
            "-" * 40,
        ]

        for item in report["census_details"]:
            if item.get("status") == "PASSED":
                lines.append(
                    f"- {item['file_name']}: {item['rows']} rows, {item['columns']} cols, "
                    f"Missing: {item['missing_values']}, Dups: {item['duplicate_rows']}"
                )
            else:
                lines.append(f"- {item['file_name']}: FAILED ({item.get('error')})")

        lines.extend([
            "",
            "CRIME FILES SUMMARY",
            "-" * 40,
        ])

        for item in report["crime_details"]:
            if item.get("status") == "PASSED":
                lines.append(
                    f"- {item['file_name']}: {item['rows']} rows, {item['columns']} cols, "
                    f"Missing: {item['missing_values']}, Dups: {item['duplicate_rows']}"
                )
            else:
                lines.append(f"- {item['file_name']}: FAILED ({item.get('error')})")

        lines.append(REPORT_LINE)

        text_content = "\n".join(lines)
        VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(text_content)

        logger.info(f"Saved validation report to {VALIDATION_REPORT_FILE}")


def main() -> None:
    """Run validator script independently."""
    validator = DatasetValidator()
    report = validator.run_full_validation()
    print(f"\nValidation completed with status: {report['overall_status']}")
    print(f"Report saved to: {VALIDATION_REPORT_FILE}")


if __name__ == "__main__":
    main()