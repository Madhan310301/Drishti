"""
Master ETL Pipeline Orchestrator for Drishti Predictive Command Console.
"""

from pathlib import Path
import time
import sys

from backend.common.logger import get_logger
from backend.common.helpers import print_header, folder_size
from backend.etl.validator import DatasetValidator
from backend.etl.census_processor import CensusProcessor
from backend.etl.crime_processor import CrimeProcessor
from backend.etl.feature_engineering import FeatureEngineer
from backend.etl.offender_network import OffenderNetworkGenerator
from backend.ml.pipeline import MLPipeline
from backend.database.seed import seed_database
from backend.etl.config import (
    DISTRICT_PROFILE_FILE,
    CRIME_STATISTICS_FILE,
    FEATURE_STORE_FILE,
    HOTSPOT_PREDICTIONS_FILE,
    ANOMALY_SCORES_FILE,
    ANALYTICS_SUMMARY_FILE,
    VALIDATION_REPORT_FILE,
    OFFENDER_NODES_FILE,
    OFFENDER_EDGES_FILE,
    PROCESSED_DIR,
    OUTPUT_DIR,
)

logger = get_logger(__name__)


def run_master_etl() -> dict[str, float | str]:
    """
    Executes end-to-end Master ETL Pipeline:
    Validation -> Census Processing -> Crime Processing -> Feature Engineering -> ML Training -> DB Seeding.

    Returns
    -------
    dict[str, float | str]
        Execution metrics summary.
    """
    pipeline_start = time.time()
    logger.info("==========================================================================")
    logger.info("START: Master ETL Pipeline Execution - Drishti Predictive Command Console")
    logger.info("==========================================================================")

    # ----------------------------------------------------
    # Step 1: Validation
    # ----------------------------------------------------
    logger.info("--- Step 1/6: Running Dataset Validation ---")
    validator = DatasetValidator()
    val_report = validator.run_full_validation()
    logger.info(f"Validation Status: {val_report['overall_status']}")

    # ----------------------------------------------------
    # Step 2: Census Processor
    # ----------------------------------------------------
    logger.info("--- Step 2/6: Processing Census Data ---")
    census_processor = CensusProcessor()
    df_census = census_processor.process()
    logger.info(f"Census Profiles Generated: {len(df_census)} rows.")

    # ----------------------------------------------------
    # Step 3: Crime Processor
    # ----------------------------------------------------
    logger.info("--- Step 3/6: Processing Crime Data ---")
    crime_processor = CrimeProcessor()
    df_crime = crime_processor.process()
    logger.info(f"Crime Statistics Generated: {len(df_crime)} records across {df_crime['district'].nunique()} districts.")

    # ----------------------------------------------------
    # Step 4: Feature Engineering
    # ----------------------------------------------------
    logger.info("--- Step 4/6: Feature Engineering ---")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.generate_features()
    logger.info(f"Feature Store Generated: {len(df_features)} rows x {df_features.shape[1]} features.")

    # ----------------------------------------------------
    # Step 5: Offender Network Generation
    # ----------------------------------------------------
    logger.info("--- Step 5/6: Generating Offender Network ---")
    offender_generator = OffenderNetworkGenerator()
    offender_summary = offender_generator.generate()
    logger.info(
        f"Offender Network Generated: {offender_summary['suspect_count']} suspects, "
        f"{offender_summary['edge_count']} relationships."
    )

    # ----------------------------------------------------
    # Step 6: Machine Learning & Database Seeding
    # ----------------------------------------------------
    logger.info("--- Step 6/6: Running ML Models & Database Seeding ---")
    ml_pipeline = MLPipeline()
    ml_summary = ml_pipeline.run_pipeline()

    seed_database()

    total_elapsed = round(time.time() - pipeline_start, 3)

    metrics = {
        "status": "SUCCESS",
        "total_census_districts": len(df_census),
        "total_crime_records": len(df_crime),
        "feature_store_rows": len(df_features),
        "offender_suspect_count": offender_summary["suspect_count"],
        "offender_edge_count": offender_summary["edge_count"],
        "anomalies_detected": ml_summary["total_anomalies_detected"],
        "top_risk_district": ml_summary["highest_crime_risk_district"],
        "processed_folder_size_mb": folder_size(PROCESSED_DIR),
        "output_folder_size_mb": folder_size(OUTPUT_DIR),
        "total_execution_time_seconds": total_elapsed,
    }

    logger.info("==========================================================================")
    logger.info(f"END: Master ETL Pipeline Execution Completed in {total_elapsed}s")
    logger.info("==========================================================================")

    return metrics


def print_completion_report(metrics: dict[str, float | str]) -> None:
    """
    Print pretty completion report to console.

    Parameters
    ----------
    metrics : dict[str, float | str]
        Pipeline execution metrics.
    """
    print_header("DRISHTI PREDICTIVE COMMAND CONSOLE - ETL PIPELINE COMPLETION REPORT")
    print(f"Pipeline Status            : {metrics['status']}")
    print(f"Total Execution Time       : {metrics['total_execution_time_seconds']} seconds")
    print(f"District Demographic Rows  : {metrics['total_census_districts']}")
    print(f"Yearly Crime Stat Records  : {metrics['total_crime_records']}")
    print(f"Feature Store Rows         : {metrics['feature_store_rows']}")
    print(f"Offender Suspects Generated: {metrics['offender_suspect_count']}")
    print(f"Offender Links Generated   : {metrics['offender_edge_count']}")
    print(f"Anomalies Identified       : {metrics['anomalies_detected']}")
    print(f"Highest Risk District      : {metrics['top_risk_district']}")
    print("\nGenerated Output Files:")
    print(f"  [OK] {DISTRICT_PROFILE_FILE}")
    print(f"  [OK] {CRIME_STATISTICS_FILE}")
    print(f"  [OK] {FEATURE_STORE_FILE}")
    print(f"  [OK] {OFFENDER_NODES_FILE}")
    print(f"  [OK] {OFFENDER_EDGES_FILE}")
    print(f"  [OK] {HOTSPOT_PREDICTIONS_FILE}")
    print(f"  [OK] {ANOMALY_SCORES_FILE}")
    print(f"  [OK] {ANALYTICS_SUMMARY_FILE}")
    print(f"  [OK] {VALIDATION_REPORT_FILE}")
    print("\n" + "=" * 70 + "\n")


def main() -> None:
    """Entry point for master ETL build pipeline."""
    try:
        metrics = run_master_etl()
        print_completion_report(metrics)
    except Exception as exc:
        logger.error(f"Master ETL Pipeline Failed: {exc}", exc_info=True)
        print(f"\n[ERROR] Master ETL Pipeline Execution Failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
