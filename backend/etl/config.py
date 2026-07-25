"""
Central ETL configuration for Drishti.
"""

from pathlib import Path

from backend.common.constants import (
    PROJECT_ROOT,
    RAW_CENSUS_DIR,
    RAW_CRIME_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DATA_DIR,
    DISTRICT_PROFILE_FILE,
    CRIME_STATS_FILE,
    FEATURE_STORE_FILE,
    HOTSPOT_PREDICTIONS_FILE,
    ANOMALY_SCORES_FILE,
    ANALYTICS_SUMMARY_FILE,
    SHAP_EXPLANATIONS_FILE,
    VALIDATION_REPORT_FILE,
    OFFENDER_NODES_FILE,
    OFFENDER_EDGES_FILE,
    GANG_NAMES,
    RELATIONSHIP_TYPES,
)

# ==============================================================================
# Directories
# ==============================================================================

CENSUS_DATASET_DIR = RAW_CENSUS_DIR
CRIME_DATASET_DIR = RAW_CRIME_DIR

PROCESSED_DIR = PROCESSED_DATA_DIR
OUTPUT_DIR = OUTPUT_DATA_DIR

# ==============================================================================
# Output Files
# ==============================================================================

DISTRICT_PROFILE_FILE = DISTRICT_PROFILE_FILE
CRIME_STATISTICS_FILE = CRIME_STATS_FILE
FEATURE_STORE_FILE = FEATURE_STORE_FILE
HOTSPOT_PREDICTIONS_FILE = HOTSPOT_PREDICTIONS_FILE
ANOMALY_SCORES_FILE = ANOMALY_SCORES_FILE
ANALYTICS_SUMMARY_FILE = ANALYTICS_SUMMARY_FILE
SHAP_EXPLANATIONS_FILE = SHAP_EXPLANATIONS_FILE
VALIDATION_REPORT_FILE = VALIDATION_REPORT_FILE
OFFENDER_NODES_FILE = OFFENDER_NODES_FILE
OFFENDER_EDGES_FILE = OFFENDER_EDGES_FILE

# ==============================================================================
# Offender Network Generation Settings
# ==============================================================================

OFFENDER_GANG_NAMES = GANG_NAMES
OFFENDER_RELATIONSHIP_TYPES = RELATIONSHIP_TYPES
OFFENDER_SUSPECT_COUNT = 45
OFFENDER_EDGE_COUNT = 70

# ==============================================================================
# Required Columns
# ==============================================================================

REQUIRED_CENSUS_COLUMNS = [
    "district_name",
]

REQUIRED_CRIME_COLUMNS = [
    "district",
]

# ==============================================================================
# Missing Value Rules
# ==============================================================================

NUMERIC_FILL_VALUE = 0.0
TEXT_FILL_VALUE = "Unknown"

# ==============================================================================
# Encoding & CSV Reading
# ==============================================================================

CSV_ENCODING = "utf-8"

READ_CSV_OPTIONS = {
    "encoding": CSV_ENCODING,
    "low_memory": False,
}

# ==============================================================================
# ML Defaults
# ==============================================================================

RANDOM_STATE = 42

# ==============================================================================
# Report Settings
# ==============================================================================

REPORT_LINE = "=" * 80