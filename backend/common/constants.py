"""
Project-wide constants for Drishti.

Do not hardcode values elsewhere in the project.
Import constants from this module.
"""

from pathlib import Path

# ------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"

RAW_CRIME_DIR = RAW_DATA_DIR / "crime"
RAW_CENSUS_DIR = RAW_DATA_DIR / "census"

# ------------------------------------------------------------------------------
# Output Files
# ------------------------------------------------------------------------------

DISTRICT_PROFILE_FILE = PROCESSED_DATA_DIR / "district_profiles.csv"
CRIME_STATS_FILE = PROCESSED_DATA_DIR / "crime_statistics.csv"
FEATURE_STORE_FILE = OUTPUT_DATA_DIR / "feature_store.csv"
HOTSPOT_PREDICTIONS_FILE = OUTPUT_DATA_DIR / "hotspot_predictions.csv"
ANOMALY_SCORES_FILE = PROCESSED_DATA_DIR / "anomaly_scores.csv"
HOTSPOT_CENTERS_FILE = PROCESSED_DATA_DIR / "hotspot_centers.csv"
ANALYTICS_SUMMARY_FILE = OUTPUT_DATA_DIR / "analytics_summary.json"
SHAP_EXPLANATIONS_FILE = OUTPUT_DATA_DIR / "shap_explanations.json"
PATROL_PLAN_FILE = OUTPUT_DATA_DIR / "patrol_plan.json"
NETWORK_GRAPH_HTML = OUTPUT_DATA_DIR / "offender_network.html"
HOTSPOT_MAP_HTML = OUTPUT_DATA_DIR / "hotspot_map.html"
VALIDATION_REPORT_FILE = OUTPUT_DATA_DIR / "validation_report.txt"
OFFENDER_NODES_FILE = PROCESSED_DATA_DIR / "offender_nodes.csv"
OFFENDER_EDGES_FILE = PROCESSED_DATA_DIR / "offender_edges.csv"

# ------------------------------------------------------------------------------
# Supported File Types
# ------------------------------------------------------------------------------

SUPPORTED_FILE_TYPES = {".csv"}

# ------------------------------------------------------------------------------
# ML Defaults
# ------------------------------------------------------------------------------

RANDOM_STATE = 42
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

LOG_FILE_NAME = "drishti.log"

# ------------------------------------------------------------------------------
# Dataset Constants
# ------------------------------------------------------------------------------

DISTRICT_COLUMN = "district"
YEAR_COLUMN = "year"
STATE_NAME = "Karnataka"

# ------------------------------------------------------------------------------
# Missing Value Handling
# ------------------------------------------------------------------------------

MISSING_NUMERIC_FILL = 0.0
MISSING_TEXT_FILL = "Unknown"

# ------------------------------------------------------------------------------
# Validation Rules
# ------------------------------------------------------------------------------

MIN_ROWS_REQUIRED = 1
MIN_COLUMNS_REQUIRED = 2

# ------------------------------------------------------------------------------
# Crime Categories
# ------------------------------------------------------------------------------

CRIME_CATEGORIES = [
    "Murder",
    "Attempt to Murder",
    "Culpable Homicide",
    "Rape",
    "Kidnapping & Abduction",
    "Dacoity",
    "Robbery",
    "Burglary",
    "Theft",
    "Riots",
    "Criminal Breach of Trust",
    "Cheating",
    "Counterfeiting",
    "Arson",
    "Hurt/Grievous Hurt",
    "Dowry Deaths",
    "Assault on Women",
    "Cruelty by Husband/Relatives",
    "Causing Death by Negligence",
    "Other IPC Crimes",
]

# ------------------------------------------------------------------------------
# Offender Network Constants
# ------------------------------------------------------------------------------

GANG_NAMES = [
    "Silk Board Syndicate",
    "Yeshwantpur Crew",
    "Northside Collective",
    "Harbor Line Gang",
    "Red Hill Outfit",
    "Station Road Network",
    "Independent",
]

RELATIONSHIP_TYPES = [
    "co_accused",
    "same_gang",
    "shared_mo",
    "family",
    "associate",
]

# ------------------------------------------------------------------------------
# Standardized District Mappings (Maps raw CSV spellings to standard names)
# ------------------------------------------------------------------------------

DISTRICT_MAPPING = {
    "BAGALKOT": "Bagalkot",
    "BAGALKOTE": "Bagalkot",
    "BANGALORE": "Bangalore",
    "BANGALORE COMMR.": "Bangalore",
    "BENGALURU CITY": "Bangalore",
    "BENGALURU DISTRICT": "Bangalore",
    "BANGALORE RURAL": "Bangalore Rural",
    "BENGALURU RURAL": "Bangalore Rural",
    "BELGAUM": "Belgaum",
    "BELAGAVI DISTRICT": "Belgaum",
    "BELAGAVI CITY": "Belgaum",
    "BELLARY": "Bellary",
    "BALLARI": "Bellary",
    "VIJAYANAGARA": "Bellary",
    "BIDAR": "Bidar",
    "BIJAPUR": "Bijapur",
    "VIJAYAPURA": "Bijapur",
    "CBPURA": "Chikkaballapura",
    "CHIKKABALLAPURA": "Chikkaballapura",
    "CHAMARAJNAGAR": "Chamarajanagar",
    "CHAMARAJANAGAR": "Chamarajanagar",
    "CHICKMAGALUR": "Chikmagalur",
    "CHIKKAMAGALURU": "Chikmagalur",
    "CHITRADURGA": "Chitradurga",
    "DAKSHIN KANNADA": "Dakshina Kannada",
    "DAKSHINA KANNADA": "Dakshina Kannada",
    "MANGALORE CITY": "Dakshina Kannada",
    "MANGALURU CITY": "Dakshina Kannada",
    "DAVANAGERE": "Davanagere",
    "DHARWAD": "Dharwad",
    "DHARWAD COMMR.": "Dharwad",
    "DHARWAD RURAL": "Dharwad",
    "HUBBALLI DHARWAD CITY": "Dharwad",
    "GADAG": "Gadag",
    "GULBARGA": "Gulbarga",
    "KALABURGI": "Gulbarga",
    "KALABURGI CITY": "Gulbarga",
    "HASSAN": "Hassan",
    "HAVERI": "Haveri",
    "K.G.F.": "Kolar",
    "KOLAR": "Kolar",
    "KODAGU": "Kodagu",
    "KOPPAL": "Koppal",
    "MANDYA": "Mandya",
    "MYSORE": "Mysore",
    "MYSORE COMMR.": "Mysore",
    "MYSORE RURAL": "Mysore",
    "MYSURU CITY": "Mysore",
    "MYSURU DISTRICT": "Mysore",
    "RAICHUR": "Raichur",
    "RAMANAGAR": "Ramanagara",
    "RAMANAGARA": "Ramanagara",
    "SHIMOGA": "Shimoga",
    "TUMKUR": "Tumkur",
    "TUMAKURU": "Tumkur",
    "UDUPI": "Udupi",
    "UTTAR KANNADA": "Uttara Kannada",
    "YADGIRI": "Yadgir",
    "YADGIR": "Yadgir",
    "RAILWAYS": "Railways",
    "K.RAILWAYS": "Railways",
}

# ------------------------------------------------------------------------------
# Directory Creation
# ------------------------------------------------------------------------------

DIRECTORIES = [
    LOG_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    RAW_CRIME_DIR,
    RAW_CENSUS_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DATA_DIR,
]