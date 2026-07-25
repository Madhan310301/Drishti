"""
Script to generate and execute SQL seed queries for Supabase project.
"""

from pathlib import Path
import pandas as pd
from backend.common.helpers import load_csv
from backend.common.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/output")


def generate_seed_sql() -> str:
    """Generate SQL statements for Supabase database."""
    sqls = []

    # 1. District Profiles
    df_prof = load_csv(PROCESSED_DIR / "district_profiles.csv")
    for _, r in df_prof.iterrows():
        sqls.append(
            f"INSERT INTO district_profiles (district, total_population, male_population, female_population, "
            f"literate_population, households, total_workers, main_workers, non_workers, urban_population, "
            f"rural_population, literacy_rate, sex_ratio, female_ratio, urban_pct, work_participation_rate) "
            f"VALUES ('{r['district']}', {int(r['total_population'])}, {int(r['male_population'])}, {int(r['female_population'])}, "
            f"{int(r['literate_population'])}, {int(r['households'])}, {int(r['total_workers'])}, {int(r['main_workers'])}, "
            f"{int(r['non_workers'])}, {int(r['urban_population'])}, {int(r['rural_population'])}, {r['literacy_rate']}, "
            f"{r['sex_ratio']}, {r['female_ratio']}, {r['urban_pct']}, {r['work_participation_rate']}) "
            f"ON CONFLICT (district) DO NOTHING;"
        )

    # 2. Crime Statistics
    df_crime = load_csv(PROCESSED_DIR / "crime_statistics.csv")
    for _, r in df_crime.iterrows():
        sqls.append(
            f"INSERT INTO crime_statistics (district, year, murder, attempt_to_murder, rape, kidnapping, dacoity, "
            f"robbery, burglary, theft, riots, cheating, dowry_deaths, total_crimes, crime_rate_per_100k) "
            f"VALUES ('{r['district']}', {int(r['year'])}, {r['murder']}, {r['attempt_to_murder']}, {r['rape']}, "
            f"{r['kidnapping']}, {r['dacoity']}, {r['robbery']}, {r['burglary']}, {r['theft']}, {r['riots']}, "
            f"{r['cheating']}, {r['dowry_deaths']}, {r['total_crimes']}, {r['crime_rate_per_100k']}) "
            f"ON CONFLICT (district, year) DO NOTHING;"
        )

    # 3. Feature Store
    df_feat = load_csv(OUTPUT_DIR / "feature_store.csv")
    for _, r in df_feat.iterrows():
        sqls.append(
            f"INSERT INTO feature_store (district, total_population, population_density, crime_rate_per_100k, "
            f"literacy_pct, female_ratio, urban_pct, population_growth, crime_growth_rate, economic_activity_index, "
            f"crime_severity_score, normalized_crime_index, hotspot_score) "
            f"VALUES ('{r['district']}', {int(r['total_population'])}, {r['population_density']}, {r['crime_rate_per_100k']}, "
            f"{r['literacy_pct']}, {r['female_ratio']}, {r['urban_pct']}, {r['population_growth']}, {r['crime_growth_rate']}, "
            f"{r['economic_activity_index']}, {r['crime_severity_score']}, {r['normalized_crime_index']}, {r['hotspot_score']}) "
            f"ON CONFLICT (district) DO NOTHING;"
        )

    # 4. Prediction Results
    df_preds = load_csv(OUTPUT_DIR / "hotspot_predictions.csv")
    for _, r in df_preds.iterrows():
        sqls.append(
            f"INSERT INTO prediction_results (district, cluster_id, hotspot_score, risk_score, crime_forecast, "
            f"current_crime_rate_per_100k, anomaly_score) "
            f"VALUES ('{r['district']}', {int(r['cluster_id'])}, {r['hotspot_score']}, {r['risk_score']}, {r['crime_forecast']}, "
            f"{r['current_crime_rate_per_100k']}, {r['anomaly_score']});"
        )

    return "\n".join(sqls)


if __name__ == "__main__":
    sql_text = generate_seed_sql()
    out_file = OUTPUT_DIR / "supabase_seed.sql"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(sql_text)
    print(f"Generated Supabase seed file with {len(sql_text.splitlines())} SQL statements at {out_file}")
