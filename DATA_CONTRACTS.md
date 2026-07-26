# DATA CONTRACTS

This file is the written agreement between all team members about exact file
names and column names, so Vijay's output matches Sai Ram's input, etc.

## Files Sai Ram Produces
- `data/raw/karnataka_socio_economic.csv`
  columns: district_name, literacy_rate, unemployment_rate, poverty_index,
  police_station_count, alcohol_outlet_density, population
- `data/raw/crime/karnataka_crime_2022.csv` (REAL KSP crime totals, 2022)
  columns: Districts, IPC Cases, SLL Cases, Total
- `data/processed/hotspot_centers.csv` (cluster centers with coordinates)
  columns: cluster_id, district, center_lat, center_lon, risk_score

## Files Vijay Produces
- `data/processed/hotspot_centers.csv` (cluster_id, center_lat, center_lon, risk_score)
- `data/processed/grid_with_anomalies.csv` (adds is_anomaly, anomaly_score)
- `data/output/shap_explanations.json`

## Functions Vijay Exposes
- `backend/ml/explainability.py` -> `explain_district(name)`, `explain_global()`

## Files Kalyan Produces
- `backend/ml/patrol_optimizer.py` -> `solve_patrol(hotspot_df, num_units, max_radius_km)`
  returns: dict with 'deployed', 'covered_pct', 'uncovered_count', 'total_hotspots'

## Files Jenifa Produces
- `app/app.py` (Streamlit dashboard) imports `solve_patrol` from `backend/ml/patrol_optimizer.py`

## Shared Column Names (DO NOT RENAME WITHOUT TEAM AGREEMENT)
- crime points: latitude, longitude, crime_type, severity, district
- hotspots: center_lat, center_lon, risk_score
- socio-economic: district_name (must match crime `district` values)
- patrol result: deployed, covered_pct, uncovered_count
