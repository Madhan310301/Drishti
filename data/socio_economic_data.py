"""
Generate Karnataka district socio-economic data (Sai Ram's Task 2).

The manual wants data/karnataka_socio_economic.csv with these columns:
    district_name, literacy_rate, unemployment_rate, poverty_index,
    police_station_count, alcohol_outlet_density, population

Exact numbers are not published cleanly per district, so we use realistic
approximate values (the manual explicitly allows this for a hackathon demo).
Bengaluru Urban is given higher literacy / more police stations, etc., to stay
realistic. District names match the crime data exactly.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT = RAW_DIR / "karnataka_socio_economic.csv"

# Values are realistic approximations (manual-allowed). Districts mirror the
# crime data district names.
DATA = [
    # district_name, literacy, unemployment, poverty, ps_count, alcohol_density, population
    ("Bengaluru Urban", 88.7, 4.2, 8.0, 108, 14.0, 10921000),
    ("Bengaluru Rural", 77.0, 4.5, 12.0, 22, 9.0, 1100000),
    ("Mysuru", 85.0, 4.0, 10.0, 28, 10.0, 3060000),
    ("Mandya", 74.0, 4.8, 14.0, 18, 8.0, 1800000),
    ("Hassan", 78.0, 4.3, 13.0, 20, 7.5, 1770000),
    ("Dakshina Kannada", 88.0, 3.5, 7.0, 30, 11.0, 2100000),
    ("Udupi", 86.0, 3.8, 6.0, 16, 9.0, 1180000),
    ("Chikmagalur", 79.0, 4.0, 11.0, 18, 8.0, 1140000),
    ("Tumakuru", 75.0, 4.6, 13.0, 22, 7.0, 2680000),
    ("Kolar", 73.0, 4.9, 15.0, 16, 6.5, 1540000),
    ("Chitradurga", 71.0, 5.0, 16.0, 14, 6.0, 1130000),
    ("Shivamogga", 80.0, 4.1, 11.0, 19, 8.0, 1750000),
    ("Davanagere", 76.0, 4.7, 14.0, 18, 7.5, 1950000),
    ("Ballari", 70.0, 5.2, 18.0, 20, 7.0, 2530000),
    ("Vijayanagara", 70.0, 5.3, 18.0, 12, 6.5, 1340000),
    ("Koppal", 68.0, 5.5, 20.0, 12, 6.0, 1390000),
    ("Raichur", 64.0, 5.8, 22.0, 14, 5.5, 1930000),
    ("Yadgir", 62.0, 6.0, 24.0, 10, 5.0, 1170000),
    ("Kalaburagi", 67.0, 5.4, 21.0, 22, 6.0, 2570000),
    ("Bidar", 69.0, 5.1, 19.0, 16, 5.5, 1700000),
    ("Vijayapura", 71.0, 4.9, 17.0, 20, 6.5, 2180000),
    ("Bagalkot", 73.0, 4.6, 15.0, 16, 6.0, 1890000),
    ("Dharwad", 81.0, 4.0, 10.0, 20, 8.0, 1850000),
    ("Gadag", 75.0, 4.5, 13.0, 12, 6.5, 1070000),
    ("Haveri", 77.0, 4.2, 12.0, 13, 7.0, 1600000),
    ("Belagavi", 80.0, 4.1, 11.0, 35, 8.5, 4780000),
    ("Uttara Kannada", 82.0, 3.6, 8.0, 18, 7.0, 1170000),
    ("Ramanagara", 73.0, 4.7, 13.0, 10, 8.0, 1080000),
    ("Chikkaballapura", 73.0, 4.8, 14.0, 12, 6.5, 1260000),
    ("Kodagu", 84.0, 3.7, 7.0, 14, 10.0, 555000),
    ("Bengaluru City", 88.7, 4.2, 8.0, 108, 14.0, 10921000),
    ("Mangaluru City", 88.0, 3.5, 7.0, 25, 11.0, 725000),
    ("Mysuru City", 85.0, 4.0, 10.0, 18, 10.0, 920000),
    ("Hubballi Dharwad", 81.0, 4.0, 10.0, 28, 8.5, 1730000),
    ("Kalaburgi City", 67.0, 5.4, 21.0, 15, 6.0, 660000),
    ("Belagavi City", 80.0, 4.1, 11.0, 18, 8.5, 610000),
    ("Vijayanagara District", 70.0, 5.3, 18.0, 12, 6.5, 1340000),
]


def main() -> None:
    df = pd.DataFrame(
        DATA,
        columns=[
            "district_name",
            "literacy_rate",
            "unemployment_rate",
            "poverty_index",
            "police_station_count",
            "alcohol_outlet_density",
            "population",
        ],
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} districts to {OUT.name}")


if __name__ == "__main__":
    main()
