"""
Approximate centroid coordinates (latitude, longitude) for Karnataka districts.

The project's hotspot data is per-district and has no lat/long. These are
publicly known district-headquarter / centroid coordinates used only to place
markers on the Folium map. They are approximate (good enough for a tactical
overview map, not survey-grade). If the team later sources official GeoJSON,
replace DISTRICT_COORDS with that source.
"""

from typing import Dict

# district (lower-cased key) -> (lat, lon)
DISTRICT_COORDS: Dict[str, tuple[float, float]] = {
    "bagalkot": (16.1783, 75.6947),
    "ballari": (15.1394, 76.9214),
    "bellary": (15.1394, 76.9214),
    "belagavi": (15.8497, 74.4977),
    "belgaum": (15.8497, 74.4977),
    "bengaluru urban": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),            # alias used by pipeline output
    "bengaluru rural": (13.1500, 77.7000),
    "bangalore rural": (13.1500, 77.7000),
    "bidar": (17.9123, 77.5199),
    "vijayapura": (16.8302, 75.7100),
    "chamarajanagar": (11.9269, 76.9411),
    "chikkaballapura": (13.4354, 77.7293),
    "chikballapur": (13.4354, 77.7293),
    "chikkamagaluru": (13.3161, 75.7720),
    "chikmagalur": (13.3161, 75.7720),
    "chitradurga": (14.2252, 76.3980),
    "dakshina kannada": (12.8700, 75.4200),
    "davanagere": (14.4644, 75.9218),
    "dharwad": (15.4589, 75.0078),
    "gadag": (15.4189, 75.6333),
    "kalaburagi": (17.3297, 76.8343),
    "gulbarga": (17.3297, 76.8343),
    "hassan": (13.0068, 76.1025),
    "haveri": (14.7947, 75.4043),
    "kodagu": (12.4244, 75.7382),
    "kolar": (13.1367, 78.1326),
    "koppal": (15.3454, 76.2105),
    "mandya": (12.5216, 76.8964),
    "mysuru": (12.2958, 76.6394),
    "mysore": (12.2958, 76.6394),
    "raichur": (16.2120, 77.3439),
    "ramanagara": (12.7187, 77.2800),
    "shivamogga": (13.9299, 75.5681),
    "shimoga": (13.9299, 75.5681),
    "tumakuru": (13.3409, 77.1025),
    "tumkur": (13.3409, 77.1025),
    "udupi": (13.3400, 74.7450),
    "uttara kannada": (14.9000, 74.5000),
    "vijayanagara": (15.0000, 76.0000),
    "yadgir": (16.2020, 76.1317),
}


def get_coords(district: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a district name, or None if unknown."""
    if not district:
        return None
    key = district.strip().lower()
    return DISTRICT_COORDS.get(key)
