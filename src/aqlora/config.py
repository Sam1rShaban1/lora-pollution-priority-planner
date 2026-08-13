"""Configuration loading and defaults for the aqlora pipeline."""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class Config:
    # --- area ---
    place: str = "Tetovo, North Macedonia"   # OSM geocodable place name
    utm_crs: str = "EPSG:32634"              # metric CRS for the area

    # --- candidates ---
    candidates_csv: Optional[str] = None     # CSV with id,latitude,longitude[,elevation]
    n_candidates: int = 100                  # used when auto-generating
    candidate_max_road_dist_m: float = 300.0
    candidate_min_spacing_m: float = 250.0

    # --- air quality layer ---
    aq_csv: Optional[str] = None             # aqicn.org daily export (AQI sub-indices)
    weather_csv: Optional[str] = None        # meteostat hourly export
    pm_baseline: float = 25.0                # fallback ug/m3 if no AQ/weather data
    frac: dict = field(default_factory=lambda: dict(
        road=0.21, ind=0.35, dens=0.21, elev=-0.21, slope=-0.07, fol=-0.11))
    decay_m: dict = field(default_factory=lambda: dict(road=500.0, ind=1000.0, fol=300.0))
    hotspot_quantile: float = 0.75
    gamma: float = 4.0                       # hotspot weight contrast (threshold; any >=4 equivalent)

    # --- radio / hardware (SX1276, EU 868) ---
    tx_dbm: float = 14.0
    ant_gain_db: float = 4.0                 # tx + rx
    freq_hz: float = 868e6
    sens_dbm: list = field(default_factory=lambda: [-123.0, -125.5, -128.0, -130.5, -133.0, -137.0])
    shadowing_sigma_db: float = 8.0
    mc_iters: int = 10000
    obstruction_db_per_m: float = 0.8
    obstruction_cap_db: float = 40.0
    min_building_height_m: float = 4.0
    default_building_height_m: float = 10.0
    reliability_threshold: float = 0.99

    # --- calibration ---
    rssi_bias_db: float = 0.0                # apply measured field bias (e.g. -12.4)

    # --- optimization ---
    k_gateways: int = 5
    pop_size: int = 500
    generations: int = 200
    seed: int = 42

    # --- outputs ---
    out_dir: str = "outputs"


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
