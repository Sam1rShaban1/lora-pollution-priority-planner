"""Configuration loading and defaults for the aqlora pipeline.

The dataclass keeps the original open-data gateway-placement options and adds
the settings used by the notebook-equivalent PM-map and joint-placement
pipelines.  Unknown YAML keys are rejected so configuration typos surface
immediately.
"""
from dataclasses import dataclass, field
import os
from typing import Optional

import yaml


@dataclass
class Config:
    # --- area ---
    place: str = "Tetovo, North Macedonia"   # OSM geocodable place name
    utm_crs: str = "EPSG:32634"              # metric CRS for the area

    # --- candidates / precomputed artifacts ---
    candidates_csv: Optional[str] = None     # base CSV with id,latitude,longitude[,elevation]
    candidates_pm_csv: Optional[str] = None  # v2 PM output read by the joint optimizer
    rel_csv: Optional[str] = None            # link reliability matrix (CSV, first column = id)
    sf_csv: Optional[str] = None             # link average-SF matrix (CSV, first column = id)
    mean_rssi_csv: Optional[str] = None      # optional mean-RSSI matrix for delta != 0
    n_candidates: int = 100                  # used when auto-generating
    candidate_max_road_dist_m: float = 300.0
    candidate_min_spacing_m: float = 250.0

    # --- air-quality layer (legacy open-data path) ---
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
    # Exact notebook sensitivity table (SF7..SF12).
    sens_dbm: list = field(default_factory=lambda: [
        -123.0, -126.0, -129.0, -132.0, -134.5, -137.0])
    shadowing_sigma_db: float = 8.0
    mc_iters: int = 10000
    obstruction_db_per_m: float = 0.8
    obstruction_cap_db: float = 40.0
    min_building_height_m: float = 4.0
    default_building_height_m: float = 10.0
    reliability_threshold: float = 0.99

    # --- calibration ---
    rssi_bias_db: float = 0.0                # legacy open-data path
    delta_db: float = -12.4                  # joint notebook radio-model offset

    # --- joint sensor + gateway optimization (notebook parity) ---
    num_sensors: int = 15                    # M
    allow_colocation: bool = False
    sf_penalty: float = 13.0                 # SF assigned to an uncovered sensor
    geo_weight: float = 0.7                  # D_ij mix of geography and PM difference
    use_hotspot_prob: bool = True
    pm_col: Optional[str] = None             # None = auto-detect predicted-PM column
    selection_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    run_baselines: bool = True
    run_delta_sweep: bool = True
    run_budget_sweep: bool = True
    delta_ref: float = -12.4
    delta_list: list = field(default_factory=lambda: [0, -5, -10, -12.4, -15, -20])
    k_list: list = field(default_factory=lambda: [3, 4, 5, 6, 8, 10])
    m_list: list = field(default_factory=lambda: [5, 10, 15, 20, 30, 50, 70, 80])
    sweep_pop: int = 200
    sweep_gens: int = 100

    # --- PM-map v2 (notebook parity) ---
    pm_is_aqi: Optional[bool] = None
    aqi_breakpoints: str = "epa_2012"
    test_fraction: float = 0.2
    n_splits: int = 5
    min_hours_per_day: int = 18
    model_choice: str = "xgboost"            # xgboost, random_forest, ridge, auto
    moepp_lat: float = 42.0053889
    moepp_lon: float = 20.9687778
    city_bbox: list = field(default_factory=lambda: [
        20.927753, 41.969382, 21.008091, 42.033166])
    grid_res_m: float = 40.0
    density_radius_m: float = 500.0
    amplitude_scale: float = 1.0
    robust_norm: bool = True
    anchor_to_station: bool = True
    candidate_elev_source: str = "raster"    # raster or csv
    n_draws: int = 1000
    perturb: float = 0.5
    rng_seed: int = 42
    dem_path: Optional[str] = None
    slope_path: Optional[str] = None
    roads_path: Optional[str] = None
    industrial_path: Optional[str] = None
    buildings_path: Optional[str] = None
    foliage_path: Optional[str] = None
    community_glob: Optional[str] = None
    community_latlon: Optional[list] = None
    pm_summary_path: Optional[str] = None
    pm_out_csv: Optional[str] = None         # None = <out_dir>/candidate_locations_with_pm.csv

    # --- optimization ---
    k_gateways: int = 5
    pop_size: int = 500
    generations: int = 200
    seed: int = 42

    # --- outputs ---
    out_dir: str = "outputs"


def _expand(value):
    """Expand ${ENV_VAR}, $ENV_VAR and ~ in string config values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, _expand(v))
    return cfg
