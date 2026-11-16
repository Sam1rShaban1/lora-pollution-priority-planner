"""PM2.5 priority surface, ported from ``synthetic_pm_map_v2.ipynb``.

The notebook is the reference implementation.  This module keeps the same
units audit, weather model, spatial corrections, station anchoring,
robustness draws, output columns and summary structure.  Heavy geospatial
dependencies are imported lazily so the rest of the package can be used
without rasterio/geopandas installed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


AQI_TABLES = {
    "epa_2012": [(0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4),
                 (101, 150, 35.5, 55.4), (151, 200, 55.5, 150.4),
                 (201, 300, 150.5, 250.4), (301, 400, 250.5, 350.4),
                 (401, 500, 350.5, 500.4)],
    "epa_2024": [(0, 50, 0.0, 9.0), (51, 100, 9.1, 35.4),
                 (101, 150, 35.5, 55.4), (151, 200, 55.5, 125.4),
                 (201, 300, 125.5, 225.4), (301, 500, 225.5, 325.4)],
}

SEASONS = {"Winter (Dec-Feb)": [12, 1, 2], "Spring (Mar-May)": [3, 4, 5],
           "Summer (Jun-Aug)": [6, 7, 8], "Autumn (Sep-Nov)": [9, 10, 11]}

DEFAULT_FEATURES = [
    "temp", "temp_min", "temp_max", "dtr", "hdd", "dwpt", "rhum", "prcp",
    "snow", "wspd", "wpgt", "pres", "dpres", "tsun", "wdir_sin", "wdir_cos",
    "temp_roll3", "wspd_roll3", "prcp_roll3", "doy_sin", "doy_cos",
    "is_weekend",
]


# ---------------------------------------------------------------------------
# Units / raster helpers
# ---------------------------------------------------------------------------
def aqi_to_pm25(aqi, table):
    aqi = np.asarray(aqi, dtype=float)
    out = np.full(aqi.shape, np.nan)
    for k, (ilo, ihi, clo, chi) in enumerate(table):
        upper = table[k + 1][0] if k + 1 < len(table) else np.inf
        m = (aqi >= ilo) & (aqi < upper)
        out[m] = clo + (aqi[m] - ilo) * (chi - clo) / (ihi - ilo)
    out[aqi > table[-1][1]] = table[-1][3]
    return out


def nearest_dist(x, y, gdf_layer):
    from shapely import STRtree
    import shapely

    g = gdf_layer[gdf_layer.geometry.notna() & ~gdf_layer.geometry.is_empty]
    tree = STRtree(list(g.geometry))
    pts = shapely.points(np.asarray(x, float), np.asarray(y, float))
    idx, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    d = np.full(len(pts), np.nan)
    d[idx[0]] = dist
    return d


def sample_raster(path, x, y, xy_crs):
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        if src.crs is not None and xy_crs is not None and src.crs != xy_crs:
            xs, ys = Transformer.from_crs(
                xy_crs, src.crs, always_xy=True).transform(
                    np.asarray(x), np.asarray(y))
        else:
            xs, ys = np.asarray(x), np.asarray(y)
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))],
                        dtype=float)
        nod = src.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return vals


# ---------------------------------------------------------------------------
# Weather / PM model
# ---------------------------------------------------------------------------
def daily_weather_features(weather_csv: str, min_hours_per_day: int = 18):
    w = pd.read_csv(weather_csv)
    w = w.rename(columns={"time": "date"})
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index("date").sort_index()
    for c in ["temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd",
              "wpgt", "pres", "tsun"]:
        if c not in w.columns:
            w[c] = np.nan
    rad = np.deg2rad(w["wdir"])
    w["_ws"], w["_wc"] = np.sin(rad), np.cos(rad)
    g = w.resample("D")
    daily = pd.DataFrame({
        "temp": g["temp"].mean(), "temp_min": g["temp"].min(),
        "temp_max": g["temp"].max(), "dwpt": g["dwpt"].mean(),
        "rhum": g["rhum"].mean(),
        "prcp": g["prcp"].sum(min_count=1), "snow": g["snow"].max(),
        "wspd": g["wspd"].mean(), "wpgt": g["wpgt"].max(),
        "pres": g["pres"].mean(), "tsun": g["tsun"].sum(min_count=1),
        "wdir_sin": g["_ws"].mean(), "wdir_cos": g["_wc"].mean(),
        "n_hours": g["temp"].count(),
    })
    daily = daily[daily["n_hours"] >= min_hours_per_day].drop(columns="n_hours")

    miss = daily.isna().mean()
    drop_cols = miss[miss > 0.30].index.tolist()
    if drop_cols:
        print("Dropping weather features with >30% missing:", drop_cols)
    daily = daily.drop(columns=drop_cols)
    for c in ["prcp", "snow"]:
        if c in daily.columns:
            daily[c] = daily[c].fillna(0.0)
    daily = daily.interpolate(limit=3, limit_area="inside")

    daily["dtr"] = daily["temp_max"] - daily["temp_min"]
    daily["hdd"] = (15.0 - daily["temp"]).clip(lower=0)
    daily["temp_roll3"] = daily["temp"].rolling(3, min_periods=1).mean()
    daily["wspd_roll3"] = daily["wspd"].rolling(3, min_periods=1).mean()
    if "prcp" in daily.columns:
        daily["prcp_roll3"] = daily["prcp"].rolling(3, min_periods=1).sum()
    daily["dpres"] = daily["pres"].diff()
    doy = daily.index.dayofyear.values
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    daily["is_weekend"] = (daily.index.dayofweek >= 5).astype(float)
    daily["month"] = daily.index.month
    return daily


def load_pm_record(pm_csv: str, pm_is_aqi=None, aqi_breakpoints="epa_2012"):
    pm_raw = pd.read_csv(pm_csv)
    pm_raw.columns = pm_raw.columns.str.strip().str.lower()
    pm_raw["date"] = pd.to_datetime(pm_raw["date"]).dt.normalize()
    pm_raw = pm_raw.sort_values("date")

    share = (float((pm_raw["pm25"] > pm_raw["pm10"]).mean())
             if "pm10" in pm_raw.columns else np.nan)
    auto_aqi = bool(share > 0.5) if not np.isnan(share) else False
    is_aqi = auto_aqi if pm_is_aqi is None else bool(pm_is_aqi)

    print("=== UNITS AUDIT ===")
    print(f"PM2.5 > PM10 on {100 * share:.1f}% of days "
          "(impossible for true concentrations; expected for AQI sub-index scores)")
    print("Treating the record as: "
          f"{'AQI sub-index scores (converting)' if is_aqi else 'concentrations in ug/m3 (no conversion)'}")
    print(f"Raw PM2.5 mean: {pm_raw['pm25'].mean():.2f}")
    if is_aqi:
        for name, tbl in AQI_TABLES.items():
            print(f"  mean after {name} conversion: "
                  f"{np.nanmean(aqi_to_pm25(pm_raw['pm25'].values, tbl)):.2f} ug/m3")
        pm_raw["pm25_ugm3"] = aqi_to_pm25(
            pm_raw["pm25"].values, AQI_TABLES[aqi_breakpoints])
        print(f"Using {aqi_breakpoints}")
    else:
        pm_raw["pm25_ugm3"] = pm_raw["pm25"].astype(float)

    pm_series = pm_raw.groupby("date")["pm25_ugm3"].mean().dropna()
    station_obs = float(pm_series.mean())
    print(f"Reference-station long-run mean PM2.5: {station_obs:.2f} ug/m3 "
          f"over {len(pm_series)} days ({station_obs / 5.0:.1f}x the WHO annual guideline of 5)")
    return pm_raw, pm_series, station_obs, is_aqi, share


def _scores(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return dict(R2=r2_score(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                MAE=mean_absolute_error(y_true, y_pred))


class MonthClimatology:
    def fit(self, X, y):
        self.m_ = y.groupby(X["month"]).mean()
        self.g_ = float(y.mean())
        return self

    def predict(self, X):
        return X["month"].map(self.m_).fillna(self.g_).values


def fit_weather_model(daily: pd.DataFrame, pm_series: pd.Series,
                      test_fraction: float = 0.2, n_splits: int = 5,
                      model_choice: str = "xgboost", seed: int = 42):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = [c for c in DEFAULT_FEATURES if c in daily.columns]
    data = (daily.join(pm_series.rename("pm25_ugm3"), how="inner")
            .dropna(subset=features + ["pm25_ugm3"]).sort_index())
    print(f"\nModelling set: {len(data)} days, {data.index.min().date()} to "
          f"{data.index.max().date()}, {len(features)} weather/calendar features")
    print("Features:", features)

    n = len(data)
    cut = int(n * (1 - test_fraction))
    X_tr = data.iloc[:cut][features + ["month"]]
    y_tr = data.iloc[:cut]["pm25_ugm3"]
    X_te = data.iloc[cut:][features + ["month"]]
    y_te = data.iloc[cut:]["pm25_ugm3"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"Train: {len(X_tr)} days (to {X_tr.index.max().date()}) | "
          f"Holdout: {len(X_te)} days (from {X_te.index.min().date()})")

    grid = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.06],
            "n_estimators": [200, 400], "min_child_weight": [3, 8]}
    base_xgb = xgb.XGBRegressor(
        objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, random_state=seed, n_jobs=1)
    gs = GridSearchCV(base_xgb, grid, cv=tscv,
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr[features], y_tr)
    best_xgb = gs.best_params_
    print("Best XGBoost params:", best_xgb)

    def make_xgb():
        return xgb.XGBRegressor(
            objective="reg:squarederror", subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, random_state=seed,
            n_jobs=-1, **best_xgb)

    models = {
        "Climatology (month mean)": (MonthClimatology, ["month"]),
        "Ridge": (lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), features),
        "Random Forest": (lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=seed), features),
        "XGBoost (tuned)": (make_xgb, features),
    }

    rows = []
    for name, (factory, cols) in models.items():
        cv = []
        for tr, te in tscv.split(X_tr):
            m = factory().fit(X_tr.iloc[tr][cols], y_tr.iloc[tr])
            cv.append(_scores(y_tr.iloc[te], m.predict(X_tr.iloc[te][cols])))
        cv = pd.DataFrame(cv)
        m = factory().fit(X_tr[cols], y_tr)
        ho = _scores(y_te, m.predict(X_te[cols]))
        rows.append({
            "Model": name,
            "CV R2": f"{cv['R2'].mean():.2f} +/- {cv['R2'].std():.2f}",
            "CV RMSE": f"{cv['RMSE'].mean():.1f} +/- {cv['RMSE'].std():.1f}",
            "CV RMSE (num)": cv["RMSE"].mean(),
            "Holdout R2": round(ho["R2"], 3),
            "Holdout RMSE": round(ho["RMSE"], 2),
            "Holdout MAE": round(ho["MAE"], 2),
        })
    model_table = pd.DataFrame(rows).set_index("Model")

    choice = {"xgboost": "XGBoost (tuned)", "random_forest": "Random Forest",
              "ridge": "Ridge"}
    if model_choice == "auto":
        cand_names = ["Ridge", "Random Forest", "XGBoost (tuned)"]
        chosen = min(cand_names,
                     key=lambda k: model_table.loc[k, "CV RMSE (num)"])
    else:
        chosen = choice[model_choice]
    print("Final model:", chosen)

    factory, cols = models[chosen]
    final_model = factory().fit(data[cols], data["pm25_ugm3"])
    data["pred"] = final_model.predict(data[cols])

    m_ho = factory().fit(X_tr[cols], y_tr)
    ho_pred = pd.Series(m_ho.predict(X_te[cols]), index=X_te.index)
    holdout = _scores(y_te, ho_pred.values)
    print(f"Chosen model, chronological holdout: R2 = {holdout['R2']:.3f}, "
          f"RMSE = {holdout['RMSE']:.2f} ug/m3, MAE = {holdout['MAE']:.2f}")
    return dict(data=data, features=features, table=model_table,
                chosen=chosen, best_xgb=best_xgb, holdout=holdout,
                cut=cut, n=n, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                ho_pred=ho_pred, final_model=final_model)


def seasonal_baselines(data: pd.DataFrame):
    baselines = {"Annual": float(data["pred"].mean())}
    observed = {"Annual": float(data["pm25_ugm3"].mean())}
    for name, months in SEASONS.items():
        mask = data["month"].isin(months)
        baselines[name] = float(data.loc[mask, "pred"].mean())
        observed[name] = float(data.loc[mask, "pm25_ugm3"].mean())
    win = data["month"].isin([12, 1, 2])
    episode = (win & (data["wspd"] <= data.loc[win, "wspd"].quantile(0.25))
               & (data["temp"] <= data.loc[win, "temp"].median()))
    scenario = "Winter stagnation (calm and cold)"
    baselines[scenario] = float(data.loc[episode, "pred"].mean())
    observed[scenario] = float(data.loc[episode, "pm25_ugm3"].mean())
    scale = {k: v / baselines["Annual"] for k, v in baselines.items()}
    return baselines, observed, scale, episode


# ---------------------------------------------------------------------------
# Geospatial surface
# ---------------------------------------------------------------------------
def load_layer(path, utm):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(utm)


def build_surface(cfg):
    """Build the geospatial grid and candidate PM values.

    Returns a dictionary with the notebook's intermediate objects.
    """
    import geopandas as gpd
    import rasterio
    import shapely
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from shapely.geometry import box

    with rasterio.open(cfg.dem_path) as src:
        dem_bounds, dem_crs = src.bounds, src.crs
    aoi_geo = gpd.GeoSeries([box(*dem_bounds)], crs=dem_crs)
    utm = aoi_geo.estimate_utm_crs()
    aoi_utm = aoi_geo.to_crs(utm).iloc[0]
    print("Projected CRS:", utm)

    roads = load_layer(cfg.roads_path, utm)
    industrial = load_layer(cfg.industrial_path, utm)
    buildings = load_layer(cfg.buildings_path, utm)
    foliage = load_layer(cfg.foliage_path, utm)
    print(f"roads {len(roads)}, industrial {len(industrial)}, "
          f"buildings {len(buildings)}, foliage {len(foliage)}")

    bc = buildings.geometry.centroid
    bld_tree = cKDTree(np.c_[bc.x.values, bc.y.values])
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)

    def raw_features(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        elev = sample_raster(cfg.dem_path, x, y, utm)
        elev[elev <= 0] = np.nan
        slope = sample_raster(cfg.slope_path, x, y, utm)
        slope[slope < 0] = np.nan
        return {
            "d_road": nearest_dist(x, y, roads),
            "d_ind": nearest_dist(x, y, industrial),
            "d_fol": nearest_dist(x, y, foliage),
            "dens": bld_tree.query_ball_point(
                np.c_[x, y], r=cfg.density_radius_m,
                return_length=True).astype(float),
            "elev": elev, "slope": slope,
        }

    lon_min, lat_min, lon_max, lat_max = cfg.city_bbox
    lat0 = 0.5 * (lat_min + lat_max)
    dlat = cfg.grid_res_m / 110574.0
    dlon = cfg.grid_res_m / (111320.0 * np.cos(np.deg2rad(lat0)))
    lon_v = np.arange(lon_min, lon_max + dlon, dlon)
    lat_v = np.arange(lat_min, lat_max + dlat, dlat)
    lon_g, lat_g = np.meshgrid(lon_v, lat_v)
    grid_shape = lon_g.shape
    xg, yg = to_utm.transform(lon_g.ravel(), lat_g.ravel())
    inside = shapely.contains_xy(aoi_utm, xg, yg)
    print(f"Grid {grid_shape[1]} x {grid_shape[0]} at ~{cfg.grid_res_m} m, "
          f"{int(inside.sum())} points inside the DEM extent")

    raw_g = raw_features(xg[inside], yg[inside])
    print("Grid features computed.")

    def _range(a):
        a = a[np.isfinite(a)]
        return ((np.percentile(a, 1), np.percentile(a, 99))
                if cfg.robust_norm else (a.min(), a.max()))

    norms = {
        "elev": _range(raw_g["elev"]), "slope": _range(raw_g["slope"]),
        "dens": (0.0, (np.percentile(raw_g["dens"], 99)
                       if cfg.robust_norm else raw_g["dens"].max())),
    }

    def _nz(v, lo, hi):
        return np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    fill = {k: float(np.nanmedian(_nz(raw_g[k], *norms[k])))
            for k in norms}
    print("Normalization ranges:",
          {k: tuple(round(float(x), 2) for x in v) for k, v in norms.items()})

    def normalise(raw):
        fn = {"d_road": raw["d_road"], "d_ind": raw["d_ind"],
              "d_fol": raw["d_fol"]}
        for k in norms:
            z = _nz(raw[k], *norms[k])
            fn[k] = np.where(np.isnan(z), fill[k], z)
        return fn

    param_names = ["a_road", "a_ind", "a_dens", "a_elev", "a_slope", "a_fol",
                   "l_road", "l_ind", "l_fol"]
    base_vec = np.array([
        cfg.frac["road"], cfg.frac["ind"], cfg.frac["dens"],
        abs(cfg.frac["elev"]), abs(cfg.frac["slope"]), abs(cfg.frac["fol"]),
        cfg.decay_m["road"], cfg.decay_m["ind"], cfg.decay_m["fol"],
    ])

    def terms_from(fn, vec=base_vec, scale=None):
        scale = cfg.amplitude_scale if scale is None else scale
        a_road, a_ind, a_dens, a_elev, a_slope, a_fol, l_road, l_ind, l_fol = vec
        return {
            "road": scale * a_road * np.exp(-fn["d_road"] / l_road),
            "industrial": scale * a_ind * np.exp(-fn["d_ind"] / l_ind),
            "density": scale * a_dens * fn["dens"],
            "elevation": -scale * a_elev * fn["elev"],
            "slope": -scale * a_slope * fn["slope"],
            "foliage": -scale * a_fol * np.exp(-fn["d_fol"] / l_fol),
        }

    def factor_from(fn, vec=base_vec):
        return np.clip(1.0 + sum(terms_from(fn, vec).values()), 0.0, None)

    fn_g = normalise(raw_g)

    sx, sy = to_utm.transform([cfg.moepp_lon], [cfg.moepp_lat])
    raw_s = raw_features(sx, sy)
    fn_s = normalise(raw_s)
    f_station = float(factor_from(fn_s)[0])
    print(f"Spatial factor at the MOEPP station: {f_station:.3f}")

    # candidates
    cand = pd.read_csv(cfg.candidates_csv)
    cand["id"] = cand["id"].astype(str)
    cx, cy = to_utm.transform(cand["longitude"].values, cand["latitude"].values)
    raw_c = raw_features(cx, cy)
    if cfg.candidate_elev_source == "csv":
        raw_c["elev"] = cand["elevation"].values.astype(float)
        print("WARNING: candidate elevation from the CSV (FABDEM) is normalized "
              "with DEM-raster constants; results are not on the same scale as the map.")
    fn_c = normalise(raw_c)

    return dict(
        utm=utm, to_utm=to_utm, raw_features=raw_features,
        roads=roads, industrial=industrial, buildings=buildings,
        foliage=foliage, lon_v=lon_v, lat_v=lat_v, lon_g=lon_g, lat_g=lat_g,
        grid_shape=grid_shape, inside=inside, raw_g=raw_g, raw_c=raw_c,
        raw_s=raw_s, norms=norms, fill=fill, normalise=normalise,
        fn_g=fn_g, fn_c=fn_c, fn_s=fn_s, f_station=f_station,
        par_names=param_names, base_vec=base_vec,
        terms_from=terms_from, factor_from=factor_from,
        cand=cand, cx=cx, cy=cy, sx=sx, sy=sy,
    )


def candidate_surface(surface, cfg, pm_station_obs, baselines):
    fn_c, fn_g = surface["fn_c"], surface["fn_g"]
    terms_c = surface["terms_from"](fn_c)
    factor_c = surface["factor_from"](fn_c)
    base_annual = (pm_station_obs / surface["f_station"]
                   if cfg.anchor_to_station else baselines["Annual"])
    pm_c = base_annual * factor_c

    terms_g = surface["terms_from"](fn_g)
    factor_g = surface["factor_from"](fn_g)
    pm_g = np.full(surface["grid_shape"][0] * surface["grid_shape"][1], np.nan)
    pm_g[surface["inside"]] = base_annual * factor_g
    pm_grid = pm_g.reshape(surface["grid_shape"])

    print(f"Candidates: n = {len(surface['cand'])}")
    print(f"Predicted PM2.5 over candidates: min {pm_c.min():.1f}, "
          f"mean {pm_c.mean():.1f}, max {pm_c.max():.1f} ug/m3 "
          f"(max/min = {pm_c.max() / pm_c.min():.2f}, "
          f"CV = {pm_c.std() / pm_c.mean():.3f})")
    print(f"Predicted PM2.5 over the city grid: p5 "
          f"{np.nanpercentile(pm_grid, 5):.1f}, p95 "
          f"{np.nanpercentile(pm_grid, 95):.1f} ug/m3")
    n_nan = {k: int(np.isnan(surface["raw_c"][k]).sum())
             for k in ["elev", "slope"]}
    print("Candidates with missing raster values "
          "(filled with the domain median):", n_nan)
    print(f"Annual baseline used for the map: {base_annual:.2f} ug/m3 "
          f"({'station-anchored' if cfg.anchor_to_station else 'model mean'}); "
          f"station long-run mean = {pm_station_obs:.2f}")

    return dict(terms_c=terms_c, factor_c=factor_c, pm_c=pm_c,
                terms_g=terms_g, factor_g=factor_g, pm_g=pm_g,
                pm_grid=pm_grid, base_annual=base_annual)


def robustness(surface, cfg, pm_station_obs, baselines):
    from scipy.stats import rankdata

    rng = np.random.default_rng(cfg.rng_seed)
    fn_c, fn_g = surface["fn_c"], surface["fn_g"]
    base_vec = surface["base_vec"]
    factor_from = surface["factor_from"]
    pm_station_obs = pm_station_obs

    def pm_candidates(vec):
        f_c = factor_from(fn_c, vec)
        base = (pm_station_obs / float(factor_from(surface["fn_s"], vec)[0])
                if cfg.anchor_to_station else baselines["Annual"])
        return base * f_c

    sub = rng.choice(len(surface["fn_g"]["d_road"]),
                     size=min(5000, len(surface["fn_g"]["d_road"])),
                     replace=False)
    fn_g_sub = {k: v[sub] for k, v in fn_g.items()}

    def rho(a, b):
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    pm0 = pm_candidates(base_vec)
    f0_sub = factor_from(fn_g_sub, base_vec)
    q0 = np.quantile(pm0, cfg.hotspot_quantile)
    hot0 = pm0 >= q0

    oat = []
    for i, name in enumerate(surface["par_names"]):
        row = {"parameter": name}
        for lab, mult in (("x0.5", 1 - cfg.perturb), ("x1.5", 1 + cfg.perturb)):
            v = base_vec.copy()
            v[i] *= mult
            row[f"rho candidates {lab}"] = round(rho(pm0, pm_candidates(v)), 3)
            row[f"rho grid {lab}"] = round(
                rho(f0_sub, factor_from(fn_g_sub, v)), 3)
        row["min rho"] = min(row[k] for k in row if k.startswith("rho"))
        oat.append(row)
    oat_df = pd.DataFrame(oat).sort_values("min rho").set_index("parameter")

    mult = rng.uniform(1 - cfg.perturb, 1 + cfg.perturb,
                       size=(cfg.n_draws, len(base_vec)))
    rho_c = np.zeros(cfg.n_draws)
    rho_g = np.zeros(cfg.n_draws)
    hot_count = np.zeros(len(surface["cand"]))
    pm_draws = np.zeros((cfg.n_draws, len(surface["cand"])))
    for d in range(cfg.n_draws):
        v = base_vec * mult[d]
        p = pm_candidates(v)
        pm_draws[d] = p
        rho_c[d] = rho(pm0, p)
        rho_g[d] = rho(f0_sub, factor_from(fn_g_sub, v))
        hot_count += (p >= np.quantile(p, cfg.hotspot_quantile))
    hot_prob = hot_count / cfg.n_draws
    pm_p05, pm_p95 = np.percentile(pm_draws, 5, axis=0), np.percentile(pm_draws, 95, axis=0)

    print("\n=== 1,000-draw joint perturbation "
          "(each parameter x U(0.5, 1.5)) ===")
    print(f"Candidate ranking: median rho = {np.median(rho_c):.3f}, "
          f"min = {rho_c.min():.3f}, 5th percentile = "
          f"{np.percentile(rho_c, 5):.3f}")
    print(f"City-grid ranking: median rho = {np.median(rho_g):.3f}, "
          f"min = {rho_g.min():.3f}, 5th percentile = "
          f"{np.percentile(rho_g, 5):.3f}")
    stable = hot0 & (hot_prob >= 0.90)
    print(f"Hotspot candidates (top {int((1 - cfg.hotspot_quantile) * 100)}%): "
          f"{int(hot0.sum())}; remain hotspots in >=90% of draws: "
          f"{int(stable.sum())}; in >=50%: "
          f"{int((hot0 & (hot_prob >= 0.5)).sum())}")
    print(f"Most influential parameter (lowest rho): {oat_df.index[0]}")

    return dict(rng=rng, pm0=pm0, hot0=hot0, q0=q0, oat=oat_df,
                rho_c=rho_c, rho_g=rho_g, hot_prob=hot_prob,
                pm_p05=pm_p05, pm_p95=pm_p95, stable=stable,
                pm_draws=pm_draws)


def community_validation(surface, cfg, pm_series):
    from scipy.stats import pearsonr, spearmanr

    community = {}
    files = sorted(glob.glob(cfg.community_glob)) if cfg.community_glob else []
    if not files:
        print(f"No community-sensor files matched {cfg.community_glob}; skipping. "
              "Put the sensor.community CSV exports there to run this check.")
        return community

    parts = []
    for f in files:
        try:
            d = pd.read_csv(f, sep=";")
            if "P2" not in d.columns:
                d = pd.read_csv(f)
            parts.append(d)
        except Exception as e:  # pragma: no cover - optional input
            print("Skipped", os.path.basename(f), "-", e)
    cs = pd.concat(parts, ignore_index=True)
    cs["timestamp"] = pd.to_datetime(
        cs["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    cs["P2"] = pd.to_numeric(cs["P2"], errors="coerce")
    cs = cs.dropna(subset=["timestamp", "P2"])
    cs = cs[(cs["P2"] >= 0) & (cs["P2"] < 1000)]
    d_cs = cs.set_index("timestamp")["P2"].resample("D").agg(["mean", "count"])
    d_cs = d_cs[d_cs["count"] >= 100]["mean"]
    d_cs.index = d_cs.index.normalize()
    both = d_cs.index.intersection(pm_series.index)
    a, b = d_cs.loc[both].values, pm_series.loc[both].values
    r_p, r_s = pearsonr(a, b)[0], spearmanr(a, b)[0]
    ratio_obs = float(a.mean() / b.mean())
    community.update({
        "n_days": int(len(both)), "pearson_r": float(r_p),
        "spearman_rho": float(r_s), "mean_community": float(a.mean()),
        "mean_moepp_same_days": float(b.mean()), "observed_ratio": ratio_obs,
    })
    print(f"Overlapping days: {len(both)}")
    print(f"Daily correlation with the corrected MOEPP record: "
          f"Pearson r = {r_p:.3f}, Spearman rho = {r_s:.3f}")
    print(f"Mean community sensor {a.mean():.1f} vs MOEPP {b.mean():.1f} "
          f"ug/m3 on the same days (ratio {ratio_obs:.2f})")

    if cfg.community_latlon is not None:
        lat_s, lon_s = cfg.community_latlon
    elif {"lat", "lon"}.issubset(cs.columns):
        lat_s = float(pd.to_numeric(cs["lat"], errors="coerce").median())
        lon_s = float(pd.to_numeric(cs["lon"], errors="coerce").median())
    else:
        lat_s = lon_s = None
    if lat_s is not None:
        ux, uy = surface["to_utm"].transform([lon_s], [lat_s])
        fn_cs = surface["normalise"](surface["raw_features"](ux, uy))
        f_cs = float(surface["factor_from"](fn_cs)[0])
        ratio_map = f_cs / surface["f_station"]
        t_cs = sum(surface["terms_from"](fn_cs).values())[0]
        t_st = sum(surface["terms_from"](surface["fn_s"]).values())[0]
        community.update({"lat": lat_s, "lon": lon_s, "map_ratio": ratio_map})
        print(f"Sensor location ({lat_s:.5f}, {lon_s:.5f}); "
              f"map-predicted ratio sensor/station = {ratio_map:.2f} "
              f"vs observed {ratio_obs:.2f}")
        denom = t_cs - ratio_obs * t_st
        if abs(denom) > 1e-9:
            kappa = (ratio_obs - 1.0) / denom
            community["amplitude_scale_to_match_contrast"] = float(kappa)
            print("Diagnostic only: matching this one contrast would need "
                  f"AMPLITUDE_SCALE of about {kappa:.1f} (1 = paper's model).")
        print("Caveat: one low-cost optical sensor is a directional check, "
              "not a calibration; humidity bias is not corrected here.")
    return community


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _savefig(fig, base_path, name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(os.path.join(base_path, name + ".png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(base_path, name + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


def make_figures(surface, pm, cfg, model, robustness_result,
                 community, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as mpatches

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 13})
    os.makedirs(out_dir, exist_ok=True)

    # model evaluation
    data = model["data"]
    y_te, ho_pred, holdout = model["y_te"], model["ho_pred"], model["holdout"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    ax = axes[0]
    ax.plot(y_te.index, y_te.values, color="gray", lw=1, label="Observed")
    ax.plot(ho_pred.index, ho_pred.values, color="crimson", lw=1,
            label="Model (holdout)")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title("Chronological holdout")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    ax = axes[1]
    ax.scatter(y_te.values, ho_pred.values, s=14, alpha=0.5)
    lim = [0, max(y_te.max(), ho_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Observed (ug/m3)")
    ax.set_ylabel("Predicted (ug/m3)")
    ax.set_title(f"R2 = {holdout['R2']:.2f}, RMSE = {holdout['RMSE']:.1f}")
    ax.grid(True, ls="--", alpha=0.4)
    ax = axes[2]
    mo = data.groupby("month")[["pm25_ugm3", "pred"]].mean()
    ax.plot(mo.index, mo["pm25_ugm3"], "o-", color="gray", label="Observed")
    ax.plot(mo.index, mo["pred"], "s--", color="crimson", label="Model")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM2.5 (ug/m3)")
    ax.set_title("Seasonal cycle")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    _savefig(fig, out_dir, "pm_model_evaluation")

    final_model = model["final_model"]
    if hasattr(final_model, "get_booster"):
        sc = final_model.get_booster().get_score(importance_type="gain")
        items = sorted(sc.items(), key=lambda kv: kv[1])[-15:]
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh([k for k, _ in items], [v for _, v in items],
                       color="steelblue", height=0.65)
        for b, (_, v) in zip(bars, items):
            ax.text(b.get_width() * 1.01,
                    b.get_y() + b.get_height() / 2, f"{v:.1f}",
                    va="center", fontsize=12)
        ax.set_xlabel("Average gain")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, max(v for _, v in items) * 1.15)
        ax.grid(True, axis="x", ls="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        _savefig(fig, out_dir, "pm_feature_importance")

    # robustness
    rr = robustness_result
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].hist(rr["rho_c"], bins=30, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Spearman rho vs original ranking")
    axes[0].set_ylabel("Draws")
    axes[0].set_title("Ranking stability (candidates)")
    axes[0].grid(True, ls="--", alpha=0.4)
    order = np.argsort(-rr["pm0"])
    axes[1].bar(range(len(order)), rr["hot_prob"][order],
                color=["crimson" if rr["hot0"][i] else "lightgray"
                       for i in order], width=1.0)
    axes[1].axhline(0.9, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Candidates (ordered by predicted PM2.5)")
    axes[1].set_ylabel("Probability of being a hotspot")
    axes[1].set_title("Hotspot stability (red = original hotspots)")
    axes[1].grid(True, ls="--", alpha=0.4)
    _savefig(fig, out_dir, "pm_parameter_robustness")

    # surface maps
    roads_ll, ind_ll, fol_ll, bld_ll = [
        g_.to_crs("EPSG:4326") for g_ in
        (surface["roads"], surface["industrial"], surface["foliage"],
         surface["buildings"])]
    hot = pm["pm_c"] >= np.quantile(pm["pm_c"], cfg.hotspot_quantile)
    extent = (surface["lon_v"][0] - (surface["lon_v"][1] - surface["lon_v"][0]) / 2,
              surface["lon_v"][-1] + (surface["lon_v"][1] - surface["lon_v"][0]) / 2,
              surface["lat_v"][0] - (surface["lat_v"][1] - surface["lat_v"][0]) / 2,
              surface["lat_v"][-1] + (surface["lat_v"][1] - surface["lat_v"][0]) / 2)
    aspect = 1.0 / np.cos(np.deg2rad(
        0.5 * (surface["lat_v"][0] + surface["lat_v"][-1])))

    def plot_surface(overlays, fname, show_candidates=True):
        fig, ax = plt.subplots(figsize=(13, 12))
        ax.set_facecolor("#d9d9d9")
        im = ax.imshow(pm["pm_grid"], extent=extent, origin="lower",
                       cmap="inferno_r", aspect=aspect)
        handles = []
        if "buildings" in overlays:
            bld_ll.plot(ax=ax, facecolor="white", edgecolor="none", alpha=0.35)
            handles.append(mpatches.Patch(facecolor="white", edgecolor="gray",
                                          label="Buildings"))
        if "foliage" in overlays:
            fol_ll.plot(ax=ax, color="green", alpha=0.35)
            handles.append(mpatches.Patch(color="green", alpha=0.5,
                                          label="Foliage"))
        if "industrial" in overlays:
            ind_ll.plot(ax=ax, color="red", alpha=0.45)
            handles.append(mpatches.Patch(color="red", alpha=0.5,
                                          label="Industrial"))
        if "roads" in overlays:
            roads_ll.plot(ax=ax, color="lightgray", linewidth=1.3, alpha=0.9)
            handles.append(mpatches.Patch(color="lightgray", label="Roads"))
        if show_candidates:
            cand = surface["cand"]
            ax.scatter(cand["longitude"][~hot], cand["latitude"][~hot],
                       s=45, c="white", edgecolors="black", linewidths=0.8,
                       zorder=6)
            ax.scatter(cand["longitude"][hot], cand["latitude"][hot],
                       s=70, c="white", edgecolors="red", linewidths=2.2,
                       zorder=7)
            handles += [
                mpatches.Patch(facecolor="white", edgecolor="black",
                               label="Candidate site"),
                mpatches.Patch(facecolor="white", edgecolor="red",
                               label="Hotspot candidate")]
        ax.plot(cfg.moepp_lon, cfg.moepp_lat, marker="*", color="cyan",
                markeredgecolor="black", markersize=22, zorder=8, linestyle="")
        handles.append(plt.Line2D([0], [0], marker="*", color="cyan",
                                  markeredgecolor="black", markersize=16,
                                  linestyle="", label="MOEPP station"))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Predicted PM2.5 (ug/m3)", fontsize=16)
        cb.ax.tick_params(labelsize=13)
        ax.legend(handles=handles, loc="upper right", fontsize=14,
                  framealpha=0.92)
        _savefig(fig, out_dir, fname)

    plot_surface([], "pm_map_clean", show_candidates=False)
    plot_surface(["buildings", "foliage", "industrial", "roads"],
                 "pm_map_all_features")

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for ax, (name, t) in zip(axes.ravel(), pm["terms_g"].items()):
        arr = np.full(surface["grid_shape"][0] * surface["grid_shape"][1],
                      np.nan)
        arr[surface["inside"]] = 100 * t
        im = ax.imshow(arr.reshape(surface["grid_shape"]), extent=extent,
                       origin="lower", cmap="viridis", aspect=aspect)
        ax.set_title(f"{name.capitalize()} term (% of baseline)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.15)
        plt.colorbar(im, cax=cax).ax.tick_params(labelsize=12)
    _savefig(fig, out_dir, "pm_term_panels")

    if community:
        # Only produced when sensor.community files are available.
        pass


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def run_pm_map(cfg, out_dir: str | None = None, make_plots: bool = True):
    """Run the notebook-equivalent v2 PM-map pipeline.

    Requires the raw DEM/slope/GeoJSON inputs configured in ``cfg``.  When
    those are unavailable, use the committed ``candidate_locations_with_pm.csv``
    artifact together with ``scripts/check_parity.py`` instead.
    """
    out_dir = out_dir or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    required = [cfg.weather_csv, cfg.aq_csv, cfg.dem_path, cfg.slope_path,
                cfg.roads_path, cfg.industrial_path, cfg.buildings_path,
                cfg.foliage_path, cfg.candidates_csv]
    missing = [p for p in required if not p or not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "PM-map inputs missing: " + ", ".join(str(p) for p in missing))

    daily = daily_weather_features(cfg.weather_csv, cfg.min_hours_per_day)
    pm_raw, pm_series, station_obs, is_aqi, share = load_pm_record(
        cfg.aq_csv, cfg.pm_is_aqi, cfg.aqi_breakpoints)
    model = fit_weather_model(daily, pm_series, cfg.test_fraction,
                              cfg.n_splits, cfg.model_choice, cfg.seed)
    baselines, observed, scale, episode = seasonal_baselines(model["data"])

    surface = build_surface(cfg)
    pm = candidate_surface(surface, cfg, station_obs, baselines)
    rr = robustness(surface, cfg, station_obs, baselines)
    community = community_validation(surface, cfg, pm_series)

    out = surface["cand"].copy()
    out_csv = cfg.pm_out_csv or os.path.join(
        out_dir, "candidate_locations_with_pm.csv")
    if os.path.exists(out_csv):
        shutil.copy(out_csv, out_csv.replace(".csv", "_previous.csv"))
        print("Existing file backed up as",
              os.path.basename(out_csv.replace(".csv", "_previous.csv")))

    hot = pm["pm_c"] >= np.quantile(pm["pm_c"], cfg.hotspot_quantile)
    out["elevation_dem"] = surface["raw_c"]["elev"]
    out["predicted_pm"] = pm["pm_c"]
    out["pm_p05"], out["pm_p95"] = rr["pm_p05"], rr["pm_p95"]
    out["hotspot"] = hot
    out["hotspot_prob"] = rr["hot_prob"]
    out["spatial_factor"] = pm["factor_c"]
    out["pm_winter"] = pm["pm_c"] * scale["Winter (Dec-Feb)"]
    out["pm_summer"] = pm["pm_c"] * scale["Summer (Jun-Aug)"]
    for name, t in pm["terms_c"].items():
        out[f"term_{name}"] = t
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Saved", out_csv, "| columns:", list(out.columns))

    summary = {
        "units": {"treated_as_aqi": is_aqi,
                  "aqi_table": cfg.aqi_breakpoints if is_aqi else None,
                  "pm25_gt_pm10_share": share,
                  "station_long_run_mean_ugm3": station_obs},
        "model": {"chosen": model["chosen"], "features": model["features"],
                  "xgboost_params": model["best_xgb"],
                  "holdout": model["holdout"],
                  "train_days": int(model["cut"]),
                  "holdout_days": int(model["n"] - model["cut"]),
                  "table": model["table"].drop(
                      columns="CV RMSE (num)").reset_index().to_dict(
                          orient="records")},
        "baselines": {"model": baselines, "observed": observed,
                      "scale": scale},
        "spatial": {"alpha": {k: abs(float(v)) for k, v in cfg.frac.items()},
                    "lambda": cfg.decay_m,
                    "amplitude_scale": cfg.amplitude_scale,
                    "robust_norm": cfg.robust_norm,
                    "anchored": cfg.anchor_to_station,
                    "station_factor": surface["f_station"],
                    "base_annual": pm["base_annual"],
                    "norms": {k: [float(x) for x in v]
                              for k, v in surface["norms"].items()}},
        "robustness": {
            "n_draws": cfg.n_draws,
            "rho_candidates_median": float(np.median(rr["rho_c"])),
            "rho_candidates_min": float(rr["rho_c"].min()),
            "rho_candidates_p5": float(np.percentile(rr["rho_c"], 5)),
            "rho_grid_median": float(np.median(rr["rho_g"])),
            "hotspots": int(rr["hot0"].sum()),
            "hotspots_stable_90": int(rr["stable"].sum())},
        "candidates": {"n": int(len(surface["cand"])),
                       "pm_min": float(pm["pm_c"].min()),
                       "pm_max": float(pm["pm_c"].max()),
                       "cv": float(pm["pm_c"].std() / pm["pm_c"].mean())},
        "community_validation": community,
    }
    summary_path = cfg.pm_summary_path or os.path.join(
        out_dir, "pm_map_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Saved", summary_path)

    if make_plots:
        make_figures(surface, pm, cfg, model, rr, community, out_dir)

    return dict(out=out, out_csv=out_csv, summary=summary,
                summary_path=summary_path, model=model, surface=surface,
                pm=pm, robustness=rr, community=community,
                pm_series=pm_series, station_obs=station_obs)
