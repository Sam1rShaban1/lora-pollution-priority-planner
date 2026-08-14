"""Radio propagation: geometric obstruction, FSPL, and link matrices.

Two evaluation engines over the same link model:
- closed_form(): exact reliability & expected SF under i.i.d. Gaussian
  shadowing (milliseconds; used as a correctness oracle)
- monte_carlo(): sampling engine (NumPy or CuPy), the general simulator that
  extends to correlated shadowing / empirical fading distributions
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.strtree import STRtree
from scipy.stats import norm

SFS = np.array([7, 8, 9, 10, 11, 12], dtype=float)


def obstruction_matrix(cand: pd.DataFrame, buildings: gpd.GeoDataFrame,
                       heights: np.ndarray, utm_crs: str,
                       min_h: float = 4.0, db_per_m: float = 0.8,
                       cap_db: float = 40.0) -> np.ndarray:
    """L_clutter[i,j]: penetration depth through buildings taller than min_h
    along the direct path, converted to dB and capped (diffraction proxy)."""
    pts = gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(
        cand.longitude, cand.latitude), crs="EPSG:4326").to_crs(utm_crs)
    P = [(g.x, g.y) for g in pts.geometry]
    tall = buildings[heights >= min_h]
    tree = STRtree(tall.geometry.values)
    n = len(P)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            line = LineString([P[i], P[j]])
            depth = 0.0
            for k in tree.query(line):
                inter = tall.geometry.values[k].intersection(line)
                if not inter.is_empty:
                    depth += inter.length
            L[i, j] = L[j, i] = min(db_per_m * depth, cap_db)
    return L


def mean_rssi(cand: pd.DataFrame, L_clutter: np.ndarray, utm_crs: str,
              tx_dbm: float, ant_gain_db: float, freq_hz: float,
              bias_db: float = 0.0) -> np.ndarray:
    """FSPL + obstruction -> mean RSSI matrix; bias_db applies a field-measured
    calibration offset (e.g. -12.4 dB)."""
    pts = gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(
        cand.longitude, cand.latitude), crs="EPSG:4326").to_crs(utm_crs)
    xy = np.array([(g.x, g.y) for g in pts.geometry])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, 1.0)
    fspl = 20 * np.log10(d) + 20 * np.log10(freq_hz) - 147.55
    rssi = tx_dbm + ant_gain_db - fspl - L_clutter + bias_db
    np.fill_diagonal(rssi, 0.0)
    return rssi


def closed_form(mu: np.ndarray, sens: list, sigma: float):
    """Exact per-link reliability and expected SF (i.i.d. Gaussian shadowing).
    Used as the correctness oracle for the Monte Carlo engine."""
    s = np.asarray(sens, dtype=float)
    p_ge = 1.0 - norm.cdf((s[None, None, :] - mu[:, :, None]) / sigma)
    rel = p_ge[:, :, -1].copy()
    probs = np.empty_like(p_ge)
    probs[:, :, 0] = p_ge[:, :, 0]
    probs[:, :, 1:] = p_ge[:, :, 1:] - p_ge[:, :, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = np.where(rel > 1e-9, (probs @ SFS) / rel, 20.0)
    np.fill_diagonal(rel, 1.0)
    np.fill_diagonal(sf, 0.0)
    return rel, sf


def monte_carlo(mu: np.ndarray, sens: list, sigma: float, n_iter: int = 10000,
                use_gpu: bool = True, seed: int = 1, sampler=None):
    """Monte Carlo link simulation (CuPy if available). Pass a fading sampler
    from aqlora.fading to run correlated / multipath / empirical models
    (none of which admits the closed form)."""
    xp = np
    if use_gpu:
        try:
            import cupy as xp  # noqa: F401
            import cupy
            xp = cupy
        except Exception:
            xp = np
    s = xp.asarray(np.asarray(sens, dtype=np.float32))
    mu_x = xp.asarray(mu.astype(np.float32))
    n = mu.shape[0]
    rel = xp.zeros((n, n), dtype=xp.float32)
    sf_sum = xp.zeros((n, n), dtype=xp.float32)
    rng = np.random.default_rng(seed)
    batch = max(1, min(n_iter, int(2e8 // (n * n))))
    done = 0
    while done < n_iter:
        b = min(batch, n_iter - done)
        if sampler is not None:
            fading = sampler(xp, n, b, rng if xp is np else None)
        elif xp is np:
            fading = rng.normal(0, sigma, size=(n, n, b)).astype(np.float32)
        else:
            fading = xp.random.normal(0, sigma, size=(n, n, b), dtype=xp.float32)
        rssi = mu_x[:, :, None] + fading
        ok = rssi >= s[-1]
        sf_idx = (rssi < s[0]).astype(xp.int8)
        for k in range(1, 5):
            sf_idx += (rssi < s[k]).astype(xp.int8)
        rel += ok.sum(axis=2)
        sf_sum += xp.where(ok, (7 + sf_idx).astype(xp.float32), 0).sum(axis=2)
        done += b
    with np.errstate(divide="ignore", invalid="ignore"):
        out_rel = rel / n_iter
        out_sf = xp.where(rel > 0, sf_sum / rel, 20.0)
    if xp is not np:
        out_rel, out_sf = out_rel.get(), out_sf.get()
    out_rel, out_sf = np.asarray(out_rel, dtype=float), np.asarray(out_sf, dtype=float)
    np.fill_diagonal(out_rel, 1.0)
    np.fill_diagonal(out_sf, 0.0)
    return out_rel, out_sf
