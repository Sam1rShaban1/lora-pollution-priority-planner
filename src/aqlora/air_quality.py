"""Air-quality layer: AQI->concentration conversion, XGBoost meteorological
baseline, and the physics-motivated PM2.5 monitoring-priority surface.

Key design point: the spatially-constant baseline CANCELS in any ranking, so
the pipeline runs for any city even with NO air-quality data (fallback
baseline from config). Training the XGBoost baseline on an aqicn daily export
+ meteostat weather refines the absolute scale and enables seasonal queries.
"""
import numpy as np
import pandas as pd

# US EPA breakpoints (2012 PM2.5 table used by aqicn exports)
BP_PM25 = [(0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4), (101, 150, 35.5, 55.4),
           (151, 200, 55.5, 150.4), (201, 300, 150.5, 250.4),
           (301, 400, 250.5, 350.4), (401, 500, 350.5, 500.4)]
BP_PM10 = [(0, 50, 0, 54), (51, 100, 55, 154), (101, 150, 155, 254),
           (151, 200, 255, 354), (201, 300, 355, 424),
           (301, 400, 425, 504), (401, 500, 505, 604)]


def aqi_to_concentration(aqi, breakpoints=BP_PM25):
    """Invert the EPA piecewise-linear AQI mapping (vectorized-friendly)."""
    def inv(a):
        if pd.isna(a):
            return np.nan
        for lo, hi, clo, chi in breakpoints:
            if a <= hi:
                return clo + (a - lo) * (chi - clo) / (hi - lo)
        return breakpoints[-1][3]
    if np.isscalar(aqi):
        return inv(aqi)
    return pd.Series(aqi).apply(inv).values


def audit_units(aq: pd.DataFrame, pm25_col="pm25", pm10_col="pm10") -> bool:
    """Detect AQI-encoded exports: PM2.5 > PM10 is impossible for
    concentrations but expected for AQI sub-indices."""
    a = pd.to_numeric(aq[pm25_col], errors="coerce")
    b = pd.to_numeric(aq[pm10_col], errors="coerce")
    both = a.notna() & b.notna()
    frac = float((a[both] > b[both]).mean()) if both.any() else 0.0
    is_aqi = frac > 0.5
    print(f"[aq] units audit: PM2.5>PM10 on {frac*100:.1f}% of days -> "
          f"{'AQI sub-indices (will convert)' if is_aqi else 'concentrations'}")
    return is_aqi


def train_baseline(aq_csv: str, weather_csv: str, seed: int = 42):
    """Train the XGBoost meteorological baseline on unit-corrected data.
    Returns (model, feature_frame, annual_baseline, metrics)."""
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    w = pd.read_csv(weather_csv).rename(columns={"time": "date"})
    w["date"] = pd.to_datetime(w.date)
    rules = {c: ("sum" if c in ("prcp", "snwd", "tsun") else "mean")
             for c in w.columns if c != "date"}
    daily = w.set_index("date").resample("D").agg(rules).reset_index().fillna(0)

    aq = pd.read_csv(aq_csv)
    aq.columns = [c.strip() for c in aq.columns]
    aq["date"] = pd.to_datetime(aq["date"], format="mixed", errors="coerce")
    for c in aq.columns:
        if c != "date":
            aq[c] = pd.to_numeric(aq[c], errors="coerce")
    if audit_units(aq):
        aq["pm25_conc"] = aqi_to_concentration(aq["pm25"], BP_PM25)
        aq["pm10_conc"] = aqi_to_concentration(aq["pm10"], BP_PM10)
    else:
        aq["pm25_conc"], aq["pm10_conc"] = aq["pm25"], aq["pm10"]

    m = daily.merge(aq, on="date").dropna(subset=["pm25_conc", "pm10_conc"])
    feats = [c for c in daily.columns if c != "date"] + \
            [c for c in ("pm10_conc", "o3", "no2", "so2", "co") if c in m.columns]
    X, y = m[feats].fillna(0), m["pm25_conc"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = XGBRegressor(n_estimators=100, objective="reg:squarederror",
                         random_state=seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = dict(r2=float(r2_score(yte, pred)),
                   rmse=float(mean_squared_error(yte, pred) ** 0.5),
                   n_days=len(m))
    baseline = float(model.predict(X.mean().to_frame().T)[0])
    print(f"[aq] XGBoost baseline: {baseline:.2f} ug/m3 "
          f"(R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}, n={metrics['n_days']})")
    return model, m[["date"] + feats], baseline, metrics


def priority_surface(feat: pd.DataFrame, baseline: float, frac: dict, decay: dict) -> np.ndarray:
    """Compose the PM2.5 priority values at candidate locations.
    Peak amplitudes are FRACTIONS of the baseline (units-coherent)."""
    Z = feat["elevation"].values.astype(float)
    Z = np.where(Z <= 0, np.median(Z[Z > 0]), Z)          # guard bad DEM samples
    zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    S = feat["slope"].values.astype(float)
    sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    dens = feat["C_dens"].values.astype(float)
    pm = baseline * (1.0
                     + frac["road"] * np.exp(-feat["d_road"].values / decay["road"])
                     + frac["ind"] * np.exp(-feat["d_ind"].values / decay["ind"])
                     + frac["dens"] * dens / max(dens.max(), 1)
                     + frac["elev"] * zn
                     + frac["slope"] * sn
                     + frac["fol"] * np.exp(-feat["d_fol"].values / decay["fol"]))
    return np.clip(pm, 0, None)


def hotspot_weights(pm: np.ndarray, quantile: float = 0.75, gamma: float = 4.0) -> np.ndarray:
    """Contrast-enhanced hotspot weights (Eq. weights in the paper).
    gamma is a threshold, not a tuning knob: any gamma >= 4 yields the same
    placement in our experiments."""
    topq = pm >= np.quantile(pm, quantile)
    w = 1.0 + gamma * topq
    return w / w.sum()


def sensitivity_analysis(feat: pd.DataFrame, baseline: float, frac: dict, decay: dict,
                         n_draws: int = 1000, seed: int = 0) -> dict:
    """Joint Monte Carlo perturbation of all forcing parameters; reports rank
    and hotspot-set stability (Sec. IV-C of the paper)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    base = priority_surface(feat, baseline, frac, decay)
    tb = base >= np.quantile(base, 0.75)
    rhos, jacs = [], []
    for _ in range(n_draws):
        f2 = {k: v * rng.uniform(0.5, 1.5) for k, v in frac.items()}
        d2 = {k: v * rng.uniform(0.5, 1.5) for k, v in decay.items()}
        p = priority_surface(feat, baseline, f2, d2)
        rhos.append(spearmanr(base, p).statistic)
        tp = p >= np.quantile(p, 0.75)
        jacs.append((tb & tp).sum() / (tb | tp).sum())
    rhos, jacs = np.array(rhos), np.array(jacs)
    out = dict(spearman_median=float(np.median(rhos)),
               spearman_p5=float(np.percentile(rhos, 5)),
               spearman_min=float(rhos.min()),
               jaccard_median=float(np.median(jacs)),
               jaccard_p5=float(np.percentile(jacs, 5)),
               frac_rho_gt_09=float((rhos > 0.9).mean()))
    print(f"[aq] sensitivity: median rho={out['spearman_median']:.3f}, "
          f"5th pct={out['spearman_p5']:.3f}, draws rho>0.9: {out['frac_rho_gt_09']*100:.0f}%")
    return out
