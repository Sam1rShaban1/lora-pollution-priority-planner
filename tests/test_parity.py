import json
import os

import numpy as np
import pandas as pd

from aqlora import joint
from aqlora.config import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = load_config(os.path.join(ROOT, "configs", "tetovo.yaml"))


def test_pm_artifact_schema():
    df = pd.read_csv(os.path.join(ROOT, CFG.candidates_pm_csv))
    assert len(df) == 99
    assert "predicted_pm" in df.columns
    assert "hotspot_prob" in df.columns
    assert 23.7 < df["predicted_pm"].min() < 23.9
    assert 39.1 < df["predicted_pm"].max() < 39.4
    assert abs(df["predicted_pm"].mean() - 34.07) < 0.05
    assert abs(df["predicted_pm"].std() / df["predicted_pm"].mean() - 0.085) < 0.002
    assert int(df["hotspot"].sum()) == 25


def test_joint_context():
    data = joint.load_joint_data(
        os.path.join(ROOT, CFG.rel_csv), os.path.join(ROOT, CFG.sf_csv),
        os.path.join(ROOT, CFG.candidates_pm_csv),
        os.path.join(ROOT, CFG.mean_rssi_csv), CFG.pm_col,
        CFG.use_hotspot_prob, CFG.hotspot_quantile, CFG.gamma, CFG.geo_weight)
    assert data["N"] == 99
    assert data["PM_NAME"] == "predicted_pm"
    assert data["HAVE_RSSI"]
    assert abs(data["W_NP"].sum() - 1.0) < 1e-9
    assert int(data["IS_HOT"].sum()) == 25
    assert abs(data["PM_Q"] - 35.60) < 0.02

    r0, _ = joint.closed_form_rf(data["MEAN_RSSI"], 0.0, 8.0)
    off = ~np.eye(data["N"], dtype=bool)
    assert np.abs(r0 - data["R_RAW"])[off].max() < 0.01


def test_quick_joint_optimization():
    data = joint.load_joint_data(
        os.path.join(ROOT, CFG.rel_csv), os.path.join(ROOT, CFG.sf_csv),
        os.path.join(ROOT, CFG.candidates_pm_csv),
        os.path.join(ROOT, CFG.mean_rssi_csv), CFG.pm_col,
        CFG.use_hotspot_prob, CFG.hotspot_quantile, CFG.gamma, CFG.geo_weight)
    ctx = joint.make_context(data, CFG.delta_db, CFG.reliability_threshold,
                             CFG.shadowing_sigma_db, CFG.sf_penalty)
    res = joint.optimize(ctx, 5, 3, seed=1, pop=30, gens=3)
    assert res["S"].sum() == 5
    assert res["G"].sum() == 3
    assert not (res["S"] & res["G"]).any()
    F, cov = joint.evaluate_batch(ctx, res["S"][None, :], res["G"][None, :])
    assert F.shape == (1, 3)
    assert 0.0 <= float(cov[0]) <= 1.0


def test_config_roundtrip():
    assert CFG.num_sensors == 15
    assert CFG.k_gateways == 5
    assert CFG.delta_db == -12.4
    assert CFG.sens_dbm == [-123.0, -126.0, -129.0, -132.0, -134.5, -137.0]


def _ctx():
    data = joint.load_joint_data(
        os.path.join(ROOT, CFG.rel_csv), os.path.join(ROOT, CFG.sf_csv),
        os.path.join(ROOT, CFG.candidates_pm_csv),
        os.path.join(ROOT, CFG.mean_rssi_csv), CFG.pm_col,
        CFG.use_hotspot_prob, CFG.hotspot_quantile, CFG.gamma, CFG.geo_weight)
    return joint.make_context(data, CFG.delta_db, CFG.reliability_threshold,
                              CFG.shadowing_sigma_db, CFG.sf_penalty)


def test_notebook_fixed_plan_metrics():
    """The notebook's saved plan must reproduce its published metrics."""
    ctx = _ctx()
    idx = {n: i for i, n in enumerate(ctx["NODE_IDS"])}
    S = np.zeros(ctx["N"], bool)
    G = np.zeros(ctx["N"], bool)
    for i in ["5", "8", "11", "15", "33", "39", "40", "41", "58", "60",
              "61", "63", "65", "87", "99"]:
        S[idx[i]] = True
    for i in ["14", "31", "36", "64", "83"]:
        G[idx[i]] = True
    row = joint.describe(ctx, S, G, "notebook")
    assert row["Covered sensors (%)"] == 100.0
    assert row["Hotspot sensors (#)"] == 12
    assert row["Hotspot sensors covered (%)"] == 100.0
    assert row["Priority-wtd coverage (% of ideal)"] == 80.8
    assert row["Avg SF"] == 7.1
    assert row["Repr. error"] == 0.0866


def test_sweeps_smoke():
    """delta/budget/seed sweeps must run and report site sets."""
    ctx = _ctx()
    delta = joint.delta_sweep(ctx, 5, 3, [-12.4, -5], pop=20, gens=3)
    assert len(delta) == 2
    assert "Gateway sites" in delta.columns
    assert bool(delta.loc["delta=-12.4", "Same gateways as ref"])
    df_k, df_m = joint.budget_sweep(ctx, 5, [3, 4], [5, 10], k_fixed=3,
                                    pop=20, gens=3)
    assert len(df_k) == 2 and len(df_m) == 2
    pairs, fixed, _ = joint.seed_control(
        ctx, 5, 3, seeds=(42, 7), pop=20, gens=3, delta_list=[-12.4, -5])
    assert len(pairs) == 1 and len(fixed) == 2


def test_pm_reference_summary():
    with open(os.path.join(ROOT, "data", "tetovo", "pm_map_summary.json")) as f:
        summary = json.load(f)
    assert abs(summary["units"]["station_long_run_mean_ugm3"] - 35.3608) < 1e-3
    assert abs(summary["units"]["pm25_gt_pm10_share"] - 0.7132) < 1e-3
    assert summary["model"]["chosen"] == "XGBoost (tuned)"
    assert abs(summary["model"]["holdout"]["R2"] - 0.70876) < 1e-4
    assert abs(summary["spatial"]["station_factor"] - 1.54376) < 1e-4
    assert abs(summary["spatial"]["base_annual"] - 22.9056) < 1e-3
    assert abs(summary["robustness"]["rho_candidates_median"] - 0.96477) < 1e-4
    assert summary["robustness"]["hotspots"] == 25
