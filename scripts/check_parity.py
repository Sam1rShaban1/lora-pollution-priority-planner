#!/usr/bin/env python3
"""Validate the repository against the notebook reference outputs.

Checks the committed v2 PM artifact, the precomputed link matrices, and a
seeded joint NSGA-II run.  The expected values are taken from
``NSGA_2_LoRa_joint.ipynb`` / ``synthetic_pm_map_v2.ipynb``.

Usage:
    python scripts/check_parity.py --config configs/tetovo.yaml
    python scripts/check_parity.py --config configs/tetovo.yaml --quick
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aqlora.config import load_config
from aqlora import joint


EXPECTED_SENSORS = {"5", "8", "11", "15", "33", "39", "40", "41", "58",
                    "60", "61", "63", "65", "87", "99"}
EXPECTED_GATEWAYS = {"14", "31", "36", "64", "83"}


class Checks:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, name, condition, detail=""):
        if condition:
            self.passes += 1
            print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failures.append(name)
            print(f"FAIL  {name}" + (f"  [{detail}]" if detail else ""))

    def close(self, name, value, expected, tol, detail=""):
        self.check(name, abs(float(value) - float(expected)) <= tol,
                   detail or f"{value:.4f} vs {expected} +/- {tol}")


def check_pm_artifact(cfg, checks):
    path = cfg.candidates_pm_csv
    checks.check("candidate_locations_with_pm.csv exists", bool(path) and os.path.exists(path), path)
    if not path or not os.path.exists(path):
        return
    df = pd.read_csv(path)
    expected_cols = [
        "id", "latitude", "longitude", "elevation", "elevation_dem",
        "predicted_pm", "pm_p05", "pm_p95", "hotspot", "hotspot_prob",
        "spatial_factor", "pm_winter", "pm_summer", "term_road",
        "term_industrial", "term_density", "term_elevation", "term_slope",
        "term_foliage"]
    checks.check("PM artifact has the v2 schema", list(df.columns) == expected_cols,
                 f"{len(df.columns)} columns")
    checks.check("PM artifact has 99 candidates", len(df) == 99, str(len(df)))
    pm = df["predicted_pm"].astype(float)
    checks.close("PM mean", pm.mean(), 34.07, 0.05)
    checks.close("PM min", pm.min(), 23.78, 0.05)
    checks.close("PM max", pm.max(), 39.24, 0.05)
    checks.close("PM CV", pm.std() / pm.mean(), 0.085, 0.002)
    checks.check("hotspot_prob column present", "hotspot_prob" in df.columns)
    if "hotspot_prob" in df.columns:
        hp = df["hotspot_prob"].astype(float)
        checks.check("hotspot_prob in [0,1]", bool((hp >= 0).all() and (hp <= 1).all()),
                     f"range {hp.min():.2f}-{hp.max():.2f}")
    checks.check("25 hotspot candidates", int(df["hotspot"].sum()) == 25,
                 str(int(df["hotspot"].sum())))


def check_link_matrices(cfg, checks):
    data = joint.load_joint_data(
        cfg.rel_csv, cfg.sf_csv, cfg.candidates_pm_csv, cfg.mean_rssi_csv,
        cfg.pm_col, cfg.use_hotspot_prob, cfg.hotspot_quantile, cfg.gamma,
        cfg.geo_weight)
    checks.check("matrix node count is 99", data["N"] == 99, str(data["N"]))
    checks.check("PM column auto-detected as predicted_pm",
                 data["PM_NAME"] == "predicted_pm", data["PM_NAME"])
    checks.check("mean-RSSI matrix available", data["HAVE_RSSI"])
    checks.close("PM range min", data["PM"].min(), 23.78, 0.05)
    checks.close("PM range max", data["PM"].max(), 39.24, 0.05)
    checks.close("hotspot threshold", data["PM_Q"], 35.60, 0.02,
                 f"{data['PM_Q']:.4f}")
    checks.check("25 hotspot sites", int(data["IS_HOT"].sum()) == 25,
                 str(int(data["IS_HOT"].sum())))
    checks.check("priority weights sum to one",
                 abs(float(data["W_NP"].sum()) - 1.0) < 1e-9)

    # Notebook cell 4: closed form vs saved Monte Carlo at delta = 0.
    r0, _ = joint.closed_form_rf(data["MEAN_RSSI"], 0.0,
                                 cfg.shadowing_sigma_db)
    off = ~np.eye(data["N"], dtype=bool)
    max_d = float(np.abs(r0 - data["R_RAW"])[off].max())
    mean_d = float(np.abs(r0 - data["R_RAW"])[off].mean())
    checks.check("closed-form vs saved MC max |dR| <= 0.01", max_d <= 0.01,
                 f"{max_d:.4f}")
    checks.check("closed-form vs saved MC mean |dR| <= 0.001", mean_d <= 0.001,
                 f"{mean_d:.5f}")
    return data


def _plan_mask(ctx, ids):
    mask = np.zeros(ctx["N"], bool)
    idx = {n: i for i, n in enumerate(ctx["NODE_IDS"])}
    for i in ids:
        mask[idx[i]] = True
    return mask


def check_notebook_fixed_plan(cfg, data, checks):
    """Deterministic parity: the notebook's saved plan must reproduce its metrics.

    This is independent of the optimizer's search trajectory, so it holds on
    both CPU and GPU.  It is the strongest exact check in this harness.
    """
    ctx = joint.make_context(
        data, cfg.delta_db, cfg.reliability_threshold,
        cfg.shadowing_sigma_db, cfg.sf_penalty)
    S = _plan_mask(ctx, EXPECTED_SENSORS)
    G = _plan_mask(ctx, EXPECTED_GATEWAYS)
    row = joint.describe(ctx, S, G, "Notebook fixed plan")
    checks.check("fixed-plan sensor count", int(S.sum()) == 15, str(int(S.sum())))
    checks.check("fixed-plan gateway count", int(G.sum()) == 5, str(int(G.sum())))
    checks.close("fixed-plan covered sensors (%)",
                 row["Covered sensors (%)"], 100.0, 1e-9)
    checks.check("fixed-plan hotspot sensors (#)",
                 row["Hotspot sensors (#)"] == 12, str(row["Hotspot sensors (#)"]))
    checks.close("fixed-plan hotspot coverage (%)",
                 row["Hotspot sensors covered (%)"], 100.0, 1e-9)
    checks.close("fixed-plan priority-weighted coverage (%)",
                 row["Priority-wtd coverage (% of ideal)"], 80.8, 0.05)
    checks.close("fixed-plan average SF", row["Avg SF"], 7.10, 0.005)
    checks.close("fixed-plan representativeness error",
                 row["Repr. error"], 0.0866, 1e-4)
    print("\nNotebook fixed-plan summary:")
    print(pd.Series(row).to_string())
    return ctx, S, G


def check_joint_run(cfg, data, checks, quick=False, strict=False):
    ctx = joint.make_context(
        data, cfg.delta_db, cfg.reliability_threshold,
        cfg.shadowing_sigma_db, cfg.sf_penalty)
    pop = 80 if quick else cfg.pop_size
    gens = 10 if quick else cfg.generations
    res = joint.optimize(
        ctx, cfg.num_sensors, cfg.k_gateways, seed=cfg.seed, pop=pop,
        gens=gens, selection_weights=tuple(cfg.selection_weights),
        allow_colocation=cfg.allow_colocation)
    row = joint.describe(ctx, res["S"], res["G"], "Proposed")
    sensors = set(joint.ids_of(ctx, res["S"]))
    gateways = set(joint.ids_of(ctx, res["G"]))

    checks.check("optimizer returns M sensors", int(res["S"].sum()) == cfg.num_sensors,
                 str(int(res["S"].sum())))
    checks.check("optimizer returns K gateways",
                 int(res["G"].sum()) == cfg.k_gateways, str(int(res["G"].sum())))
    checks.check("sensors and gateways are disjoint",
                 not bool((res["S"] & res["G"]).any()))
    if quick:
        print("Quick mode: skipped coverage and exact site/metric assertions.")
    else:
        checks.close("optimizer converges to full sensor coverage",
                     row["Covered sensors (%)"], 100.0, 1e-9)

    if strict and not quick:
        checks.check("selected sensor set matches saved GPU notebook run",
                     sensors == EXPECTED_SENSORS,
                     f"{sorted(sensors)} expected {sorted(EXPECTED_SENSORS)}")
        checks.check("selected gateway set matches saved GPU notebook run",
                     gateways == EXPECTED_GATEWAYS,
                     f"{sorted(gateways)} expected {sorted(EXPECTED_GATEWAYS)}")
        checks.close("priority-weighted coverage (% of ideal)",
                     row["Priority-wtd coverage (% of ideal)"], 80.8, 0.05)
        checks.close("average SF", row["Avg SF"], 7.1, 0.005)
        checks.close("representativeness error", row["Repr. error"], 0.0866, 1e-4)
    else:
        print("Note: exact site sets are only asserted with --strict. The saved "
              "notebook run used CuPy/GPU, so CPU NSGA-II explores a different "
              "Pareto frontier. Algorithm and metric parity is proven by the "
              "fixed-plan checks above.")
        print(f"Optimizer sensors  : {sorted(sensors)}")
        print(f"Optimizer gateways : {sorted(gateways)}")

    if quick:
        print("Quick mode: skipped exact site/metric assertions.")
    print("\nOptimizer run summary:")
    print(pd.Series(row).to_string())
    return ctx, res


def deployment_tag(cfg):
    """Notebook tag for saved deployments (cell 10 of the joint notebook)."""
    return (f"M{cfg.num_sensors}_K{cfg.k_gateways}_delta{cfg.delta_db}"
            f"_deltaS_{cfg.run_delta_sweep}_budgetS_{cfg.run_budget_sweep}"
            f"_geoW_{cfg.geo_weight}_hotspotP_{cfg.use_hotspot_prob}")


def check_deployment_outputs(cfg, ctx, res, checks):
    out_dir = cfg.out_dir
    tag = deployment_tag(cfg)
    csv_path = os.path.join(out_dir, f"joint_deployment_{tag}.csv")
    json_path = os.path.join(out_dir, f"joint_deployment_{tag}.json")
    checks.check("deployment CSV written", os.path.exists(csv_path), csv_path)
    checks.check("deployment JSON written", os.path.exists(json_path), json_path)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"id": str})
        checks.check("deployment CSV has 20 selected sites", len(df) == 20,
                     str(len(df)))
        checks.check("deployment roles are valid",
                     set(df["role"]) <= {"sensor", "gateway", "sensor+gateway"},
                     str(sorted(set(df["role"]))))
        checks.check("deployment sensors match selection",
                     set(df.loc[df["role"].isin(["sensor", "sensor+gateway"]), "id"])
                     == set(joint.ids_of(ctx, res["S"])))
        checks.check("deployment gateways match selection",
                     set(df.loc[df["role"].isin(["gateway", "sensor+gateway"]), "id"])
                     == set(joint.ids_of(ctx, res["G"])))
    if os.path.exists(json_path):
        with open(json_path) as f:
            summary = json.load(f)
        for key in ("M", "K", "delta_db", "sensors", "gateways", "metrics"):
            checks.check(f"deployment JSON has '{key}'", key in summary)
        checks.check("deployment JSON sensor list matches",
                     set(summary.get("sensors", [])) == set(joint.ids_of(ctx, res["S"])))
        checks.check("deployment JSON gateway list matches",
                     set(summary.get("gateways", [])) == set(joint.ids_of(ctx, res["G"])))


def _same_nested(a, b, atol=1e-9, rtol=1e-6):
    """Recursively compare JSON-like structures, tolerating float noise."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same_nested(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _same_nested(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    return a == b


def check_pm_summary(cfg, checks):
    """Check the canonical notebook summary and, if present, a fresh rerun.

    ``data/tetovo/pm_map_summary.json`` is the summary saved by the reference
    notebook run.  A rerun written to ``cfg.pm_summary_path`` is compared on
    everything that does not depend on the weather model: the weather-model
    metrics differ when the available weather CSV lacks dewpoint/hourly gust
    columns (the notebook's Drive copy had ``dwpt`` and ``wpgt``), while the
    station-anchored spatial surface and robustness draws are identical.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    canonical = os.path.join(repo, "data", "tetovo", "pm_map_summary.json")
    path = canonical if os.path.exists(canonical) else None
    if path is None:
        path = next((p for p in (cfg.pm_summary_path,
                                 os.path.join(cfg.out_dir, "pm_map_summary.json"))
                     if p and os.path.exists(p)), None)
    if path is None:
        print("SKIP  pm_map_summary.json not present.")
        return
    with open(path) as f:
        summary = json.load(f)
    print(f"PM reference summary: {os.path.relpath(path, repo)}")
    checks.check("PM summary has units/model/robustness",
                 all(k in summary for k in ("units", "model", "robustness")))
    units = summary.get("units", {})
    checks.close("PM summary station mean",
                 units.get("station_long_run_mean_ugm3", np.nan), 35.36, 0.05)
    checks.close("PM summary AQI share",
                 units.get("pm25_gt_pm10_share", np.nan), 0.713, 0.005)
    checks.check("PM summary treats record as AQI",
                 units.get("treated_as_aqi") is True)
    model = summary.get("model", {})
    checks.check("PM summary chose XGBoost",
                 model.get("chosen") == "XGBoost (tuned)", str(model.get("chosen")))
    checks.close("PM summary holdout R2",
                 model.get("holdout", {}).get("R2", np.nan), 0.709, 0.005)
    checks.close("PM summary holdout RMSE",
                 model.get("holdout", {}).get("RMSE", np.nan), 12.84, 0.05)
    spatial = summary.get("spatial", {})
    checks.close("PM summary station factor",
                 spatial.get("station_factor", np.nan), 1.544, 0.002)
    checks.close("PM summary annual baseline",
                 spatial.get("base_annual", np.nan), 22.91, 0.05)
    rob = summary.get("robustness", {})
    checks.close("PM summary candidate median rho",
                 rob.get("rho_candidates_median", np.nan), 0.965, 0.005)
    checks.close("PM summary candidate p5 rho",
                 rob.get("rho_candidates_p5", np.nan), 0.860, 0.005)
    checks.check("PM summary has 25 hotspots",
                 rob.get("hotspots") == 25, str(rob.get("hotspots")))

    # Optional: compare a fresh rerun on the invariant fields.
    rerun = cfg.pm_summary_path
    if rerun and os.path.exists(rerun) and os.path.abspath(rerun) != os.path.abspath(path):
        with open(rerun) as f:
            fresh = json.load(f)
        print(f"PM rerun summary:     {os.path.relpath(rerun, repo)}")
        for key in ("units", "spatial", "robustness", "candidates"):
            ok = _same_nested(fresh.get(key), summary.get(key))
            checks.check(f"PM rerun reproduces '{key}'", ok,
                         "" if ok else "differs beyond float tolerance")
        checks.check("PM rerun chose the same model",
                     fresh.get("model", {}).get("chosen")
                     == summary.get("model", {}).get("chosen"))
        feat_ref = summary.get("model", {}).get("features", [])
        feat_new = fresh.get("model", {}).get("features", [])
        if feat_ref != feat_new:
            print("NOTE  weather feature set differs (available CSV lacks "
                  f"{sorted(set(feat_ref) - set(feat_new))}); "
                  "seasonal scale and holdout metrics will differ.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="short optimization for smoke checks")
    ap.add_argument("--strict", action="store_true",
                    help="also assert exact optimizer site sets from the saved "
                         "GPU notebook run (only reproducible with CuPy/GPU)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg.out_dir, exist_ok=True)

    checks = Checks()
    check_pm_artifact(cfg, checks)
    data = check_link_matrices(cfg, checks)
    check_notebook_fixed_plan(cfg, data, checks)
    ctx, res = check_joint_run(cfg, data, checks, quick=args.quick,
                               strict=args.strict)
    # Write outputs so the schema checks exercise the real writer.
    joint.save_deployment(
        ctx, {**res, "pop_size": 80 if args.quick else cfg.pop_size,
              "generations": 10 if args.quick else cfg.generations,
              "seed": cfg.seed}, cfg.out_dir,
        tag=deployment_tag(cfg))
    check_deployment_outputs(cfg, ctx, res, checks)
    check_pm_summary(cfg, checks)

    print(f"\n{checks.passes} checks passed, {len(checks.failures)} failed.")
    if checks.failures:
        print("Failed:", checks.failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
