#!/usr/bin/env python3
"""End-to-end pipeline: pick a place, get a pollution-priority-aware LoRaWAN
gateway plan.

Usage:
    python scripts/run_pipeline.py --config configs/tetovo.yaml
    python scripts/run_pipeline.py --config configs/template.yaml  # any city

Stages:
 1. acquire   : OSM layers (roads/industry/foliage/buildings), terrain tiles
 2. candidates: load CSV or auto-generate near-road grid inside the boundary
 3. aq layer  : units-audited XGBoost baseline (optional) + priority surface
                + parameter sensitivity analysis
 4. radio     : obstruction matrix -> mean RSSI (with optional field bias)
                -> reliability & SF matrices (closed form + MC cross-check)
 5. optimize  : priority-weighted NSGA-II vs greedy set-cover
 6. report    : JSON + CSV outputs in cfg.out_dir
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aqlora.config import load_config
from aqlora import acquire, air_quality, propagation, optimize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--skip-mc", action="store_true",
                    help="use only the closed-form link model (fast)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()

    # 1-2. layers + candidates ------------------------------------------------
    layers = acquire.fetch_layers(cfg.place, cfg.utm_crs)
    if cfg.candidates_csv:
        cand = pd.read_csv(cfg.candidates_csv)
        print(f"[pipeline] loaded {len(cand)} candidates from {cfg.candidates_csv}")
    else:
        cand = acquire.generate_candidates(
            layers, cfg.utm_crs, cfg.n_candidates,
            cfg.candidate_max_road_dist_m, cfg.candidate_min_spacing_m, cfg.seed)
    feat = acquire.candidate_features(cand, layers, cfg.utm_crs)

    # 3. air-quality priority surface ----------------------------------------
    baseline = cfg.pm_baseline
    if cfg.aq_csv and cfg.weather_csv:
        _, _, baseline, metrics = air_quality.train_baseline(
            cfg.aq_csv, cfg.weather_csv, cfg.seed)
    else:
        print(f"[aq] no AQ/weather data configured -> fallback baseline "
              f"{baseline} ug/m3 (note: rankings are baseline-invariant)")
    pm = air_quality.priority_surface(feat, baseline, cfg.frac, cfg.decay_m)
    feat["priority_pm"] = pm
    weights = air_quality.hotspot_weights(pm, cfg.hotspot_quantile, cfg.gamma)
    sens = air_quality.sensitivity_analysis(feat, baseline, cfg.frac, cfg.decay_m)

    # 4. radio ---------------------------------------------------------------
    heights = acquire.building_heights(layers["buildings"], cfg.default_building_height_m)
    print("[radio] computing obstruction matrix ...")
    L = propagation.obstruction_matrix(
        cand, layers["buildings"], heights, cfg.utm_crs,
        cfg.min_building_height_m, cfg.obstruction_db_per_m, cfg.obstruction_cap_db)
    mu = propagation.mean_rssi(cand, L, cfg.utm_crs, cfg.tx_dbm,
                               cfg.ant_gain_db, cfg.freq_hz, cfg.rssi_bias_db)
    rel_cf, sf_cf = propagation.closed_form(mu, cfg.sens_dbm, cfg.shadowing_sigma_db)
    if args.skip_mc:
        rel, sf = rel_cf, sf_cf
    else:
        rel, sf = propagation.monte_carlo(mu, cfg.sens_dbm, cfg.shadowing_sigma_db,
                                          cfg.mc_iters, use_gpu=not args.no_gpu)
        err = np.abs(rel - rel_cf).max()
        print(f"[radio] MC vs closed-form oracle: max |delta reliability| = {err:.4f}")

    # 5. optimize -------------------------------------------------------------
    print("[optimize] NSGA-II (priority-weighted) ...")
    sel_w = optimize.nsga2_placement(rel, sf, cfg.k_gateways, weights,
                                     cfg.reliability_threshold,
                                     cfg.pop_size, cfg.generations, cfg.seed)
    print("[optimize] NSGA-II (uniform) ...")
    sel_u = optimize.nsga2_placement(rel, sf, cfg.k_gateways, None,
                                     cfg.reliability_threshold,
                                     cfg.pop_size, cfg.generations, cfg.seed)
    sel_g = optimize.greedy_placement(rel, cfg.k_gateways, weights,
                                      cfg.reliability_threshold)

    # 6. report ---------------------------------------------------------------
    report = dict(
        place=cfg.place, k=cfg.k_gateways, rssi_bias_db=cfg.rssi_bias_db,
        baseline_ugm3=baseline, sensitivity=sens,
        priority_weighted=optimize.evaluate(sel_w, rel, sf, pm, cfg.hotspot_quantile),
        uniform=optimize.evaluate(sel_u, rel, sf, pm, cfg.hotspot_quantile),
        greedy_weighted=optimize.evaluate(sel_g, rel, sf, pm, cfg.hotspot_quantile),
        runtime_s=round(time.time() - t0, 1),
    )
    feat.to_csv(os.path.join(cfg.out_dir, "candidates_with_priority.csv"), index=False)
    np.savetxt(os.path.join(cfg.out_dir, "reliability_matrix.csv"), rel, delimiter=",")
    np.savetxt(os.path.join(cfg.out_dir, "sf_matrix.csv"), sf, delimiter=",")
    with open(os.path.join(cfg.out_dir, "deployment_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("priority_weighted", "uniform", "greedy_weighted")}, indent=2))
    print(f"[pipeline] done in {report['runtime_s']}s -> {cfg.out_dir}/")


if __name__ == "__main__":
    main()
