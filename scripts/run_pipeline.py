#!/usr/bin/env python3
"""End-to-end pipelines for the LoRa pollution-priority planner.

The canonical notebook-equivalent stages are ``joint`` (NSGA-II sensor +
gateway placement) and ``pm-map`` (the v2 PM2.5 priority surface).  The
original open-data gateway-only pipeline remains available as ``gateway``.

Usage:
    python scripts/run_pipeline.py --config configs/tetovo.yaml
    python scripts/run_pipeline.py --config configs/tetovo.yaml --stage joint
    python scripts/run_pipeline.py --config configs/tetovo.yaml --stage pm-map
    python scripts/run_pipeline.py --config configs/tetovo.yaml --stage gateway
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
from aqlora import joint, pm_map


def _require(value, name):
    if not value:
        raise SystemExit(f"Configuration value '{name}' is required for this stage.")
    return value


def run_legacy(cfg, args):
    """Original open-data, gateway-only pipeline."""
    # Lazy import: the notebook-equivalent joint stage should not require the
    # full OSM/geospatial stack.
    from aqlora import acquire, air_quality, optimize, propagation

    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()

    layers = acquire.fetch_layers(cfg.place, cfg.utm_crs)
    if cfg.candidates_csv:
        cand = pd.read_csv(cfg.candidates_csv)
        print(f"[pipeline] loaded {len(cand)} candidates from {cfg.candidates_csv}")
    else:
        cand = acquire.generate_candidates(
            layers, cfg.utm_crs, cfg.n_candidates,
            cfg.candidate_max_road_dist_m, cfg.candidate_min_spacing_m, cfg.seed)
    feat = acquire.candidate_features(cand, layers, cfg.utm_crs)

    baseline = cfg.pm_baseline
    if cfg.aq_csv and cfg.weather_csv:
        _, _, baseline, _ = air_quality.train_baseline(
            cfg.aq_csv, cfg.weather_csv, cfg.seed)
    else:
        print(f"[aq] no AQ/weather data configured -> fallback baseline "
              f"{baseline} ug/m3 (note: rankings are baseline-invariant)")
    pm = air_quality.priority_surface(feat, baseline, cfg.frac, cfg.decay_m)
    feat["priority_pm"] = pm
    weights = air_quality.hotspot_weights(pm, cfg.hotspot_quantile, cfg.gamma)
    sens = air_quality.sensitivity_analysis(feat, baseline, cfg.frac, cfg.decay_m)

    heights = acquire.building_heights(layers["buildings"],
                                       cfg.default_building_height_m)
    print("[radio] computing obstruction matrix ...")
    L = propagation.obstruction_matrix(
        cand, layers["buildings"], heights, cfg.utm_crs,
        cfg.min_building_height_m, cfg.obstruction_db_per_m,
        cfg.obstruction_cap_db)
    mu = propagation.mean_rssi(cand, L, cfg.utm_crs, cfg.tx_dbm,
                               cfg.ant_gain_db, cfg.freq_hz, cfg.rssi_bias_db)
    rel_cf, sf_cf = propagation.closed_form(mu, cfg.sens_dbm,
                                            cfg.shadowing_sigma_db)
    if args.skip_mc:
        rel, sf = rel_cf, sf_cf
    else:
        rel, sf = propagation.monte_carlo(
            mu, cfg.sens_dbm, cfg.shadowing_sigma_db, cfg.mc_iters,
            use_gpu=not args.no_gpu)
        err = np.abs(rel - rel_cf).max()
        print(f"[radio] MC vs closed-form oracle: max |delta reliability| = {err:.4f}")

    print("[optimize] NSGA-II (priority-weighted) ...")
    sel_w = optimize.nsga2_placement(
        rel, sf, cfg.k_gateways, weights, cfg.reliability_threshold,
        cfg.pop_size, cfg.generations, cfg.seed)
    print("[optimize] NSGA-II (uniform) ...")
    sel_u = optimize.nsga2_placement(
        rel, sf, cfg.k_gateways, None, cfg.reliability_threshold,
        cfg.pop_size, cfg.generations, cfg.seed)
    sel_g = optimize.greedy_placement(
        rel, cfg.k_gateways, weights, cfg.reliability_threshold)

    report = dict(
        place=cfg.place, k=cfg.k_gateways, rssi_bias_db=cfg.rssi_bias_db,
        baseline_ugm3=baseline, sensitivity=sens,
        priority_weighted=optimize.evaluate(sel_w, rel, sf, pm,
                                            cfg.hotspot_quantile),
        uniform=optimize.evaluate(sel_u, rel, sf, pm, cfg.hotspot_quantile),
        greedy_weighted=optimize.evaluate(sel_g, rel, sf, pm,
                                          cfg.hotspot_quantile),
        runtime_s=round(time.time() - t0, 1),
    )
    feat.to_csv(os.path.join(cfg.out_dir, "candidates_with_priority.csv"),
                index=False)
    np.savetxt(os.path.join(cfg.out_dir, "reliability_matrix.csv"),
               rel, delimiter=",")
    np.savetxt(os.path.join(cfg.out_dir, "sf_matrix.csv"), sf, delimiter=",")
    with open(os.path.join(cfg.out_dir, "deployment_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("priority_weighted", "uniform", "greedy_weighted")},
                     indent=2))
    print(f"[pipeline] done in {report['runtime_s']}s -> {cfg.out_dir}/")
    return report


def run_joint_stage(cfg, args):
    rel_csv = _require(cfg.rel_csv, "rel_csv")
    sf_csv = _require(cfg.sf_csv, "sf_csv")
    loc_csv = _require(cfg.candidates_pm_csv, "candidates_pm_csv")
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    pop = args.pop or cfg.pop_size
    gens = args.gens or cfg.generations
    run_delta = cfg.run_delta_sweep and not args.no_sweeps
    run_budget = cfg.run_budget_sweep and not args.no_sweeps
    result = joint.run_joint(
        rel_csv=rel_csv, sf_csv=sf_csv, loc_csv=loc_csv,
        mean_rssi_csv=cfg.mean_rssi_csv,
        num_sensors=cfg.num_sensors, num_gateways=cfg.k_gateways,
        delta_db=cfg.delta_db,
        reliability_threshold=cfg.reliability_threshold,
        shadowing_sigma_db=cfg.shadowing_sigma_db,
        sf_penalty=cfg.sf_penalty,
        hotspot_quantile=cfg.hotspot_quantile,
        hotspot_gamma=cfg.gamma,
        geo_weight=cfg.geo_weight,
        use_hotspot_prob=cfg.use_hotspot_prob,
        pm_col=cfg.pm_col,
        allow_colocation=cfg.allow_colocation,
        population_size=pop, generations=gens, seed=cfg.seed,
        selection_weights=tuple(cfg.selection_weights),
        run_baselines_flag=cfg.run_baselines and not args.no_baselines,
        out_dir=out_dir,
        tag=(f"M{cfg.num_sensors}_K{cfg.k_gateways}_delta{cfg.delta_db}"
             f"_deltaS_{run_delta}_budgetS_{run_budget}"
             f"_geoW_{cfg.geo_weight}_hotspotP_{cfg.use_hotspot_prob}"),
        verbose=args.verbose,
        use_gpu=not args.no_gpu,
        make_plots=not args.no_plots,
    )
    ctx, res = result["ctx"], result["res"]
    print("\n" + "=" * 60)
    print(f"JOINT DEPLOYMENT  (M = {cfg.num_sensors} sensors, "
          f"K = {cfg.k_gateways} gateways, delta = {cfg.delta_db} dB)")
    print("=" * 60)
    print("Sensor sites :", joint.ids_of(ctx, res["S"]))
    print("Gateway sites:", joint.ids_of(ctx, res["G"]))
    print(result["summary"].T)

    if result["baselines"] is not None:
        print("\nBaselines:")
        print(result["baselines"])

    if run_delta:
        print("\nDelta sweep:")
        delta_df = joint.delta_sweep(
            ctx, cfg.num_sensors, cfg.k_gateways,
            cfg.delta_list, cfg.delta_ref, cfg.sweep_pop, cfg.sweep_gens,
            cfg.seed, cfg.allow_colocation)
        path = os.path.join(out_dir,
                            f"delta_sweep_M{cfg.num_sensors}_K{cfg.k_gateways}.csv")
        delta_df.to_csv(path)
        print(delta_df)
        print("Saved", path)

    if run_budget:
        print("\nBudget sweep:")
        df_k, df_m = joint.budget_sweep(
            ctx, cfg.num_sensors, cfg.k_list, cfg.m_list,
            cfg.k_gateways, cfg.sweep_pop, cfg.sweep_gens, cfg.seed,
            cfg.allow_colocation)
        df_k.to_csv(os.path.join(out_dir, "budget_sweep_K.csv"))
        df_m.to_csv(os.path.join(out_dir, "budget_sweep_M.csv"))
        print(df_k)
        print(df_m)

    print(f"\nJoint pipeline finished in {result['elapsed_s']:.1f}s")
    if result["out_paths"]:
        print("Saved deployment:", result["out_paths"][0], result["out_paths"][1])
    return result


def run_pm_map_stage(cfg, args):
    os.makedirs(cfg.out_dir, exist_ok=True)
    result = pm_map.run_pm_map(cfg, out_dir=cfg.out_dir,
                               make_plots=not args.no_plots)
    print(f"PM-map finished: {result['out_csv']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["joint", "pm-map", "gateway", "all"],
                    default="joint",
                    help="joint = notebook-equivalent optimizer (default); "
                         "gateway = original open-data pipeline")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--skip-mc", action="store_true",
                    help="legacy gateway stage: closed-form link model only")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stage == "joint":
        run_joint_stage(cfg, args)
    elif args.stage == "gateway":
        run_legacy(cfg, args)
    elif args.stage == "pm-map":
        run_pm_map_stage(cfg, args)
    else:
        run_pm_map_stage(cfg, args)
        run_joint_stage(cfg, args)


if __name__ == "__main__":
    main()
