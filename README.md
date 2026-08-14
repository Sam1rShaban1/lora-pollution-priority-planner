# LoRa Pollution-Priority Planner

**Pollution-priority-aware LoRaWAN gateway placement for any city, from open data alone.**

This repository is the open-source artifact for the paper *"Robust Pollution-Priority-Aware
LoRaWAN Gateway Placement under 3D Propagation Uncertainty"* (Shabani & Selimi). It plans an
air-quality sensor network for a city that has (at most) a single reference monitoring station:

1. **PM2.5 monitoring-priority surface** — an XGBoost meteorological baseline (trained on a
   units-audited public record) extended city-wide by six physics-motivated spatial corrections.
   We treat the result as a robust *ranking*, not a calibrated map: the induced candidate ranking
   retains Spearman ρ ≥ 0.85 under simultaneous ±50% perturbation of every hand-chosen parameter.
2. **3D digital twin & radio simulation** — building obstruction from OSM footprints
   (or your own DSM/DTM fusion), free-space path loss + log-normal shadowing, evaluated by both a
   closed-form oracle and a Monte Carlo engine (NumPy/CuPy) that generalizes to correlated and
   non-Gaussian fading.
3. **Priority-weighted placement** — a coverage-consistent NSGA-II objective in which pollution
   priority appears *explicitly*; benchmarked against uniform-grid, topographic, and greedy
   set-cover baselines, and evaluated under propagation-model uncertainty (RSSI bias sweep).

## Run it on any city

```bash
pip install -r requirements.txt

# the paper's study area (Tetovo, North Macedonia), calibrated operating point:
python scripts/run_pipeline.py --config configs/tetovo.yaml

# any other city: copy configs/template.yaml, change `place`, and run
python scripts/run_pipeline.py --config configs/my_city.yaml
```

The pipeline needs **zero credentials and zero local data** for a new city: road/industrial/
foliage/building layers come from OpenStreetMap, elevation and slope from AWS Terrain Tiles,
and candidate sensor locations are auto-generated (near-road grid inside the city boundary,
minimum spacing enforced). Air-quality and weather records are *optional* refinements — the
priority **ranking is invariant to the spatially-constant baseline**, so placement decisions
are valid even without them.

Outputs (in `outputs/`): candidate list with priority values, reliability & SF matrices,
and a `deployment_report.json` comparing priority-weighted NSGA-II, uniform NSGA-II, and
greedy set-cover placements (coverage, hotspot coverage, average SF).

## Reproducing the paper's experiments

| Paper section | What | How |
|---|---|---|
| IV — units audit | aqicn exports are AQI sub-indices, not µg/m³ | `air_quality.audit_units` (PM2.5 > PM10 on 99% of days) + EPA breakpoint inversion |
| IV-B — priority surface | baseline-relative spatial forcing | `air_quality.priority_surface` |
| IV-C — robustness | ±50% joint MC over all 9 parameters | `air_quality.sensitivity_analysis` (1,000 draws) |
| V — link model | closed-form oracle vs Monte Carlo engine | `propagation.closed_form` / `propagation.monte_carlo` |
| V-D — weighted objective | hotspot weights (γ threshold) + coverage-consistent f1/f2 | `air_quality.hotspot_weights`, `optimize.GatewayPlacement` |
| VI-D — calibrated evaluation | −12.4 dB operating point / bias sweep | set `rssi_bias_db` in the config and re-run |
| VI — baselines | greedy weighted max-coverage | `optimize.greedy_placement` |

The paper's exact input data for Tetovo ships in `data/tetovo/`: the 99 surveyed candidate
locations (with the corrected node-46 elevation), the MOEPP/aqicn daily record (AQI-encoded —
the pipeline audits and converts it), and the meteostat hourly weather export.

Set `rssi_bias_db: 0` for the optimistic radio model or `-12.4` for the field-calibrated
operating point used in the paper. Sweep it over `{0, -5, -10, -12.4, -15, -20}` to reproduce
the robustness analysis.

## Fading models (why the Monte Carlo engine exists)

`aqlora.fading` provides samplers for models with **no closed-form** link statistics:
spatially **correlated shadowing** (Gudmundson, exponential decorrelation),
**Rician multipath** (K-factor), and an **empirical** sampler that bootstraps
field-measured RSSI residuals. The i.i.d. Gaussian case has an exact closed form
(`propagation.closed_form`), which the engine uses as a correctness oracle; the Monte
Carlo engine is what makes the richer models tractable, on CPU or GPU:

```python
from aqlora import propagation, fading
sampler = fading.correlated_shadowing(coords_xy, sigma_db=8.0, decorrelation_m=150)
rel, sf = propagation.monte_carlo(mu, sens, 8.0, sampler=sampler)  # GPU if cupy present
```

## Notes & provenance

- **Building heights:** by default estimated from OSM `height`/`building:levels` tags with a
  configurable fallback. The paper's Tetovo study used an ALOS PALSAR DSM + FABDEM DTM fusion
  over Microsoft ML footprints; those rasters require (free) registration, so the repo ships the
  OSM path for out-of-the-box replication and documents the fusion as the higher-fidelity option.
- **Air-quality data:** aqicn.org daily exports are AQI sub-indices. The pipeline detects this
  (PM2.5 > PM10 test) and inverts the EPA breakpoints before any training. All model outputs
  are µg/m³.
- **Hardware model:** LilyGO T-Beam v1.2 / Semtech SX1276, EU 868 MHz (Table I of the paper);
  all link-budget parameters are configurable.
- **GPU:** install `cupy-cuda12x` to run the Monte Carlo engine on GPU; the pipeline falls back
  to NumPy (and the closed-form oracle via `--skip-mc` is exact and instant for the i.i.d.
  Gaussian model).

## Citation

```bibtex
@inproceedings{shabani2027lora,
  title     = {Robust Pollution-Priority-Aware LoRaWAN Gateway Placement
               under 3D Propagation Uncertainty},
  author    = {Shabani, Samir and Selimi, Mennan},
  year      = {2027},
  note      = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
