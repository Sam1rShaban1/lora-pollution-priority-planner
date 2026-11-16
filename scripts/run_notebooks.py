#!/usr/bin/env python3
r"""Execute a copy of a notebook against the repository data.

The committed notebooks keep their original Google-Colab paths.  This runner
copies the notebook to a temporary directory, rewrites only the configuration
cell in the copy, executes it, and leaves the committed notebook untouched.

Usage:
    python scripts/run_notebooks.py joint
    python scripts/run_notebooks.py joint --fast
    python scripts/run_notebooks.py pm-map --pm-raw-dir /path/to/drive/LoRa\ Placement
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data", "tetovo")


def _patch_cell(nb, replacements):
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_lines = []
        for line in src.splitlines():
            stripped = line.strip()
            for key, value in replacements.items():
                if stripped.startswith(key + " "):
                    line = f"{key} = {value}"
                    changed += 1
                    break
            new_lines.append(line)
        new_src = "\n".join(new_lines)
        if new_src != src:
            # nbformat stores source as a list of lines with newlines.
            cell["source"] = [ln + "\n" for ln in new_src.splitlines()]
    return changed


def _execute(path):
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as e:
        raise SystemExit(
            "Notebook execution needs nbclient/nbformat. "
            "Install with: pip install nbclient nbformat jupyter") from e
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path)}})
    client.execute()
    nbformat.write(nb, path)


def run_joint(fast=False):
    work = tempfile.mkdtemp(prefix="aqlora_joint_")
    files = [
        "link_reliability_matrix_gpu_realistic.csv",
        "link_average_sf_matrix_gpu_realistic.csv",
        "link_mean_rssi_matrix.csv",
        "candidate_locations_with_pm.csv",
    ]
    for name in files:
        shutil.copy(os.path.join(DATA, name), os.path.join(work, name))

    import nbformat
    src = os.path.join(REPO, "notebooks", "NSGA_2_LoRa_joint.ipynb")
    dst = os.path.join(work, "NSGA_2_LoRa_joint.ipynb")
    nb = nbformat.read(src, as_version=4)
    replacements = {"BASE_PATH": json.dumps(work + os.sep)}
    if fast:
        replacements.update({
            "POPULATION_SIZE": "100",
            "GENERATIONS": "20",
            "RUN_DELTA_SWEEP": "False",
            "RUN_BUDGET_SWEEP": "False",
        })
    _patch_cell(nb, replacements)
    nbformat.write(nb, dst)
    print(f"Executing joint notebook in {work} ...")
    _execute(dst)
    print("Notebook outputs:")
    for name in sorted(os.listdir(work)):
        if name.startswith("joint_deployment") or name.startswith("joint_pareto") \
                or name.startswith("joint_convergence") or name.startswith("delta_sweep"):
            print(" ", os.path.join(work, name))
    return work


def run_pm_map(raw_dir=None):
    if not raw_dir:
        raise SystemExit(
            "The full PM-map notebook needs the raw DEM/slope/GeoJSON inputs. "
            "Pass --pm-raw-dir pointing at the Drive folder, or validate the "
            "saved artifacts with scripts/check_parity.py instead.")
    work = tempfile.mkdtemp(prefix="aqlora_pm_")
    required = {
        "WEATHER_PATH": os.path.join(DATA, "tetovo_hourly_weather.csv"),
        "PM_PATH": os.path.join(DATA, "tetovo-air-quality-aqi.csv"),
        "DEM_PATH": os.path.join(raw_dir, "AP_01806_FBS_F0820_RT1",
                                 "AP_01806_FBS_F0820_RT1",
                                 "AP_01806_FBS_F0820_RT1_reprojected.dem.tif"),
        "SLOPE_PATH": os.path.join(raw_dir, "viz", "viz.hh_slope.tif"),
        "ROADS_PATH": os.path.join(raw_dir, "roads.geojson"),
        "INDUSTRIAL_PATH": os.path.join(raw_dir, "industrial_areas.geojson"),
        "BUILDINGS_PATH": os.path.join(raw_dir, "tetovo_buildings.geojson"),
        "FOLIAGE_PATH": os.path.join(raw_dir, "foliage.geojson"),
        "CANDIDATES_PATH": os.path.join(DATA, "candidate_locations_with_elevation.csv"),
        "OUT_CSV": os.path.join(work, "candidate_locations_with_pm.csv"),
    }
    missing = [v for v in required.values() if not os.path.exists(v)]
    if missing:
        raise SystemExit("PM-map inputs missing:\n  " + "\n  ".join(missing))

    import nbformat
    src = os.path.join(REPO, "notebooks", "synthetic_pm_map_v2.ipynb")
    dst = os.path.join(work, "synthetic_pm_map_v2.ipynb")
    nb = nbformat.read(src, as_version=4)
    replacements = {"BASE_PATH": json.dumps(work + os.sep)}
    replacements.update({k: json.dumps(v) for k, v in required.items()})
    _patch_cell(nb, replacements)
    nbformat.write(nb, dst)
    print(f"Executing PM-map notebook in {work} ...")
    _execute(dst)
    print("Notebook outputs:")
    for name in sorted(os.listdir(work)):
        if name.endswith((".csv", ".json", ".png", ".pdf")):
            print(" ", os.path.join(work, name))
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", choices=["joint", "pm-map"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--pm-raw-dir", default=None)
    args = ap.parse_args()
    if args.notebook == "joint":
        run_joint(fast=args.fast)
    else:
        run_pm_map(args.pm_raw_dir)


if __name__ == "__main__":
    main()
