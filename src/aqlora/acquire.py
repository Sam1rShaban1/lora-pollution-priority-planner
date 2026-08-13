"""Open-data acquisition: OSM layers, terrain elevation/slope, candidate generation.

Everything here works for ANY OSM-geocodable place with zero credentials:
- roads / industrial / foliage / buildings: OpenStreetMap via osmnx
- elevation & slope: AWS Terrain Tiles (terrarium encoding), public S3
"""
import io
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import requests
from PIL import Image
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_tile_cache: dict = {}


def fetch_layers(place: str, utm_crs: str) -> dict:
    """Fetch roads, industrial, foliage, buildings for a place from OSM."""
    print(f"[acquire] fetching OSM layers for '{place}' ...")
    G = ox.graph_from_place(place, network_type="drive")
    _, edges = ox.graph_to_gdfs(G)
    roads = edges[edges["highway"].astype(str).str.contains(
        "motorway|trunk|primary|secondary|tertiary", na=False)].to_crs(utm_crs)

    ind = ox.features_from_place(place, tags={"landuse": ["industrial", "commercial", "quarry"]})
    ind = ind[ind.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(utm_crs)

    fol = ox.features_from_place(place, tags={
        "landuse": ["forest", "grass", "meadow", "orchard", "vineyard"],
        "natural": ["wood", "scrub", "grassland", "heath"],
        "leisure": ["park", "garden", "nature_reserve"]})
    fol = fol[fol.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(utm_crs)

    bld = ox.features_from_place(place, tags={"building": True})
    bld = bld[bld.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(utm_crs)

    boundary = ox.geocode_to_gdf(place).to_crs(utm_crs)
    print(f"[acquire] roads={len(roads)} industrial={len(ind)} foliage={len(fol)} buildings={len(bld)}")
    return dict(roads=roads, industrial=ind, foliage=fol, buildings=bld, boundary=boundary)


def building_heights(bld: gpd.GeoDataFrame, default_h: float = 10.0) -> np.ndarray:
    """Estimate per-building heights from OSM tags (height / building:levels),
    falling back to a default. For higher fidelity, substitute a DSM-DTM fusion
    (ALOS PALSAR + FABDEM) as described in the paper."""
    h = np.full(len(bld), default_h)
    if "height" in bld.columns:
        v = pd.to_numeric(bld["height"], errors="coerce")
        h = np.where(v.notna(), v.fillna(default_h), h)
    if "building:levels" in bld.columns:
        lv = pd.to_numeric(bld["building:levels"], errors="coerce")
        h = np.where((h == default_h) & lv.notna(), lv.fillna(0) * 3.0, h)
    return np.clip(h.astype(float), 2.0, 60.0)


def _tile(z: int, x: int, y: int) -> np.ndarray:
    if (z, x, y) not in _tile_cache:
        r = requests.get(TERRARIUM.format(z=z, x=x, y=y), timeout=60)
        r.raise_for_status()
        a = np.asarray(Image.open(io.BytesIO(r.content))).astype(np.float64)
        _tile_cache[(z, x, y)] = a[:, :, 0] * 256 + a[:, :, 1] + a[:, :, 2] / 256 - 32768
    return _tile_cache[(z, x, y)]


def elevation_at(lat: float, lon: float, z: int = 13) -> float:
    n = 2 ** z
    xt = (lon + 180) / 360 * n
    yt = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    zx, zy = int(xt), int(yt)
    px, py = min(int((xt - zx) * 256), 255), min(int((yt - zy) * 256), 255)
    return float(_tile(z, zx, zy)[py, px])


def slope_at(lat: float, lon: float, step_m: float = 60.0) -> float:
    dlat = step_m / 111320.0
    dlon = step_m / (111320.0 * math.cos(math.radians(lat)))
    dzdx = (elevation_at(lat, lon + dlon) - elevation_at(lat, lon - dlon)) / (2 * step_m)
    dzdy = (elevation_at(lat + dlat, lon) - elevation_at(lat - dlat, lon)) / (2 * step_m)
    return math.degrees(math.atan(math.hypot(dzdx, dzdy)))


def generate_candidates(layers: dict, utm_crs: str, n: int = 100,
                        max_road_dist_m: float = 300.0,
                        min_spacing_m: float = 250.0, seed: int = 42) -> pd.DataFrame:
    """Auto-generate candidate sensor locations: grid points inside the city
    boundary, near roads (source proximity), with minimum spacing."""
    boundary = layers["boundary"].geometry.union_all()
    minx, miny, maxx, maxy = boundary.bounds
    step = max(min_spacing_m / 2, math.sqrt(boundary.area / (n * 20)))
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
        *np.meshgrid(xs, ys)).ravel(), crs=utm_crs) if False else None
    XX, YY = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=utm_crs)
    pts = pts[pts.within(boundary)]

    road_tree = STRtree(layers["roads"].geometry.values)
    idx = road_tree.nearest(pts.geometry.values)
    d = np.array([pts.geometry.values[i].distance(layers["roads"].geometry.values[j])
                  for i, j in enumerate(idx)])
    pts = pts[d <= max_road_dist_m]

    # greedy spacing filter
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pts))
    kept: list = []
    coords = np.array([(g.x, g.y) for g in pts.geometry.values])
    for i in order:
        if all((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2 >= min_spacing_m**2
               for j in kept):
            kept.append(i)
        if len(kept) >= n:
            break
    sel = pts.iloc[kept].to_crs("EPSG:4326")
    df = pd.DataFrame({"id": range(1, len(sel) + 1),
                       "latitude": sel.geometry.y.values,
                       "longitude": sel.geometry.x.values})
    print(f"[acquire] generated {len(df)} candidates")
    return df


def candidate_features(cand: pd.DataFrame, layers: dict, utm_crs: str) -> pd.DataFrame:
    """Compute the six spatial-correction features at candidate locations."""
    pts = gpd.GeoDataFrame(cand.copy(), geometry=gpd.points_from_xy(
        cand.longitude, cand.latitude), crs="EPSG:4326").to_crs(utm_crs)

    def min_dist(gdf):
        tree = STRtree(gdf.geometry.values)
        idx = tree.nearest(pts.geometry.values)
        return np.array([pts.geometry.values[i].distance(gdf.geometry.values[j])
                         for i, j in enumerate(idx)])

    out = cand.copy()
    out["d_road"] = min_dist(layers["roads"])
    out["d_ind"] = min_dist(layers["industrial"]) if len(layers["industrial"]) else 1e6
    out["d_fol"] = min_dist(layers["foliage"]) if len(layers["foliage"]) else 1e6
    cent = layers["buildings"].geometry.centroid
    cxy = np.array([(g.x, g.y) for g in cent])
    P = np.array([(g.x, g.y) for g in pts.geometry])
    out["C_dens"] = cKDTree(cxy).query_ball_point(P, 500, return_length=True)
    if "elevation" not in out.columns or out.get("elevation") is None:
        out["elevation"] = [elevation_at(r.latitude, r.longitude) for r in out.itertuples()]
    out["slope"] = [slope_at(r.latitude, r.longitude) for r in out.itertuples()]
    return out
