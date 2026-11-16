"""Joint sensor + gateway placement, ported from ``NSGA_2_LoRa_joint.ipynb``.

This module is the canonical notebook-equivalent optimizer.  It consumes the
artifacts produced by the radio-simulation and PM-map notebooks:

* ``link_reliability_matrix_gpu_realistic.csv``
* ``link_average_sf_matrix_gpu_realistic.csv``
* ``link_mean_rssi_matrix.csv`` (required for a radio-model offset != 0)
* ``candidate_locations_with_pm.csv`` (contains ``predicted_pm`` and,
  preferably, ``hotspot_prob``)

The implementation is a faithful NumPy port: the same link model, the same
three objectives, the same cardinality-preserving operators, and the same
compromise selection rule as the notebook.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# SX1276 sensitivity table used in the notebook (SF7..SF12).
SENS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0,
        11: -134.5, 12: -137.0}

DEFAULT_DELTA_LIST = [0, -5, -10, -12.4, -15, -20]


# ---------------------------------------------------------------------------
# Link model
# ---------------------------------------------------------------------------
def closed_form_rf(mean_rssi: np.ndarray, delta: float, sigma: float):
    """Reliability and mean SF under i.i.d. Gaussian shadowing.

    Matches ``closed_form_RF`` in the joint notebook.
    """
    mu = np.asarray(mean_rssi, dtype=np.float64) + delta
    p_ge = {sf: ndtr((mu - SENS[sf]) / sigma) for sf in SENS}
    R = p_ge[12]
    pk = {7: p_ge[7]}
    for sf in range(8, 13):
        pk[sf] = p_ge[sf] - p_ge[sf - 1]
    F = sum(sf * pk[sf] for sf in pk) / np.maximum(R, 1e-12)
    F = np.where(R < 1e-9, 0.0, F)
    return R.astype(np.float32), F.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading and context
# ---------------------------------------------------------------------------
def detect_pm_column(df: pd.DataFrame, pm_col: str | None = None) -> str:
    if pm_col:
        return pm_col
    cands = [c for c in df.columns
             if "pm" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if not cands:
        raise ValueError("No numeric PM column found; set pm_col in the configuration.")
    for key in ("pred", "2_5", "2.5", "25"):
        for c in cands:
            if key in c.lower():
                return c
    return cands[0]


def load_joint_data(rel_csv: str, sf_csv: str, loc_csv: str,
                    mean_rssi_csv: str | None = None,
                    pm_col: str | None = None,
                    use_hotspot_prob: bool = True,
                    hotspot_quantile: float = 0.75,
                    hotspot_gamma: float = 4.0,
                    geo_weight: float = 0.7):
    """Load the notebook artifacts and build the evaluation context.

    The candidate rows are aligned to the matrix node order exactly as in the
    notebook (``loc.set_index("id").loc[NODE_IDS]``).
    """
    rel_df = pd.read_csv(rel_csv, index_col=0)
    sf_df = pd.read_csv(sf_csv, index_col=0)
    node_ids = rel_df.index.astype(str).tolist()
    if sf_df.index.astype(str).tolist() != node_ids:
        raise ValueError("Reliability and SF matrices have different node orders.")
    n = len(node_ids)
    r_raw = rel_df.values.astype(np.float32)
    f_raw = sf_df.values.astype(np.float32)

    loc = pd.read_csv(loc_csv)
    loc["id"] = loc["id"].astype(str)
    missing = sorted(set(node_ids) - set(loc["id"]))
    if missing:
        raise ValueError(f"Candidate CSV is missing matrix node ids: {missing[:10]}")
    loc = loc.set_index("id").loc[node_ids].reset_index()

    pm_name = detect_pm_column(loc, pm_col)
    pm = loc[pm_name].values.astype(float)
    lat = loc["latitude"].values.astype(float)
    lon = loc["longitude"].values.astype(float)

    pm_q = float(np.quantile(pm, hotspot_quantile))
    is_hot = pm >= pm_q
    if use_hotspot_prob and "hotspot_prob" in loc.columns:
        raw_w = 1.0 + hotspot_gamma * loc["hotspot_prob"].values.astype(float)
    else:
        raw_w = 1.0 + hotspot_gamma * is_hot
    w_np = raw_w / raw_w.sum()

    lat0 = np.deg2rad(lat.mean())
    x = 6371000.0 * np.deg2rad(lon) * np.cos(lat0)
    y = 6371000.0 * np.deg2rad(lat)
    dgeo = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    pm_range = (pm.max() - pm.min()) + 1e-9
    d_np = (geo_weight * dgeo / dgeo.max()
            + (1.0 - geo_weight) * np.abs(pm[:, None] - pm[None, :]) / pm_range)

    has_rssi = bool(mean_rssi_csv) and os.path.exists(str(mean_rssi_csv))
    mean_rssi = None
    if has_rssi:
        mu_df = pd.read_csv(mean_rssi_csv, index_col=0)
        mu_df.index = mu_df.index.astype(str)
        mu_df.columns = mu_df.columns.astype(str)
        mean_rssi = mu_df.loc[node_ids, node_ids].values.astype(np.float64)

    return dict(
        N=n, NODE_IDS=node_ids, R_RAW=r_raw, F_RAW=f_raw,
        loc=loc, PM=pm, PM_NAME=pm_name, LAT=lat, LON=lon,
        PM_Q=pm_q, IS_HOT=is_hot, W_NP=w_np, DGEO=dgeo, D_NP=d_np,
        HAVE_RSSI=has_rssi, MEAN_RSSI=mean_rssi,
    )


def _xp_for(use_gpu: bool):
    """Return (xp, is_gpu); CuPy when requested and available, else NumPy."""
    if use_gpu:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            a = cp.ones((2, 2), dtype=cp.float32)
            float((a @ a).sum())
            return cp, True
        except Exception as exc:  # pragma: no cover - depends on hardware
            print("GPU backend unavailable, falling back to NumPy:",
                  type(exc).__name__, str(exc)[:120])
    return np, False


def _to_np(a, xp):
    if xp is not np and hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def make_context(data: dict, delta: float = 0.0,
                 reliability_threshold: float = 0.99,
                 shadowing_sigma_db: float = 8.0,
                 sf_penalty: float = 13.0,
                 use_gpu: bool = False) -> dict:
    """Build the notebook evaluation context for a radio-model offset.

    ``use_gpu=True`` mirrors the notebook's CuPy path; the default NumPy path
    keeps the same algorithm and is what runs on machines without a GPU.
    """
    if data["HAVE_RSSI"]:
        R, F = closed_form_rf(data["MEAN_RSSI"], delta, shadowing_sigma_db)
    elif abs(delta) < 1e-9:
        R, F = data["R_RAW"].copy(), data["F_RAW"].copy()
    else:
        raise FileNotFoundError(
            "delta != 0 needs link_mean_rssi_matrix.csv (set mean_rssi_csv).")

    rok = R > reliability_threshold
    np.fill_diagonal(rok, True)
    feff = np.where(rok, F, sf_penalty).astype(np.float32)
    np.fill_diagonal(feff, 7.0)

    xp, is_gpu = _xp_for(use_gpu)
    ctx = dict(data)
    ctx.update(
        delta=float(delta), R=R, ROK_NP=rok, FEFF_NP=feff,
        ROK=rok.astype(np.float32), FEFF=feff,
        W=data["W_NP"].astype(np.float32),
        D=data["D_NP"].astype(np.float32), sf_penalty=sf_penalty,
        reliability_threshold=reliability_threshold,
        xp=xp, use_gpu=is_gpu,
        ROK_X=xp.asarray(rok.astype(np.float32)),
        FEFF_X=xp.asarray(feff),
        W_X=xp.asarray(data["W_NP"].astype(np.float32)),
        D_X=xp.asarray(data["D_NP"].astype(np.float32)),
    )
    return ctx


def evaluate_batch(ctx: dict, S: np.ndarray, G: np.ndarray):
    """Evaluate a batch of sensor/gateway masks -> (objectives, coverage).

    This mirrors ``evaluate_batch`` in the notebook, including the optional
    CuPy backend.  Results are always returned as NumPy arrays because pymoo
    itself runs on the host.
    """
    xp = ctx.get("xp", np)
    S = xp.asarray(S, dtype=xp.bool_)
    G = xp.asarray(G, dtype=xp.bool_)
    ST, GT = S.T, G.T
    covered = (ctx["ROK_X"] @ GT.astype(xp.float32)) > 0.5
    sel_cov = ST & covered
    wcov = (ctx["W_X"][:, None] * sel_cov).sum(axis=0)

    fm = xp.where(GT[None, :, :], ctx["FEFF_X"][:, :, None],
                  ctx["sf_penalty"])
    best_sf = fm.min(axis=1)
    n_s = ST.sum(axis=0)
    sf_mean = (best_sf * ST).sum(axis=0) / n_s

    dm = xp.where(ST[None, :, :], ctx["D_X"][:, :, None], 9.0)
    rep = dm.min(axis=1).mean(axis=0)

    F = xp.stack([-wcov, rep, sf_mean], axis=1)
    cov_frac = sel_cov.sum(axis=0) / n_s
    return (_to_np(F, xp).astype(np.float64),
            _to_np(cov_frac, xp))


# ---------------------------------------------------------------------------
# NSGA-II operators (cardinality preserving)
# ---------------------------------------------------------------------------
def _repair_overlap(s, g, rng):
    ov = np.flatnonzero(s & g)
    if len(ov) == 0:
        return g
    free = np.flatnonzero(~s & ~g)
    rng.shuffle(free)
    for t, j in enumerate(ov):
        g[j] = False
        g[free[t]] = True
    return g


def _cross_block(a, b, k, rng):
    child = a & b
    diff = np.flatnonzero(a ^ b)
    need = k - int(child.sum())
    if need > 0:
        child[rng.choice(diff, need, replace=False)] = True
    return child


def _swap_one(blk, other, allow, rng):
    ones = np.flatnonzero(blk)
    zero_mask = ~blk if allow else (~blk & ~other)
    zeros = np.flatnonzero(zero_mask)
    if len(ones) == 0 or len(zeros) == 0:
        return
    blk[rng.choice(ones)] = False
    blk[rng.choice(zeros)] = True


class CardinalitySampling(Sampling):
    def __init__(self, n, m, k, allow, seed=0):
        super().__init__()
        self.n, self.m, self.k, self.allow = n, m, k, allow
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, *args, **kwargs):
        n, X = self.n, np.zeros((n_samples, 2 * self.n), dtype=bool)
        for r in range(n_samples):
            s_idx = self.rng.choice(n, self.m, replace=False)
            X[r, s_idx] = True
            pool = (np.arange(n) if self.allow
                    else np.setdiff1d(np.arange(n), s_idx))
            X[r, n + self.rng.choice(pool, self.k, replace=False)] = True
        return X


class CardinalityCrossover(Crossover):
    def __init__(self, n, m, k, allow, seed=0, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n, self.m, self.k, self.allow = n, m, k, allow
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, *args, **kwargs):
        n = self.n
        _, n_matings, _ = X.shape
        Y = np.zeros((2, n_matings, 2 * n), dtype=bool)
        for q in range(n_matings):
            a, b = X[0, q].astype(bool), X[1, q].astype(bool)
            for c in range(2):
                s = _cross_block(a[:n], b[:n], self.m, self.rng)
                g = _cross_block(a[n:], b[n:], self.k, self.rng)
                if not self.allow:
                    g = _repair_overlap(s, g, self.rng)
                Y[c, q, :n], Y[c, q, n:] = s, g
        return Y


class SwapMutation(Mutation):
    def __init__(self, n, m, k, allow, p_swap=0.35, max_swaps=2,
                 seed=0, **kwargs):
        super().__init__(**kwargs)
        self.n, self.m, self.k, self.allow = n, m, k, allow
        self.p_swap, self.max_swaps = p_swap, max_swaps
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, *args, **kwargs):
        n = self.n
        Y = np.array(X, dtype=bool, copy=True)
        for r in range(len(Y)):
            s, g = Y[r, :n], Y[r, n:]
            if self.rng.random() < self.p_swap:
                for _ in range(int(self.rng.integers(1, self.max_swaps + 1))):
                    _swap_one(s, g, self.allow, self.rng)
            if self.rng.random() < self.p_swap:
                for _ in range(int(self.rng.integers(1, self.max_swaps + 1))):
                    _swap_one(g, s, self.allow, self.rng)
        return Y


class JointPlacementProblem(Problem):
    def __init__(self, ctx, m, k, allow):
        super().__init__(n_var=2 * ctx["N"], n_obj=3, n_ieq_constr=3,
                         xl=0, xu=1, vtype=bool)
        self.ctx, self.m, self.k, self.allow = ctx, m, k, allow

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=bool)
        n = self.ctx["N"]
        S, G = x[:, :n], x[:, n:]
        F, _ = evaluate_batch(self.ctx, S, G)
        cv_s = (S.sum(axis=1) - self.m) ** 2
        cv_g = (G.sum(axis=1) - self.k) ** 2
        cv_o = (np.zeros(len(x)) if self.allow
                else (S & G).sum(axis=1).astype(float))
        out["F"] = F
        out["G"] = np.column_stack([cv_s, cv_g, cv_o]).astype(float)


class ConvergenceCallback(Callback):
    def __init__(self, wmax, sf_penalty=13.0):
        super().__init__()
        self.wmax = wmax
        self.sf_penalty = sf_penalty
        self.data["hv"], self.data["cov"], self.data["sf"], self.data["rep"] = (
            [], [], [], [])

    def notify(self, algorithm):
        F, CV = algorithm.pop.get("F"), algorithm.pop.get("CV")
        feas = (np.ones(len(F), dtype=bool) if CV is None
                else (np.asarray(CV).reshape(len(F), -1)[:, 0] <= 0))
        if not feas.any():
            for key in ("hv", "cov", "sf", "rep"):
                self.data[key].append(np.nan)
            return
        Ff = F[feas]
        Fn = np.column_stack([
            1.0 - (-Ff[:, 0]) / self.wmax,
            Ff[:, 1],
            (Ff[:, 2] - 7.0) / (self.sf_penalty - 7.0),
        ])
        Fn = np.clip(Fn, 0.0, 1.0)
        try:
            hv = float(HV(ref_point=np.array([1.1, 1.1, 1.1]))(Fn))
        except Exception:
            hv = np.nan
        self.data["hv"].append(hv)
        self.data["cov"].append(float((-Ff[:, 0]).max() / self.wmax))
        self.data["sf"].append(float(Ff[:, 2].min()))
        self.data["rep"].append(float(Ff[:, 1].min()))


def select_compromise(F: np.ndarray, weights=(1.0, 1.0, 1.0)) -> int:
    span = F.max(axis=0) - F.min(axis=0)
    norm = (F - F.min(axis=0)) / (span + 1e-12)
    wsel = np.asarray(weights, dtype=float)
    return int(np.argmin(np.sqrt(((norm * wsel) ** 2).sum(axis=1))))


def optimize(ctx: dict, m: int, k: int, seed: int = 42,
             pop: int = 500, gens: int = 200, verbose: bool = False,
             selection_weights=(1.0, 1.0, 1.0),
             allow_colocation: bool = False) -> dict:
    wmax = float(np.sort(ctx["W_NP"])[-m:].sum())
    problem = JointPlacementProblem(ctx, m, k, allow_colocation)
    algorithm = NSGA2(
        pop_size=pop,
        sampling=CardinalitySampling(ctx["N"], m, k, allow_colocation, seed),
        crossover=CardinalityCrossover(ctx["N"], m, k, allow_colocation,
                                       seed + 1),
        mutation=SwapMutation(ctx["N"], m, k, allow_colocation, seed=seed + 2),
        eliminate_duplicates=True,
    )
    cb = ConvergenceCallback(wmax, ctx.get("sf_penalty", 13.0))
    res = minimize(problem, algorithm, get_termination("n_gen", gens),
                   callback=cb, seed=seed, verbose=verbose)
    if res.X is None:
        raise RuntimeError("No feasible solution found. Check M/K/allow_colocation.")
    X = np.atleast_2d(res.X).astype(bool)
    F = np.atleast_2d(res.F)
    best = select_compromise(F, selection_weights)
    n = ctx["N"]
    return dict(S=X[best, :n], G=X[best, n:], X=X, F=F, best=best,
                cb=cb.data, M=m, K=k, delta=ctx["delta"], wmax=wmax)


# ---------------------------------------------------------------------------
# Reporting, baselines, outputs
# ---------------------------------------------------------------------------
def describe(ctx: dict, S: np.ndarray, G: np.ndarray, name: str = "") -> dict:
    S, G = np.asarray(S, bool), np.asarray(G, bool)
    F, _ = evaluate_batch(ctx, S[None, :], G[None, :])
    s_idx, g_idx = np.flatnonzero(S), np.flatnonzero(G)
    m = len(s_idx)
    wmax = float(np.sort(ctx["W_NP"])[-m:].sum())
    cov_s = ctx["ROK_NP"][np.ix_(s_idx, g_idx)].any(axis=1)
    hot_s = ctx["IS_HOT"][s_idx]
    nearest = ctx["DGEO"][:, s_idx].min(axis=1)
    return {
        "Method": name, "Sensors": m, "Gateways": len(g_idx),
        "Covered sensors (%)": round(100 * cov_s.mean(), 1),
        "Hotspot sensors (#)": int(hot_s.sum()),
        "Hotspot sensors covered (%)": (
            round(100 * cov_s[hot_s].mean(), 1) if hot_s.any() else np.nan),
        "Priority-wtd coverage (% of ideal)": round(100 * (-F[0, 0]) / wmax, 1),
        "Avg SF": round(F[0, 2], 2),
        "Repr. error": round(F[0, 1], 4),
        "Mean dist to nearest sensor (m)": round(nearest.mean(), 0),
        "Max dist to nearest sensor (m)": round(nearest.max(), 0),
        "PM range sampled": (
            f"{ctx['PM'][s_idx].min():.1f}-{ctx['PM'][s_idx].max():.1f}"),
    }


def ids_of(ctx: dict, mask: np.ndarray):
    return [ctx["NODE_IDS"][i] for i in np.flatnonzero(mask)]


def greedy_gateways(ctx: dict, S: np.ndarray, k: int,
                    allow_colocation: bool = False) -> np.ndarray:
    rok, fe = ctx["ROK_NP"], ctx["FEFF_NP"]
    s_idx = np.flatnonzero(S)
    w = ctx["W_NP"][s_idx]
    allowed = np.ones(ctx["N"], bool)
    if not allow_colocation:
        allowed[s_idx] = False
    covered = np.zeros(len(s_idx), bool)
    chosen = []
    for _ in range(k):
        best_j, best_gain, best_sf = None, -1.0, 1e9
        for j in np.flatnonzero(allowed):
            if j in chosen:
                continue
            gain = float((w * (rok[s_idx, j] & ~covered)).sum())
            sf = float(fe[s_idx, j].mean())
            if (gain > best_gain + 1e-12
                    or (abs(gain - best_gain) <= 1e-12 and sf < best_sf)):
                best_j, best_gain, best_sf = j, gain, sf
        chosen.append(best_j)
        covered |= rok[s_idx, best_j]
    G = np.zeros(ctx["N"], bool)
    G[chosen] = True
    return G


def sensors_top_pm(ctx: dict, m: int) -> np.ndarray:
    S = np.zeros(ctx["N"], bool)
    S[np.argsort(-ctx["PM"])[:m]] = True
    return S


def sensors_spread(ctx: dict, m: int) -> np.ndarray:
    chosen = [int(np.argmax(ctx["PM"]))]
    while len(chosen) < m:
        d = ctx["DGEO"][:, chosen].min(axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    S = np.zeros(ctx["N"], bool)
    S[chosen] = True
    return S


def random_baseline(ctx: dict, m: int, k: int, draws: int = 200,
                    seed: int = 0, allow_colocation: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        S = np.zeros(ctx["N"], bool)
        S[rng.choice(ctx["N"], m, replace=False)] = True
        pool = (np.arange(ctx["N"]) if allow_colocation
                else np.flatnonzero(~S))
        G = np.zeros(ctx["N"], bool)
        G[rng.choice(pool, k, replace=False)] = True
        rows.append(describe(ctx, S, G,
                             "Random sensors + random gateways (mean of %d)" % draws))
    df = pd.DataFrame(rows)
    out = df.drop(columns=["Method", "PM range sampled"]).mean().round(2).to_dict()
    out.update({"Method": rows[0]["Method"], "PM range sampled": "-"})
    return out


def run_baselines(ctx: dict, S_best, G_best, m: int, k: int,
                  allow_colocation: bool = False) -> pd.DataFrame:
    s_hot = sensors_top_pm(ctx, m)
    g_hot = greedy_gateways(ctx, s_hot, k, allow_colocation)
    s_spr = sensors_spread(ctx, m)
    g_spr = greedy_gateways(ctx, s_spr, k, allow_colocation)
    rows = [
        describe(ctx, S_best, G_best, "Proposed (joint NSGA-II)"),
        describe(ctx, s_hot, g_hot, "Top-PM sensors + greedy gateways"),
        describe(ctx, s_spr, g_spr, "Spatially spread sensors + greedy gateways"),
        random_baseline(ctx, m, k, allow_colocation=allow_colocation),
    ]
    return pd.DataFrame(rows).set_index("Method")


def delta_sweep(ctx: dict, m: int, k: int, delta_list=None,
                delta_ref: float = -12.4, pop: int = 200, gens: int = 100,
                seed: int = 42, allow_colocation: bool = False):
    data = ctx
    use_gpu = bool(ctx.get("use_gpu", False))
    ref = optimize(make_context(data, delta_ref,
                                ctx["reliability_threshold"],
                                ctx.get("shadowing_sigma_db", 8.0),
                                ctx.get("sf_penalty", 13.0), use_gpu=use_gpu),
                   m, k, seed=seed, pop=pop, gens=gens,
                   allow_colocation=allow_colocation)
    ref_g = set(ids_of(ref, ref["G"]))
    ref_s = set(ids_of(ref, ref["S"]))
    rows = []
    for d in (delta_list or DEFAULT_DELTA_LIST):
        c = make_context(data, d, ctx["reliability_threshold"],
                         ctx.get("shadowing_sigma_db", 8.0),
                         ctx.get("sf_penalty", 13.0), use_gpu=use_gpu)
        r = optimize(c, m, k, seed=seed, pop=pop, gens=gens,
                     allow_colocation=allow_colocation)
        g_set, s_set = set(ids_of(r, r["G"])), set(ids_of(r, r["S"]))
        row = describe(c, r["S"], r["G"], f"delta={d}")
        row["Same gateways as ref"] = (g_set == ref_g)
        row["Sensor overlap with ref (Jaccard)"] = (
            round(len(s_set & ref_s) / len(s_set | ref_s), 2))
        row["Gateway sites"] = sorted(g_set)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def budget_sweep(ctx: dict, m: int, k_list, m_list, k_fixed: int = 5,
                 pop: int = 200, gens: int = 100, seed: int = 42,
                 allow_colocation: bool = False):
    """Notebook budget sweep: fixed M across K, then fixed K across M."""
    rows_k = []
    for k in k_list:
        if (not allow_colocation) and m + k > ctx["N"]:
            continue
        r = optimize(ctx, m, k, seed=seed, pop=pop, gens=gens,
                     allow_colocation=allow_colocation)
        rows_k.append(describe(ctx, r["S"], r["G"], f"M={m}, K={k}"))
    rows_m = []
    for mm in m_list:
        if (not allow_colocation) and mm + k_fixed > ctx["N"]:
            continue
        r = optimize(ctx, mm, k_fixed, seed=seed, pop=pop, gens=gens,
                     allow_colocation=allow_colocation)
        rows_m.append(describe(ctx, r["S"], r["G"],
                               f"M={mm}, K={k_fixed}"))
    return (pd.DataFrame(rows_k).set_index("Method"),
            pd.DataFrame(rows_m).set_index("Method"))


def seed_control(data: dict, m: int, k: int, seeds=(42, 7, 123),
                 delta_ref: float = -12.4, delta_list=None,
                 pop: int = 500, gens: int = 200,
                 allow_colocation: bool = False):
    """Notebook control: search noise vs radio-offset effect."""
    use_gpu = bool(data.get("use_gpu", False))
    c_ref = make_context(data, delta_ref, data.get("reliability_threshold", 0.99),
                         data.get("shadowing_sigma_db", 8.0),
                         data.get("sf_penalty", 13.0), use_gpu=use_gpu)
    runs = [optimize(c_ref, m, k, seed=s, pop=pop, gens=gens,
                     allow_colocation=allow_colocation) for s in seeds]

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    rows = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            rows.append({
                "Seed A": seeds[a], "Seed B": seeds[b],
                "Gateway Jaccard": round(jaccard(
                    ids_of(data, runs[a]["G"]), ids_of(data, runs[b]["G"])), 2),
                "Sensor Jaccard": round(jaccard(
                    ids_of(data, runs[a]["S"]), ids_of(data, runs[b]["S"])), 2),
            })
    base = runs[0]
    fixed = [describe(make_context(
        data, d, data.get("reliability_threshold", 0.99),
        data.get("shadowing_sigma_db", 8.0), data.get("sf_penalty", 13.0),
        use_gpu=use_gpu),
        base["S"], base["G"], f"fixed plan @ delta={d}")
        for d in (delta_list or DEFAULT_DELTA_LIST)]
    return pd.DataFrame(rows), pd.DataFrame(fixed).set_index("Method"), base


def to_mercator(lon, lat):
    """Web-Mercator coordinates (m) used by the notebook's deployment map."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    x = lon * 20037508.34 / 180.0
    y = (np.log(np.tan((90.0 + lat) * np.pi / 360.0))
         / (np.pi / 180.0) * 20037508.34 / 180.0)
    return x, y


def plot_deployment(ctx: dict, S: np.ndarray, G: np.ndarray, title: str,
                    save_stem: str | None = None):
    """Port of ``plot_deployment`` (deployment map with links and basemap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    S, G = np.asarray(S, bool), np.asarray(G, bool)
    s_idx, g_idx = np.flatnonzero(S), np.flatnonzero(G)
    X, Y = to_mercator(ctx["LON"], ctx["LAT"])
    PM = ctx["PM"]
    fig, ax = plt.subplots(figsize=(14, 14))
    sc = ax.scatter(X, Y, c=PM, cmap="YlOrRd", vmin=PM.min(), vmax=PM.max(),
                    s=70, alpha=0.6, edgecolors="none", zorder=2)

    rok, fe = ctx["ROK_NP"], ctx["FEFF_NP"]
    for i in s_idx:
        ok = [j for j in g_idx if rok[i, j]]
        if ok:
            j = min(ok, key=lambda jj: fe[i, jj])
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#444444", lw=0.7,
                    alpha=0.5, zorder=3)

    ax.scatter(X[s_idx], Y[s_idx], s=260, c="#1f77b4",
               edgecolors=["red" if ctx["IS_HOT"][i] else "white"
                           for i in s_idx],
               linewidths=3, zorder=5)
    ax.scatter(X[g_idx], Y[g_idx], s=700, c="#e74c3c", marker="s",
               edgecolors="white", linewidths=3, zorder=6)
    ax.set_aspect("equal")
    try:
        import contextily as cx
        cx.add_basemap(ax, source=cx.providers.Esri.WorldGrayCanvas,
                       crs="EPSG:3857")
    except Exception as exc:
        print("Basemap skipped:", exc)

    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(f"Predicted PM2.5 ({ctx['PM_NAME']})", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=14,
               markerfacecolor="#1f77b4", markeredgecolor="red",
               markeredgewidth=2.5,
               label=f"Sensor (M = {len(s_idx)}), red ring = hotspot site"),
        Line2D([0], [0], marker="s", linestyle="", markersize=14,
               markerfacecolor="#e74c3c", markeredgecolor="white",
               markeredgewidth=2, label=f"Gateway (K = {len(g_idx)})"),
        Line2D([0], [0], color="#444444", lw=1.5,
               label="Sensor-to-gateway link"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=14, frameon=True,
              framealpha=0.95, labelspacing=1.1, handletextpad=1.2,
              borderpad=0.9)
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    if save_stem:
        fig.savefig(save_stem + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(save_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pareto(res: dict, save_stem: str | None = None):
    """Port of ``plot_pareto`` (three objective projections)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F, b = res["F"], res["best"]
    cov = 100 * (-F[:, 0]) / res["wmax"]
    pairs = [
        (F[:, 2], cov, "Average SF (lower is better)",
         "Priority-weighted coverage (% of ideal)"),
        (F[:, 1], cov, "Representativeness error (lower is better)",
         "Priority-weighted coverage (% of ideal)"),
        (F[:, 1], F[:, 2], "Representativeness error (lower is better)",
         "Average SF (lower is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (xv, yv, xl, yl) in zip(axes, pairs):
        ax.scatter(xv, yv, s=25, alpha=0.5, c="tab:blue",
                   label="Pareto front")
        ax.scatter(xv[b], yv[b], marker="*", s=300, c="red",
                   edgecolors="black", label="Selected", zorder=5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    fig.tight_layout()
    if save_stem:
        fig.savefig(save_stem + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(save_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(res: dict, save_stem: str | None = None):
    """Port of ``plot_convergence`` (HV, coverage, SF, representativeness)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cb = res["cb"]
    gens = np.arange(1, len(cb["hv"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    titles = ["Hypervolume (normalized)",
              "Best weighted coverage (frac. of ideal)",
              "Best average SF", "Best representativeness error"]
    for ax, key, ttl in zip(axes, ["hv", "cov", "sf", "rep"], titles):
        ax.plot(gens, cb[key], linewidth=1.8)
        ax.set_xlabel("Generation")
        ax.set_title(ttl)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_stem:
        fig.savefig(save_stem + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(save_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_deployment(ctx: dict, res: dict, out_dir: str, tag: str,
                    config: dict | None = None):
    """Write the notebook's deployment CSV and JSON."""
    os.makedirs(out_dir, exist_ok=True)
    S_best, G_best = res["S"], res["G"]
    loc = ctx["loc"]
    roles = np.where(S_best & G_best, "sensor+gateway",
                     np.where(S_best, "sensor",
                              np.where(G_best, "gateway", "unused")))
    out = loc[["id", "latitude", "longitude"]].copy()
    out["pm_pred"] = ctx["PM"]
    out["hotspot"] = ctx["IS_HOT"]
    out["role"] = roles
    out_sel = out[out["role"] != "unused"].sort_values(["role", "id"])
    csv_path = os.path.join(out_dir, f"joint_deployment_{tag}.csv")
    out_sel.to_csv(csv_path, index=False)

    metrics = describe(ctx, S_best, G_best, "Proposed")
    summary = dict(
        M=res["M"], K=res["K"], delta_db=res["delta"],
        reliability_threshold=ctx["reliability_threshold"],
        hotspot_quantile=ctx.get("hotspot_quantile", 0.75),
        hotspot_gamma=ctx.get("hotspot_gamma", 4.0),
        geo_weight=ctx.get("geo_weight", 0.7),
        population=res.get("pop_size"), generations=res.get("generations"),
        seed=res.get("seed"),
        sensors=ids_of(ctx, S_best), gateways=ids_of(ctx, G_best),
        metrics={k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                 for k, v in metrics.items()},
    )
    if config:
        summary.update(config)
    json_path = os.path.join(out_dir, f"joint_deployment_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return csv_path, json_path


def run_joint(rel_csv: str, sf_csv: str, loc_csv: str,
              mean_rssi_csv: str | None = None,
              num_sensors: int = 15, num_gateways: int = 5,
              delta_db: float = -12.4,
              reliability_threshold: float = 0.99,
              shadowing_sigma_db: float = 8.0,
              sf_penalty: float = 13.0,
              hotspot_quantile: float = 0.75,
              hotspot_gamma: float = 4.0,
              geo_weight: float = 0.7,
              use_hotspot_prob: bool = True,
              pm_col: str | None = None,
              allow_colocation: bool = False,
              population_size: int = 500,
              generations: int = 200,
              seed: int = 42,
              selection_weights=(1.0, 1.0, 1.0),
              run_baselines_flag: bool = True,
              out_dir: str | None = None,
              tag: str | None = None,
              verbose: bool = False,
              use_gpu: bool = False,
              make_plots: bool = True):
    """End-to-end notebook-equivalent joint optimization."""
    data = load_joint_data(
        rel_csv, sf_csv, loc_csv, mean_rssi_csv, pm_col,
        use_hotspot_prob, hotspot_quantile, hotspot_gamma, geo_weight)
    data.update(hotspot_quantile=hotspot_quantile, hotspot_gamma=hotspot_gamma,
                geo_weight=geo_weight, shadowing_sigma_db=shadowing_sigma_db,
                sf_penalty=sf_penalty)
    ctx = make_context(data, delta_db, reliability_threshold,
                       shadowing_sigma_db, sf_penalty, use_gpu=use_gpu)
    t0 = time.time()
    res = optimize(ctx, num_sensors, num_gateways, seed=seed,
                   pop=population_size, gens=generations, verbose=verbose,
                   selection_weights=selection_weights,
                   allow_colocation=allow_colocation)
    res.update(pop_size=population_size, generations=generations, seed=seed)
    elapsed = time.time() - t0

    summary_main = pd.DataFrame([
        describe(ctx, res["S"], res["G"], "Proposed (joint NSGA-II)")])
    baseline_df = (run_baselines(ctx, res["S"], res["G"],
                                 num_sensors, num_gateways, allow_colocation)
                   if run_baselines_flag else None)

    out_paths = None
    if out_dir:
        if tag is None:
            tag = (f"M{num_sensors}_K{num_gateways}_delta{delta_db}"
                   f"_geoW_{geo_weight}_hotspotP_{use_hotspot_prob}")
        out_paths = save_deployment(
            ctx, res, out_dir, tag,
            config=dict(reliability_threshold=reliability_threshold,
                        hotspot_quantile=hotspot_quantile,
                        hotspot_gamma=hotspot_gamma, geo_weight=geo_weight,
                        population=population_size, generations=generations,
                        seed=seed))
        if make_plots:
            os.makedirs(out_dir, exist_ok=True)
            plot_deployment(
                ctx, res["S"], res["G"],
                f"Joint sensor + gateway placement "
                f"(M={num_sensors}, K={num_gateways}, delta={delta_db} dB)",
                save_stem=os.path.join(out_dir, f"joint_deployment_{tag}"))
            plot_pareto(res, save_stem=os.path.join(
                out_dir, f"joint_pareto_{tag}"))
            plot_convergence(res, save_stem=os.path.join(
                out_dir, f"joint_convergence_{tag}"))
    return dict(data=data, ctx=ctx, res=res, summary=summary_main,
                baselines=baseline_df, elapsed_s=elapsed, out_paths=out_paths,
                tag=tag)
