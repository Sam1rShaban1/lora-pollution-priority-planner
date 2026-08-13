"""Gateway placement: priority-weighted coverage-consistent NSGA-II + greedy
set-cover baseline (the formulation of the paper's Sec. V-D)."""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class GatewayPlacement(Problem):
    """f1: -sum_i w_i * [max_j selected rel_ij > threshold]  (maximize coverage)
    f2: mean best SF (minimize airtime/energy)
    constraint: exactly K gateways."""

    def __init__(self, rel: np.ndarray, sf: np.ndarray, k: int,
                 weights: np.ndarray = None, threshold: float = 0.99):
        n = rel.shape[0]
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.k = k
        self.w = np.ones(n) / n if weights is None else weights
        self.feas = rel > threshold
        np.fill_diagonal(self.feas, True)
        self.sf = np.where(sf == 0, 20.0, sf)

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.astype(bool)
        out["G"] = ((X.sum(axis=1) - self.k) ** 2)[:, None]
        covered = (self.feas[None, :, :] & X[:, None, :]).any(axis=2)
        f1 = -(covered * self.w[None, :]).sum(axis=1)
        masked = np.where(X.T[None, :, :], self.sf[:, :, None], 20.0)
        f2 = masked.min(axis=1).mean(axis=0)
        out["F"] = np.column_stack([f1, f2])


def nsga2_placement(rel, sf, k, weights=None, threshold=0.99,
                    pop_size=500, generations=200, seed=42) -> np.ndarray:
    """Returns indices of selected gateways (ideal-point pick from the front)."""
    prob = GatewayPlacement(rel, sf, k, weights, threshold)
    algo = NSGA2(pop_size=pop_size, sampling=BinaryRandomSampling(),
                 crossover=TwoPointCrossover(), mutation=BitflipMutation(),
                 eliminate_duplicates=True)
    res = minimize(prob, algo, get_termination("n_gen", generations),
                   seed=seed, verbose=False)
    F, X = res.F, res.X
    if F.ndim == 1:
        F, X = F[None, :], X[None, :]
    best = np.argmin(np.linalg.norm(F - np.array([-1.0, 7.0]), axis=1))
    return np.where(X[best].astype(bool))[0]


def greedy_placement(rel, k, weights=None, threshold=0.99) -> np.ndarray:
    """Classical greedy weighted maximum-coverage baseline."""
    n = rel.shape[0]
    w = np.ones(n) / n if weights is None else weights
    feas = rel > threshold
    np.fill_diagonal(feas, True)
    covered = np.zeros(n, dtype=bool)
    sel: list = []
    for _ in range(k):
        gains = np.array([w[feas[:, c] & ~covered].sum() if c not in sel else -1
                          for c in range(n)])
        c = int(np.argmax(gains))
        sel.append(c)
        covered |= feas[:, c]
    return np.array(sel)


def evaluate(sel: np.ndarray, rel: np.ndarray, sf: np.ndarray,
             pm: np.ndarray, hotspot_quantile: float = 0.75,
             threshold: float = 0.99) -> dict:
    """Deployment report card for a gateway set."""
    n = rel.shape[0]
    sfm = np.where(sf == 0, 20.0, sf)
    selset = set(sel.tolist())
    nongw = np.array([i for i in range(n) if i not in selset])
    cov = rel[nongw][:, sel].max(axis=1) > threshold
    topq = (pm >= np.quantile(pm, hotspot_quantile))[nongw]
    return dict(
        gateways=sorted(sel.tolist()),
        coverage=float(cov.mean()),
        hotspot_coverage=float(cov[topq].mean()) if topq.any() else None,
        avg_sf_covered=float(sfm[nongw][:, sel].min(axis=1)[cov].mean()) if cov.any() else None,
        n_uncovered=int((~cov).sum()),
    )
