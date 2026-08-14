"""Fading models for the Monte Carlo link engine.

These are the models that motivate a sampling engine: none of them admits the
closed-form reliability/SF expressions available for i.i.d. Gaussian shadowing.

- iid_gaussian          : baseline model (has a closed form; oracle exists)
- correlated_shadowing  : Gudmundson-style spatially correlated log-normal
                          shadowing. Per-node Gaussian fields with exponential
                          spatial decorrelation; link shadowing combines the
                          endpoint fields, so links sharing (or near) endpoints
                          are correlated. No closed form for the joint link
                          statistics.
- rician_multipath      : log-normal shadowing + Rician small-scale fading
                          (K-factor); the dB-domain sum of a Gaussian and a
                          log-Rician term has no closed form.
- empirical             : residuals bootstrapped from field-measured RSSI
                          samples (i.i.d. resampling of the measured error
                          distribution around its mean).

Every sampler has the signature sample(xp, n, b, rng) -> (n, n, b) fading
tensor in dB, where xp is numpy or cupy.
"""
import numpy as np


def iid_gaussian(sigma_db: float = 8.0):
    def sample(xp, n, b, seed_state):
        if xp is np:
            return seed_state.normal(0, sigma_db, size=(n, n, b)).astype(np.float32)
        return xp.random.normal(0, sigma_db, size=(n, n, b), dtype=xp.float32)
    sample.name = "iid_gaussian"
    return sample


def correlated_shadowing(coords_xy: np.ndarray, sigma_db: float = 8.0,
                         decorrelation_m: float = 150.0):
    """Per-node shadowing fields S_i with Cov = sigma^2 exp(-d/d_corr);
    link shadowing S_ij = (S_i + S_j) / sqrt(2) (preserves marginal sigma)."""
    d = np.hypot(coords_xy[:, None, 0] - coords_xy[None, :, 0],
                 coords_xy[:, None, 1] - coords_xy[None, :, 1])
    cov = sigma_db ** 2 * np.exp(-d / decorrelation_m)
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(coords_xy)))

    def sample(xp, n, b, seed_state):
        Lx = xp.asarray(L.astype(np.float32))
        if xp is np:
            z = seed_state.normal(0, 1, size=(n, b)).astype(np.float32)
        else:
            z = xp.random.normal(0, 1, size=(n, b), dtype=xp.float32)
        S = Lx @ z                                    # (n, b) node fields
        return ((S[:, None, :] + S[None, :, :]) / np.sqrt(2)).astype(xp.float32)
    sample.name = "correlated_shadowing"
    return sample


def rician_multipath(sigma_db: float = 8.0, k_factor: float = 6.0):
    """Log-normal shadowing plus Rician small-scale fading with K-factor
    (linear); the multipath term enters as 20*log10(|h|)."""
    k = float(k_factor)
    los = np.sqrt(k / (k + 1))
    scat = np.sqrt(1 / (2 * (k + 1)))

    def sample(xp, n, b, seed_state):
        if xp is np:
            g = seed_state.normal(0, sigma_db, size=(n, n, b)).astype(np.float32)
            x = seed_state.normal(0, 1, size=(n, n, b)).astype(np.float32)
            y = seed_state.normal(0, 1, size=(n, n, b)).astype(np.float32)
        else:
            g = xp.random.normal(0, sigma_db, size=(n, n, b), dtype=xp.float32)
            x = xp.random.normal(0, 1, size=(n, n, b), dtype=xp.float32)
            y = xp.random.normal(0, 1, size=(n, n, b), dtype=xp.float32)
        h = xp.sqrt((los + scat * x) ** 2 + (scat * y) ** 2)
        return g + 20 * xp.log10(xp.maximum(h, 1e-6))
    sample.name = "rician_multipath"
    return sample


def empirical(residuals_db: np.ndarray):
    """Bootstrap fading from field-measured RSSI residuals (samples minus
    their mean). Pass e.g. per-packet RSSI logs from a measurement campaign."""
    res = np.asarray(residuals_db, dtype=np.float32)
    res = res - res.mean()

    def sample(xp, n, b, seed_state):
        r = xp.asarray(res)
        if xp is np:
            idx = seed_state.integers(0, len(res), size=(n, n, b))
        else:
            idx = xp.random.randint(0, len(res), size=(n, n, b))
        return r[idx]
    sample.name = "empirical"
    return sample
