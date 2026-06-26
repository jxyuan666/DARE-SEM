"""Core sAER computation — off-boundary soft enrichment."""

import numpy as np
from typing import Dict, List, Tuple


def _build_target(
    A_boundary: np.ndarray,
    channels: List[np.ndarray],
    weights: Tuple[float, ...],
) -> np.ndarray:
    """Soft-OR of channels, then suppress by boundary proximity.

    T = [1 - prod(1 - w_i * C_i)] * (1 - A1^2)
    """
    A1 = np.asarray(A_boundary, dtype=np.float64)
    T = np.ones_like(A1)
    for w, C in zip(weights, channels):
        T *= 1.0 - w * np.clip(np.asarray(C, dtype=np.float64), 0.0, 1.0)
    T = 1.0 - T
    T *= (1.0 - A1 ** 2.0)
    return T


def compute_saer_off(
    log_var: np.ndarray,
    A_boundary: np.ndarray,
    channels: List[np.ndarray],
    weights: Tuple[float, ...],
    tau: float = 0.2,
    beta: float = 5.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Off-boundary Soft Ambiguity Enrichment Ratio.

    Steps
    -----
    1. Build target T = soft-OR(channels) * (1 - A1^2).
    2. Mask: keep pixels where A1 < tau.
    3. Normalise log-var within mask -> softmax -> attention weights w.
    4. sAER = sum(w * T) / mean(T).

    Parameters
    ----------
    log_var : np.ndarray, shape (H, W)
    A_boundary : np.ndarray, shape (H, W), A1 channel.
    channels : list of np.ndarray, target channels to soft-OR.
    weights : tuple of float, soft-OR weights.
    tau : float, A1 threshold for off-boundary mask.
    beta : float, softmax temperature.

    Returns
    -------
    dict with sAER_off, sAAS_off, mean_channel, mean_log_var, n_pixels, tau.
    """
    T = _build_target(A_boundary, channels, weights)
    A1 = np.asarray(A_boundary, dtype=np.float64)

    mask = A1 < tau
    U = np.asarray(log_var, dtype=np.float64).ravel()
    T = T.ravel()
    m = mask.ravel()

    U_m = U[m]; T_m = T[m]
    valid = np.isfinite(U_m) & np.isfinite(T_m)
    U_m = U_m[valid]; T_m = T_m[valid]
    n = int(U_m.size)

    if n < 3:
        return {"sAER_off": float("nan"), "sAAS_off": float("nan"),
                "mean_channel": float("nan"), "mean_log_var": float("nan"),
                "n_pixels": n, "tau": tau}

    lo, hi = np.percentile(U_m, [1.0, 99.0])
    U_hat = np.clip((U_m - lo) / (hi - lo + eps), 0.0, 1.0)
    w = np.exp(beta * U_hat)
    w /= w.sum() + eps

    sAAS = float(np.sum(w * T_m))
    mean_T = float(np.mean(T_m))
    sAER = sAAS / (mean_T + eps)

    return {
        "sAER_off": sAER, "sAAS_off": sAAS,
        "mean_channel": mean_T, "mean_log_var": float(np.mean(U_m)),
        "n_pixels": n, "tau": tau,
    }
