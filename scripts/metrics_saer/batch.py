"""Batch sAER — three channel groupings at once."""

import numpy as np
from typing import Dict

from .enrichment import compute_saer_off


def compute_saer_all(
    log_var: np.ndarray,
    A_boundary: np.ndarray,
    A_weak: np.ndarray,
    A_conflict: np.ndarray,
    A_topology: np.ndarray,
    tau: float = 0.2,
    beta: float = 5.0,
) -> Dict[str, float]:
    """Off-boundary sAER for weak, structural, and nonB channels.

    Returns a flat dict with:
        weak_sAER_off, struct_sAER_off, nonB_sAER_off, tau, n_pixels.
    """
    results = {"tau": tau}

    for prefix, keys, wts in [
        ("weak", [A_weak], (0.25,)),
        ("struct", [A_conflict, A_topology], (0.20, 0.25)),
        ("nonB", [A_weak, A_conflict, A_topology], (0.25, 0.20, 0.25)),
    ]:
        r = compute_saer_off(log_var, A_boundary, keys, wts, tau, beta)
        for k, v in r.items():
            if k != "tau":
                results[f"{prefix}_{k}"] = v

    return results
