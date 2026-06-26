"""Spearman rank correlation between A(x) and model log-variance."""

import numpy as np
from typing import Dict


def _filter(U: np.ndarray, A: np.ndarray):
    """Flatten and drop non-finite pixels."""
    U, A = np.asarray(U, dtype=np.float64).reshape(-1), np.asarray(A, dtype=np.float64).reshape(-1)
    valid = np.isfinite(U) & np.isfinite(A)
    return U[valid], A[valid]


def compute_spearman_r(
    log_var: np.ndarray,
    A_map: np.ndarray,
) -> Dict[str, float]:
    """Spearman rank correlation between A(x) and model log-var.

    Measures overall spatial agreement: do pixels that A(x) considers
    ambiguous also receive high log-var from the model?

    Uses ranks rather than raw values — only monotonic ordering matters.

    Parameters
    ----------
    log_var : np.ndarray, shape (H, W)
    A_map : np.ndarray, shape (H, W)

    Returns
    -------
    dict with spearman_r, spearman_p, n_pixels.
    """
    from scipy.stats import spearmanr

    U, A = _filter(log_var, A_map)
    if U.size < 3:
        return {"spearman_r": float("nan"), "spearman_p": float("nan"),
                "n_pixels": int(U.size)}

    r, p = spearmanr(U, A)
    return {"spearman_r": float(r), "spearman_p": float(p),
            "n_pixels": int(U.size)}
