"""A3 — Strong image edges away from GT boundary."""

import numpy as np

from ._helpers import robust_norm


def compute_edge_conflict(
    edge_strength: np.ndarray,
    A_boundary: np.ndarray,
    edge_percentile: float = 90.0,
) -> np.ndarray:
    """Edge conflict: SEM shows crisp edge, but GT says no boundary.

        A3 = 1[E >= P_k] * E * (1 - A1)

    Parameters
    ----------
    edge_strength : np.ndarray, shape (H, W), float32 in [0, 1]
    A_boundary : np.ndarray, shape (H, W), float32 in [0, 1]
    edge_percentile : float
        Percentile threshold (0-100). Higher = stricter, less noise.

    Returns
    -------
    A_conflict : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    thr = np.percentile(edge_strength, edge_percentile)
    strong = (edge_strength >= thr).astype(np.float32)
    A = strong * edge_strength * (1.0 - A_boundary)
    return robust_norm(A)
