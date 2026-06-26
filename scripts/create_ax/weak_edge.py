"""A2 — Near GT boundary but image evidence is weak."""

import numpy as np

from ._helpers import robust_norm


def compute_weak_boundary(
    A_boundary: np.ndarray,
    edge_strength: np.ndarray,
) -> np.ndarray:
    """Weak boundary: near GT boundary, but SEM edge signal is low.

        A2 = A1 * (1 - E)

    Parameters
    ----------
    A_boundary : np.ndarray, shape (H, W), float32 in [0, 1]
    edge_strength : np.ndarray, shape (H, W), float32 in [0, 1]

    Returns
    -------
    A_weak : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    A = A_boundary * (1.0 - edge_strength)
    return robust_norm(A)
