"""A1 — Soft proximity to GT grain boundaries."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from ._helpers import robust_norm, to_boundary


def compute_boundary_proximity(
    mask: np.ndarray,
    mask_is_boundary: bool = True,
    sigma_boundary: float = 3.0,
) -> np.ndarray:
    """Soft proximity to GT boundaries via Gaussian kernel on distance transform.

        A1(x) = exp(-d(x, B)^2 / (2 * sigma^2))

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), binary GT mask.
    mask_is_boundary : bool
        True if mask encodes boundary pixels directly.
    sigma_boundary : float
        Gaussian sigma (px). Larger = wider "near-boundary" band.

    Returns
    -------
    A_boundary : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    B = to_boundary(mask > 0, mask_is_boundary)
    dist = distance_transform_edt(~B).astype(np.float32)
    A = np.exp(-(dist ** 2) / (2.0 * sigma_boundary ** 2))
    return robust_norm(A)
