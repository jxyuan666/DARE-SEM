"""A4 — Topological complexity: boundary density + junction points."""

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize

from ._helpers import robust_norm, to_boundary


def compute_topology_complexity(
    mask: np.ndarray,
    mask_is_boundary: bool = True,
    sigma_density: float = 4.0,
    sigma_junction: float = 6.0,
) -> np.ndarray:
    """Topological complexity: local boundary density + junction proximity.

        A4 = max(Density, Junction)

    - Density: Gaussian blur of GT boundary mask.
    - Junction: skeleton branch-points (>=3 neighbors), Gaussian-diffused.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W), binary GT mask.
    mask_is_boundary : bool
    sigma_density : float, sigma_junction : float

    Returns
    -------
    A_topology : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    B = to_boundary(mask > 0, mask_is_boundary)

    # Density
    A_density = gaussian_filter(B.astype(np.float32), sigma=sigma_density)
    A_density = robust_norm(A_density)

    # Junction points from skeleton
    skel = skeletonize(B).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    nbr = convolve(skel.astype(np.float32), kernel, mode="constant", cval=0)
    junction = (skel > 0) & ((nbr - skel.astype(np.float32)) >= 3)

    if junction.sum() > 0:
        dist = distance_transform_edt(~junction).astype(np.float32)
        A_junction = np.exp(-(dist ** 2) / (2.0 * sigma_junction ** 2))
        A_junction = robust_norm(A_junction)
    else:
        A_junction = np.zeros_like(B, dtype=np.float32)

    A = np.maximum(A_density, A_junction)
    return robust_norm(A)
