"""Shared helpers for the create-ax package."""

import numpy as np


def robust_norm(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Percentile-clip then min-max normalize to [0, 1]."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)


def to_boundary(mask: np.ndarray, mask_is_boundary: bool) -> np.ndarray:
    """Return a boolean boundary mask from either boundary or grain mask."""
    from skimage.morphology import binary_dilation, binary_erosion, disk

    if mask_is_boundary:
        return mask.astype(bool)
    y = mask.astype(bool)
    return (binary_dilation(y, disk(1)) ^ binary_erosion(y, disk(1))).astype(bool)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] float32."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = robust_norm(img)
    return img
