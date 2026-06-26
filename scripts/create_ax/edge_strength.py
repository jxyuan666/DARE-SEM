"""Sobel edge strength — shared input for A2 and A3."""

import numpy as np
from skimage.filters import sobel

from ._helpers import normalize_image, robust_norm


def compute_edge_strength(image: np.ndarray) -> np.ndarray:
    """Extract Sobel edge magnitude from a grayscale image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Grayscale SEM/TEM image, uint8 or float32.

    Returns
    -------
    E : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    img = normalize_image(image)
    grad = sobel(img)
    return robust_norm(grad)
