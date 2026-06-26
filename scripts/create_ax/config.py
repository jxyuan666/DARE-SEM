"""Hyperparameter configuration for Ambiguity Prior A(x)."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AmbiguityConfig:
    """Hyperparameters for the four-channel ambiguity prior.

    Attributes
    ----------
    mask_is_boundary : bool
        True = white pixels are grain boundary lines.
        False = white pixels are grain interior (boundary auto-extracted).
    target_size : int or None
        Resize to (N, N) before processing. None keeps original size.
    sigma_boundary : float
        A1 — Gaussian sigma (px) for boundary proximity band.
    sigma_junction : float
        A4 — Gaussian sigma (px) for junction-point influence.
    sigma_density : float
        A4 — Gaussian sigma (px) for boundary density smoothing.
    edge_percentile : float
        A3 — Percentile threshold (0-100) for "strong" image edges.
    weights : tuple of 4 floats
        Soft-OR weights for (A1, A2, A3, A4).
    """

    mask_is_boundary: bool = True
    target_size: int | None = None
    sigma_boundary: float = 3.0
    sigma_junction: float = 6.0
    sigma_density: float = 4.0
    edge_percentile: float = 90.0
    weights: Tuple[float, float, float, float] = (0.30, 0.25, 0.20, 0.25)
