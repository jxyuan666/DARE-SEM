"""Orchestrator: build A(x) from image + mask."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ._helpers import to_boundary
from .config import AmbiguityConfig
from .edge_strength import compute_edge_strength
from .proximity import compute_boundary_proximity
from .weak_edge import compute_weak_boundary
from .edge_conflict import compute_edge_conflict
from .topology import compute_topology_complexity
from .combine import soft_or_channels


def build_ambiguity_prior(
    image: np.ndarray,
    mask: np.ndarray,
    config: AmbiguityConfig | None = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build A(x) from image + mask.

    Parameters
    ----------
    image : np.ndarray, shape (H, W), grayscale SEM/TEM.
    mask : np.ndarray, shape (H, W), binary GT mask.
    config : AmbiguityConfig or None
    **kwargs : individual overrides (e.g. edge_percentile=98)

    Returns
    -------
    A : np.ndarray, shape (H, W), float32 in [0, 1]
    components : dict with A_boundary, A_weak_boundary, A_edge_conflict,
                 A_topology, edge_strength, boundary.
    """
    if config is None:
        config = AmbiguityConfig()

    def _p(name):
        return kwargs.get(name, getattr(config, name))

    mask_is_boundary = _p("mask_is_boundary")
    sigma_boundary = _p("sigma_boundary")
    sigma_junction = _p("sigma_junction")
    sigma_density = _p("sigma_density")
    edge_percentile = _p("edge_percentile")
    weights = _p("weights")

    E = compute_edge_strength(image)
    A1 = compute_boundary_proximity(mask, mask_is_boundary, sigma_boundary)
    A2 = compute_weak_boundary(A1, E)
    A3 = compute_edge_conflict(E, A1, edge_percentile)
    A4 = compute_topology_complexity(mask, mask_is_boundary,
                                     sigma_density, sigma_junction)

    A = soft_or_channels([A1, A2, A3, A4], weights)
    B = to_boundary(mask > 0, mask_is_boundary)

    return A.astype(np.float32), {
        "A_boundary": A1,
        "A_weak_boundary": A2,
        "A_edge_conflict": A3,
        "A_topology": A4,
        "edge_strength": E,
        "boundary": B.astype(np.float32),
    }


def build_from_paths(
    image_path: str | Path,
    mask_path: str | Path,
    config: AmbiguityConfig | None = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read image + mask from disk, then build A(x)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if config is None:
        config = AmbiguityConfig()
    target_size = kwargs.pop("target_size", config.target_size)
    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return build_ambiguity_prior(image, mask, config=config, **kwargs)
