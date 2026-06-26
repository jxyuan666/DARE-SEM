"""Soft-OR combination of ambiguity channels."""

from typing import List, Tuple

import numpy as np

from ._helpers import robust_norm


def soft_or_channels(
    channels: List[np.ndarray],
    weights: Tuple[float, ...] = (0.30, 0.25, 0.20, 0.25),
) -> np.ndarray:
    """Weighted soft-OR of N ambiguity channels.

        A = 1 - prod_i(1 - w_i * A_i)

    Parameters
    ----------
    channels : list of np.ndarray, N arrays, each (H, W), values in [0, 1]
    weights : tuple of float, one weight per channel

    Returns
    -------
    A : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    if len(channels) != len(weights):
        raise ValueError(f"Got {len(channels)} channels but {len(weights)} weights.")
    H, W = channels[0].shape
    A = np.ones((H, W), dtype=np.float32)
    for w, c in zip(weights, channels):
        A *= 1.0 - w * np.clip(c, 0.0, 1.0)
    A = 1.0 - A
    return robust_norm(A)
