"""Visualization for ambiguity prior A(x)."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def save_ambiguity_panel(
    output_path: str | Path,
    image: np.ndarray,
    A: np.ndarray,
    components: Dict[str, np.ndarray],
    title: str = "Ambiguity Prior A(x)",
) -> None:
    """7-panel figure: original | boundary | A1 | A2 | A3 | A4 | final."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = [
        ("Original", image, "gray"),
        ("GT Boundary", components.get("boundary"), "binary"),
        ("A1_boundary", components.get("A_boundary"), "magma"),
        ("A2_weak-boundary", components.get("A_weak_boundary"), "magma"),
        ("A3_edge-conflict", components.get("A_edge_conflict"), "magma"),
        ("A4_topology", components.get("A_topology"), "magma"),
        ("Final A(x)", A, "magma"),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    for ax, (label, arr, cmap) in zip(axes, items):
        if arr is None or arr.size == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            ax.axis("off")
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if cmap == "binary":
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        elif cmap == "gray":
            lo, hi = np.percentile(arr, [1, 99]) if arr.max() > 1.0 else (0, 1)
            ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
