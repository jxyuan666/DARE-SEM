"""
create-ax — Build Ambiguity Prior A(x) from SEM image + GT mask.

Four independent channels + soft-OR combination.

    from scripts.create_ax import build_ambiguity_prior, AmbiguityConfig
    A, comps = build_ambiguity_prior(image, mask)
"""

from .config import AmbiguityConfig
from .edge_strength import compute_edge_strength
from .proximity import compute_boundary_proximity
from .weak_edge import compute_weak_boundary
from .edge_conflict import compute_edge_conflict
from .topology import compute_topology_complexity
from .combine import soft_or_channels
from .build import build_ambiguity_prior, build_from_paths
from .visualize import save_ambiguity_panel
