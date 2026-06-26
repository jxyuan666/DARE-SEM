"""
Full pipeline demo: SEM + Mask -> A(x) -> evaluation.

Input         examples/input/
    image.jpg     SEM grayscale image
    mask.png      GT grain boundary mask
    log_var.npy   model log-variance map

Output        examples/output/
    saer_results.json     spearman_r + sAER values
    ambiguity_panel.png   7-panel A(x) diagnostic
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE.parent))

from scripts.create_ax import build_ambiguity_prior, save_ambiguity_panel
from scripts.metrics_saer import compute_saer_all
from scripts.metrics_spearman import compute_spearman_r

# ---- Load ----
image = cv2.imread(str(INPUT / "image.jpg"), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(INPUT / "mask.png"), cv2.IMREAD_GRAYSCALE)
log_var = np.load(INPUT / "log_var.npy")

print(f"image: {image.shape}  mask: {mask.shape}  log_var mean: {log_var.mean():.4f}")

# ---- Step 1: Build A(x) ----
A, comps = build_ambiguity_prior(image, mask)
print(f"A(x): mean={A.mean():.4f}")

save_ambiguity_panel(OUTPUT / "ambiguity_panel.png", image, A, comps)
print(f"Panel: {OUTPUT / 'ambiguity_panel.png'}")

# ---- Step 2: Evaluate ----
spearman = compute_spearman_r(log_var, A)
saer = compute_saer_all(
    log_var,
    comps["A_boundary"],
    comps["A_weak_boundary"],
    comps["A_edge_conflict"],
    comps["A_topology"],
    tau=0.2, beta=5.0,
)

result = {
    "spearman_r": spearman["spearman_r"],
    "struct_sAER_off": saer["struct_sAER_off"],
    "weak_sAER_off": saer["weak_sAER_off"],
    "nonB_sAER_off": saer["nonB_sAER_off"],
}

print(f"\nspearman_r:        {result['spearman_r']:.4f}")
print(f"struct_sAER_off:   {result['struct_sAER_off']:.4f}")
print(f"weak_sAER_off:     {result['weak_sAER_off']:.4f}")
print(f"nonB_sAER_off:     {result['nonB_sAER_off']:.4f}")

with open(OUTPUT / "saer_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nDone. Output in: {OUTPUT.resolve()}")
