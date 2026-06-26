"""
Generate spearman_r vs struct_sAER_off scatter plot across all models.

Output: figure/output/spearman_vs_saer.png, results.csv
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE.parent))

from scripts.create_ax import build_ambiguity_prior
from scripts.metrics_saer import compute_saer_all
from scripts.metrics_spearman import compute_spearman_r

MODELS_ROOT = Path("/public/home/jxyuan/workplace/unet_single_dual_model/loss_log-var/models")
SIDS = ["13", "15", "21", "23"]

MODELS = [
    "L0_current", "L1_attenuation_only", "L2_sampling_only",
    "L3_stopgrad_sampling", "L4_stopgrad_weighting",
    "L7_langevin_logvar",
]
SHORT = ["L0", "L1", "L2", "L3", "L4", "L6"]
COLORS = ["#0a40a3", "#470aa3", "#a30a8d", "#a30a1a", "#a36d0a", "#0aa393"]

MASKS = {"21": HERE.parent / "examples/input/mask.png"}

results = {mn: {"sp": [], "sa": []} for mn in MODELS}

for sid in SIDS:
    img_path = MODELS_ROOT / "L2_sampling_only/analysis/figures/fold_0_val" / f"sample_{sid}/png/{sid}_original.png"
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if sid in MASKS:
        mask = cv2.imread(str(MASKS[sid]), cv2.IMREAD_GRAYSCALE)
    else:
        mask = cv2.imread(
            str(MODELS_ROOT / "L2_sampling_only/analysis/figures/fold_0_val"
                / f"sample_{sid}/png/{sid}_gt_mask.png"), cv2.IMREAD_GRAYSCALE)

    if image.shape[:2] != (256, 256):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (256, 256):
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    A, comps = build_ambiguity_prior(image, mask)

    for mn in MODELS:
        npy_path = MODELS_ROOT / mn / "analysis/figures/fold_0_val" / f"sample_{sid}/npy/{sid}_aleatoric_uncertainty.npy"
        U = np.load(npy_path)
        results[mn]["sp"].append(compute_spearman_r(U, A)["spearman_r"])
        results[mn]["sa"].append(compute_saer_all(
            U, comps["A_boundary"], comps["A_weak_boundary"],
            comps["A_edge_conflict"], comps["A_topology"],
            tau=0.2,
        )["struct_sAER_off"])

    print(f"[{sid}] done")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 7.5))

offsets = {
    "L0_current": (10, 10), "L1_attenuation_only": (10, -12),
    "L2_sampling_only": (10, -10), "L3_stopgrad_sampling": (-18, -12),
    "L4_stopgrad_weighting": (12, 8),
    "L7_langevin_logvar": (12, 10),
}

for mn, lbl, c in zip(MODELS, SHORT, COLORS):
    x = np.array(results[mn]["sa"]).mean()
    y = np.array(results[mn]["sp"]).mean()
    sz = 300 if "L7" in mn else 140
    ax.scatter(x, y, c=c, s=sz, edgecolors="white", linewidth=0.8, zorder=5)
    dx, dy = offsets.get(mn, (8, 6))
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=12, fontweight="bold", color=c)

ax.axhline(y=0.3, color="gray", ls="--", lw=0.7, alpha=0.3)
ax.axvline(x=1.0, color="gray", ls="--", lw=0.7, alpha=0.5)
ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("struct-sAER-off", fontsize=14)
ax.set_ylabel("spearman-r", fontsize=14)
ax.grid(True, alpha=0.15)

fig.tight_layout()
fig.savefig(OUTPUT / "spearman_vs_saer.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# ---- CSV ----
with open(OUTPUT / "results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "spearman_r_mean", "spearman_r_std", "struct_sAER_off_mean", "struct_sAER_off_std"])
    for mn, lbl in zip(MODELS, SHORT):
        sp = np.array(results[mn]["sp"])
        sa = np.array(results[mn]["sa"])
        writer.writerow([lbl, sp.mean(), sp.std(ddof=1), sa.mean(), sa.std(ddof=1)])

# ---- Table ----
print(f"\n{'Model':<6} {'spearman_r':>10} {'struct_sAER_off':>16}")
print("-" * 36)
for mn, lbl in zip(MODELS, SHORT):
    sp = np.array(results[mn]["sp"]).mean()
    sa = np.array(results[mn]["sa"]).mean()
    m = "  <-- best" if "L7" in mn else ""
    print(f"{lbl:<6} {sp:>10.4f} {sa:>16.4f}{m}")

print(f"\nSaved: {OUTPUT / 'spearman_vs_saer.png'}")
print(f"Saved: {OUTPUT / 'results.csv'}")

# ---- sAER diagnostic for L6 ----
image = cv2.imread(str(MODELS_ROOT / "L2_sampling_only/analysis/figures/fold_0_val/sample_21/png/21_original.png"), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(HERE.parent / "examples/input/mask.png"), cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

A, comps = build_ambiguity_prior(image, mask)
log_var = np.load(MODELS_ROOT / "L7_langevin_logvar/analysis/figures/fold_0_val/sample_21/npy/21_aleatoric_uncertainty.npy")

from scripts.metrics_saer.enrichment import _build_target

A1 = np.asarray(comps["A_boundary"], dtype=np.float64)
TAU = 0.2
T = _build_target(A1, [np.asarray(comps["A_edge_conflict"]), np.asarray(comps["A_topology"])], (0.20, 0.25))
off_mask = A1 < TAU

A_off = np.where(off_mask, A, np.nan)
U_off = np.where(off_mask, log_var, np.nan)
T_off = np.where(off_mask, T, np.nan)

U_mask = log_var.ravel()[off_mask.ravel()]
lo, hi = np.percentile(U_mask, [1.0, 99.0])
U_hat = np.clip((U_mask - lo) / (hi - lo + 1e-8), 0.0, 1.0)
w = np.exp(5.0 * U_hat)
w /= w.sum()
W_full = np.full(log_var.shape, np.nan)
idx = np.where(off_mask.ravel())[0]
for i, pos in enumerate(idx):
    W_full.ravel()[pos] = w[i]

fig, axes = plt.subplots(2, 3, figsize=(16, 12))

im00 = axes[0, 0].imshow(A, cmap="magma", vmin=0, vmax=1)
axes[0, 0].set_title("A(x) — Ambiguity Prior"); axes[0, 0].axis("off")
fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

axes[0, 1].imshow(off_mask, cmap="coolwarm", vmin=0, vmax=1)
axes[0, 1].set_title(f"Off-Boundary Mask (A1 < {TAU})\n{off_mask.mean()*100:.0f}% pixels kept"); axes[0, 1].axis("off")

im02 = axes[0, 2].imshow(T_off, cmap="magma", vmin=0, vmax=np.nanmax(T))
axes[0, 2].set_title("Target Channel (A3+A4) — Off-Boundary"); axes[0, 2].axis("off")
fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

im10 = axes[1, 0].imshow(log_var, cmap="magma")
axes[1, 0].set_title("log_var — Model Uncertainty"); axes[1, 0].axis("off")
fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(U_off, cmap="magma")
axes[1, 1].set_title(f"log_var — Off-Boundary"); axes[1, 1].axis("off")
fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

im12 = axes[1, 2].imshow(W_full, cmap="hot", vmin=0)
axes[1, 2].set_title("Attention Weights w"); axes[1, 2].axis("off")
fig.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

sa = compute_saer_all(log_var, comps["A_boundary"], comps["A_weak_boundary"],
                      comps["A_edge_conflict"], comps["A_topology"], tau=TAU)
fig.suptitle(f"sAER Diagnostic — L6  |  struct={sa['struct_sAER_off']:.2f}  weak={sa['weak_sAER_off']:.2f}  nonB={sa['nonB_sAER_off']:.2f}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT / "saer_diagnostic_L6.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUTPUT / 'saer_diagnostic_L6.png'}")
