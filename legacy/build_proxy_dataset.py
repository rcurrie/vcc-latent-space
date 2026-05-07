"""
Geo-JEPA Proxy Dataset Builder
================================
Generates a high-fidelity biological proxy from MNIST that approximates
the key statistical and topological properties of the Virtual Cell Challenge
(VCC) benchmark dataset.

CONCEPTUAL MAPPING
------------------
  MNIST concept                  VCC / scRNA-seq concept
  ------------------------------------------------------------
  Pixel intensity (0-1)          Normalized gene expression
  784-dim pixel vector           ~20k-dim gene expression vector
  Random projection -> 2000D     High-dimensional gene space
  MNAR dropout applied           Technical dropout (low-expr genes lost)
  NB noise applied               Count-based sequencing noise
  Digit class (0,1,3,4,7,8,9)   Cell type / homeostatic "room"
  Digit 9 cells                  Pluripotent progenitor (ambiguous fate)
  Blend(9, mean_4)               CRISPRi knockdown -> erythroid fate
  Blend(9, mean_1)               CRISPRi knockdown -> lymphoid fate
  Rotation of digit 4            In-fate transcriptional modulation
  30% blend toward fate          Partial fate priming (Regime 3)
  Held-out blend -> digit 7      OOD perturbation (unseen at train time)
  Train on digits 0-8,           Train on K562/A375, test on hESC
    test on digit 9 blends         (distributional shift in VCC)

OUTPUT FILES
------------
Saves a single .npz archive + a JSON metadata file:

    data/
    |-- proxy_dataset.npz          (all arrays, see schema below)
    +-- proxy_dataset_metadata.json  (human-readable labels, VCC analogs, stats)

OUTPUT SCHEMA
-------------

homeostatic_cells  [N_home, gene_dim]  float32
    The "normal cell" atlas. Each row = one cell's gene expression profile.
    These are the anchor rooms. Used for Phase A (scaffold) training.
    ~75-85% of values are exactly zero (MNAR dropout applied).
    Nonzero values follow a negative binomial-like distribution.

    VCC analog: unperturbed control cells across multiple cell types.

homeostatic_labels  [N_home]  int32  (values 0-6)
    Integer class label for each homeostatic cell (0-indexed into room_digits).
    7 classes: digits 0,1,3,4,7,8,9 -> labels 0-6.
    Used for MCR2 compression step (the Pi matrix).

    VCC analog: cell type annotation (e.g., CD34+, erythroid, myeloid).

homeostatic_digit_ids  [N_home]  int32
    Actual MNIST digit for each cell (redundant but explicit).

perturbation_cells  [N_pert, gene_dim]  float32
    All perturbed cells stacked. 5 perturbation types x 3 dose levels.

    Perturbation types:
      pert_id 0: Fate-A push (toward digit 4)     "GATA1 analog"     Regime 1
      pert_id 1: Fate-B push (toward digit 1)     "PU.1 analog"      Regime 1
      pert_id 2: In-room rotation (digit 4, 15°)  "stress response"  Regime 2
      pert_id 3: Partial priming (30% fate-A)     "epigenetic prime"  Regime 3
      pert_id 4: Combinatorial (fate-A + rotation) "dual knockout"   Regime 1+2

    VCC analog: post-perturbation Perturb-seq profiles.

perturbation_labels  [N_pert]  int32
    The homeostatic room label that each perturbed cell STARTED from.
    (Fate-push cells start from progenitor digit-9 room.)

    VCC analog: pre-perturbation cell type annotation.

perturbation_ids  [N_pert]  int32  (values 0-4)
    Which perturbation condition. Maps to perturbation_names in metadata.

perturbation_strengths  [N_pert]  float32  (values 0.3 / 0.6 / 1.0)
    Dose-response parameter.
    0.3: weak effect (partial priming / low dose)
    0.6: intermediate
    1.0: full fate commitment / maximum dose

    VCC analog: sgRNA knockdown efficiency or drug concentration.

ood_cells  [N_ood, gene_dim]  float32
    OOD test set. Progenitor cells pushed toward digit 7 -- this direction
    was NEVER used as a perturbation during training.
    Only full strength (1.0) included.

    VCC analog: held-out perturbations in the VCC test set (H1 hESC).

ood_labels  [N_ood]  int32
    Source room label for OOD cells (all progenitor / digit-9 room).

projection_matrix  [784, gene_dim]  float32
    The fixed random projection used to map pixel space -> gene space.
    Saved so you can project new MNIST samples at inference time.

gene_dim_metadata  scalar  int32
    The gene_dim value used. Needed to correctly load and reconstruct.

HOW TO LOAD
-----------

    import numpy as np, json

    data = np.load("data/proxy_dataset.npz")
    with open("data/proxy_dataset_metadata.json") as f:
        meta = json.load(f)

    # Phase A training (homeostatic scaffold)
    X_home = data["homeostatic_cells"]      # (N, 2000)
    y_home = data["homeostatic_labels"]     # (N,)  class 0-6

    # Phase B training (seen perturbations only, full dose)
    mask = (data["perturbation_ids"] <= 1) & (data["perturbation_strengths"] == 1.0)
    X_fate = data["perturbation_cells"][mask]

    # OOD evaluation
    X_ood  = data["ood_cells"]              # (N_ood, 2000)

    # Ground truth fate destinations
    fate_dest = meta["fate_destinations"]
    # fate_dest["0"] = 4  means pert_id-0 cells should embed near digit-4 room

    # Discrimination ratio ground truth:
    # correct_room = digit 4 centroid, wrong_room = digit 1 centroid
    # DR = dist(predicted, wrong_room) / dist(predicted, correct_room)
    # Target: DR > 2.0

USAGE
-----
    python build_proxy_dataset.py                          # defaults
    python build_proxy_dataset.py --gene_dim 2000 --n_per_class 800
    python build_proxy_dataset.py --no_nb_noise            # faster build
    python build_proxy_dataset.py --output_dir ./my_data
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate as scipy_rotate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOM_DIGITS = [0, 1, 3, 4, 7, 8, 9]  # homeostatic cell types (7 classes)
PROGENITOR_DIGIT = 9  # ambiguous progenitor
FATE_A_DIGIT = 4  # fate-A target room
FATE_B_DIGIT = 1  # fate-B target room
OOD_DIGIT = 7  # OOD perturbation target (held-out)
DOSE_STRENGTHS = [0.3, 0.6, 1.0]  # perturbation dose levels

PERTURBATION_NAMES = {
    0: "fate_A_toward_4",  # Regime 1: fate-determining
    1: "fate_B_toward_1",  # Regime 1: fate-determining (different fate)
    2: "inroom_rotation",  # Regime 2: in-room modulation
    3: "partial_priming",  # Regime 3: probabilistic commitment
    4: "combinatorial_A_rot",  # Bonus: combinatorial perturbation
}

# Ground truth: which homeostatic room should perturbed cells land in?
# None = probabilistic (no single destination)
FATE_DESTINATIONS = {
    0: FATE_A_DIGIT,
    1: FATE_B_DIGIT,
    2: FATE_A_DIGIT,  # in-room: stays in digit-4 room
    3: None,  # partial priming: probabilistic
    4: FATE_A_DIGIT,  # combinatorial: fate-A wins
}


# ---------------------------------------------------------------------------
# Step 0: Load MNIST
# ---------------------------------------------------------------------------


def load_mnist():
    """Returns X_raw (70000, 784) float32 in [0,1] and y_raw (70000,) int."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError("Run: pip install scikit-learn")

    print("Loading MNIST (downloads ~12MB on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    print(f"  Loaded: X={X.shape}, y={y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Step 1: Random projection  784 -> gene_dim
#
# Each "gene" is a random linear combination of pixels, with sparse weights
# (10% nonzero) mimicking sparse transcription factor -> gene regulatory links.
# ReLU + scale maps to non-negative count-like values in [0, ~100].
# ---------------------------------------------------------------------------


def make_projection_matrix(gene_dim, rng):
    W = rng.standard_normal((784, gene_dim)).astype(np.float32) * 0.08
    sparsity_mask = (rng.random((784, gene_dim)) > 0.90).astype(np.float32)
    return W * sparsity_mask


def project_to_gene_space(pixels, W, scale=60.0):
    """
    pixels : (N, 784) float32 in [0, 1]
    W      : (784, gene_dim) sparse projection matrix
    Returns: (N, gene_dim) non-negative, count-like values in [0, ~100]
    """
    projected = pixels @ W
    return np.maximum(projected, 0) * scale


# ---------------------------------------------------------------------------
# Step 2: MNAR dropout (Missing Not At Random)
#
# Real scRNA-seq: low-expression genes are lost preferentially because fewer
# mRNA molecules are captured. This is fundamentally different from uniform
# random masking.
#
# Model:  P(dropout | x_ij) = sigmoid(-alpha * expr_normalized_ij + beta)
#   alpha=2.0, beta=1.5 => ~80% overall sparsity
#   high-expression genes: dropout prob ~5%
#   zero-expression genes: dropout prob ~82%
# ---------------------------------------------------------------------------


def mnar_dropout(X, rng, alpha=2.0, beta=1.5):
    """
    Apply expression-dependent dropout.
    X   : (N, gene_dim) non-negative counts
    Returns: (N, gene_dim) with ~75-85% zeros.
    """
    gene_max = X.max(axis=0, keepdims=True) + 1e-8
    expr_norm = X / gene_max  # normalize to [0,1]
    logit = -alpha * expr_norm + beta  # high expr -> negative logit
    p_dropout = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
    keep_mask = (rng.random(X.shape) > p_dropout).astype(np.float32)
    return X * keep_mask


# ---------------------------------------------------------------------------
# Step 3: Negative Binomial noise
#
# Real scRNA-seq counts are overdispersed relative to Poisson:
#   Var(X) = mu + mu^2 / r    (NB dispersion parameter r)
# Small r (1-5) = noisy genes (cytokines, TFs)
# Large r (10+) = stable genes (housekeeping)
# Per-gene r drawn from Gamma to simulate biological heterogeneity.
# ---------------------------------------------------------------------------


def add_negative_binomial_noise(X, rng, r_mean=5.0):
    """
    Add NB count noise to expression matrix.
    X      : (N, gene_dim) non-negative floats
    r_mean : mean gene-specific dispersion
    Returns: (N, gene_dim) float32 with integer-like count values.
    """
    n_cells, gene_dim = X.shape
    r_per_gene = rng.gamma(shape=2.0, scale=r_mean / 2.0, size=gene_dim)
    r_per_gene = np.maximum(r_per_gene, 0.5)

    noisy = np.zeros_like(X)
    for j in range(gene_dim):
        mu = X[:, j]
        r_j = r_per_gene[j]
        p = r_j / (r_j + mu + 1e-8)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        noisy[:, j] = rng.negative_binomial(max(int(r_j), 1), p).astype(np.float32)

    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared helper: project + dropout + optional NB noise
# ---------------------------------------------------------------------------


def make_cells(pixels, W, rng, add_nb_noise):
    cells = project_to_gene_space(pixels, W)
    cells = mnar_dropout(cells, rng)
    if add_nb_noise:
        cells = add_negative_binomial_noise(cells, rng)
        cells = mnar_dropout(cells, rng, alpha=1.5, beta=2.0)  # re-dropout post-noise
    return cells.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Homeostatic cells (the Anchor Rooms)
#
# 7 MNIST digit classes -> 7 cell types. Each digit is a distinct attractor
# in the gene expression manifold. These form Phase A training data.
# ---------------------------------------------------------------------------


def build_homeostatic_cells(X_raw, y_raw, W, rng, n_per_class, add_nb_noise):
    """
    Returns:
      cells        (N, gene_dim)
      labels       (N,)  0-indexed class
      digit_ids    (N,)  actual MNIST digit
      digit_means  dict {digit: mean_pixel_vector}
    """
    all_cells, all_labels, all_digit_ids = [], [], []
    digit_means = {}

    print(f"\nBuilding homeostatic rooms ({len(ROOM_DIGITS)} cell types):")
    for label_idx, digit in enumerate(ROOM_DIGITS):
        idx = np.where(y_raw == digit)[0][:n_per_class]
        pixels = X_raw[idx]
        digit_means[digit] = pixels.mean(axis=0)

        cells = make_cells(pixels, W, rng, add_nb_noise)

        all_cells.append(cells)
        all_labels.extend([label_idx] * len(idx))
        all_digit_ids.extend([digit] * len(idx))
        print(
            f"  digit={digit} (label={label_idx}): {len(idx)} cells, "
            f"sparsity={100*(cells==0).mean():.1f}%"
        )

    return (
        np.vstack(all_cells).astype(np.float32),
        np.array(all_labels, dtype=np.int32),
        np.array(all_digit_ids, dtype=np.int32),
        digit_means,
    )


# ---------------------------------------------------------------------------
# Step 5: Perturbation cells
#
# Five perturbation types covering the three biological regimes.
# Each is built at three dose strengths (0.3, 0.6, 1.0).
#
# Regime 1 (fate-determining):
#   pert_id 0  Progenitor (digit 9) -> Fate A (digit 4)
#              Biology: GATA1 CRISPRi in HSC -> erythroid commitment
#   pert_id 1  Progenitor (digit 9) -> Fate B (digit 1)
#              Biology: PU.1 CRISPRi in HSC -> myeloid commitment
#
# Regime 2 (in-room modulation):
#   pert_id 2  Digit 4 cells, rotated 15 deg in pixel space
#              Biology: ribosomal gene knockdown -- same cell type,
#                       altered stress-response transcriptional program
#
# Regime 3 (partial priming):
#   pert_id 3  Progenitor 30% blended toward fate A (single fixed dose)
#              Biology: epigenetic priming without fate commitment
#
# Combinatorial:
#   pert_id 4  Fate-A push + rotation (double perturbation)
#              Biology: combinatorial CRISPRi screen
# ---------------------------------------------------------------------------


def blend_toward(source, target_mean, strength, rng, noise_scale=0.01):
    """Linear interpolation from source toward target_mean with small noise."""
    blended = (1.0 - strength) * source + strength * target_mean[None, :]
    noise = rng.standard_normal(blended.shape).astype(np.float32) * noise_scale
    return np.clip(blended + noise, 0.0, 1.0)


def rotate_pixels(pixels, angle_deg):
    """Rotate each 28x28 MNIST image by angle_deg degrees."""
    N = len(pixels)
    out = np.zeros_like(pixels)
    for i in range(N):
        img = pixels[i].reshape(28, 28)
        out[i] = scipy_rotate(img, angle_deg, reshape=False).ravel()
    return np.clip(out, 0.0, 1.0)


def build_perturbation_cells(
    X_raw, y_raw, W, rng, digit_means, n_per_class, add_nb_noise
):
    """
    Returns:
      cells      (N_pert, gene_dim)
      src_labels (N_pert,)  source room label
      pert_ids   (N_pert,)  perturbation type 0-4
      strengths  (N_pert,)  dose 0.3/0.6/1.0
    """
    all_cells, all_src, all_ids, all_str = [], [], [], []

    prog_label = ROOM_DIGITS.index(PROGENITOR_DIGIT)
    d4_label = ROOM_DIGITS.index(FATE_A_DIGIT)

    prog_pixels = X_raw[np.where(y_raw == PROGENITOR_DIGIT)[0][:n_per_class]]
    d4_pixels = X_raw[np.where(y_raw == FATE_A_DIGIT)[0][:n_per_class]]

    def add(cells, src_label, pert_id, strength):
        all_cells.append(cells)
        all_src.extend([src_label] * len(cells))
        all_ids.extend([pert_id] * len(cells))
        all_str.extend([strength] * len(cells))

    # -- pert_id 0: Fate A at each dose -----------------------------------
    print(f"\n  pert_id=0 (Fate-A toward digit {FATE_A_DIGIT}):")
    for s in DOSE_STRENGTHS:
        blended = blend_toward(prog_pixels, digit_means[FATE_A_DIGIT], s, rng)
        cells = make_cells(blended, W, rng, add_nb_noise)
        add(cells, prog_label, 0, s)
        print(f"    s={s}: {len(cells)} cells, sparsity={100*(cells==0).mean():.1f}%")

    # -- pert_id 1: Fate B at each dose -----------------------------------
    print(f"\n  pert_id=1 (Fate-B toward digit {FATE_B_DIGIT}):")
    for s in DOSE_STRENGTHS:
        blended = blend_toward(prog_pixels, digit_means[FATE_B_DIGIT], s, rng)
        cells = make_cells(blended, W, rng, add_nb_noise)
        add(cells, prog_label, 1, s)
        print(f"    s={s}: {len(cells)} cells, sparsity={100*(cells==0).mean():.1f}%")

    # -- pert_id 2: In-room rotation of digit 4 ---------------------------
    print(f"\n  pert_id=2 (in-room rotation of digit {FATE_A_DIGIT}):")
    for s in DOSE_STRENGTHS:
        angle = s * 15.0
        rotated = rotate_pixels(d4_pixels, angle)
        cells = make_cells(rotated, W, rng, add_nb_noise)
        add(cells, d4_label, 2, s)
        print(
            f"    s={s} (angle={angle:.0f}): {len(cells)} cells, "
            f"sparsity={100*(cells==0).mean():.1f}%"
        )

    # -- pert_id 3: Partial priming (fixed s=0.3) -------------------------
    print(f"\n  pert_id=3 (partial priming, s=0.3):")
    blended = blend_toward(
        prog_pixels, digit_means[FATE_A_DIGIT], 0.3, rng, noise_scale=0.005
    )
    cells = make_cells(blended, W, rng, add_nb_noise)
    add(cells, prog_label, 3, 0.3)
    print(f"    {len(cells)} cells, sparsity={100*(cells==0).mean():.1f}%")

    # -- pert_id 4: Combinatorial (fate-A + rotation) ---------------------
    print(f"\n  pert_id=4 (combinatorial: Fate-A + rotation):")
    for s in DOSE_STRENGTHS:
        blended = blend_toward(prog_pixels, digit_means[FATE_A_DIGIT], s, rng)
        blended = rotate_pixels(blended, s * 10.0)
        cells = make_cells(blended, W, rng, add_nb_noise)
        add(cells, prog_label, 4, s)
        print(f"    s={s}: {len(cells)} cells, sparsity={100*(cells==0).mean():.1f}%")

    return (
        np.vstack(all_cells).astype(np.float32),
        np.array(all_src, dtype=np.int32),
        np.array(all_ids, dtype=np.int32),
        np.array(all_str, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Step 6: OOD test set
#
# Progenitor cells pushed toward digit 7 -- this direction was NEVER used
# as a perturbation during training. Tests whether the model can generalize
# to unseen perturbation paths using structured perturbation embeddings.
#
# VCC analog: the H1 hESC context in the VCC test set -- a genuinely new
# cell type not seen during training.
# ---------------------------------------------------------------------------


def build_ood_cells(X_raw, y_raw, W, rng, digit_means, n_per_class, add_nb_noise):
    prog_label = ROOM_DIGITS.index(PROGENITOR_DIGIT)
    prog_pixels = X_raw[np.where(y_raw == PROGENITOR_DIGIT)[0][:n_per_class]]

    print(f"\n  OOD set (toward digit {OOD_DIGIT}, s=1.0 -- never seen in training):")
    blended = blend_toward(prog_pixels, digit_means[OOD_DIGIT], 1.0, rng)
    cells = make_cells(blended, W, rng, add_nb_noise)
    labels = np.full(len(cells), prog_label, dtype=np.int32)
    print(f"    {len(cells)} cells, sparsity={100*(cells==0).mean():.1f}%")
    return cells, labels


# ---------------------------------------------------------------------------
# Step 7: Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(
    home_cells, home_labels, pert_cells, pert_ids, pert_str, ood_cells
):
    print("\nSanity checks:")

    # 1. Room separability (cosine distance between centroids)
    centroids = {}
    for label in np.unique(home_labels):
        c = home_cells[home_labels == label].mean(axis=0)
        centroids[label] = c / (np.linalg.norm(c) + 1e-8)

    labels = sorted(centroids.keys())
    dists = [
        1.0 - centroids[labels[i]] @ centroids[labels[j]]
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    sep = np.mean(dists)
    status = "PASS" if sep > 0.05 else "FAIL"
    print(
        f"  [{status}] Mean inter-room cosine distance: {sep:.4f}  " f"(target > 0.05)"
    )

    # 2. Fate-A discrimination (raw gene space, before any model)
    prog_c = home_cells[home_labels == ROOM_DIGITS.index(PROGENITOR_DIGIT)].mean(0)
    fate_a = home_cells[home_labels == ROOM_DIGITS.index(FATE_A_DIGIT)].mean(0)
    fate_b = home_cells[home_labels == ROOM_DIGITS.index(FATE_B_DIGIT)].mean(0)
    p0_full = pert_cells[(pert_ids == 0) & (pert_str == 1.0)]
    if len(p0_full) > 0:
        p0_c = p0_full.mean(0)
        da = np.linalg.norm(p0_c - fate_a)
        db = np.linalg.norm(p0_c - fate_b)
        dr = db / (da + 1e-8)
        status = "PASS" if dr > 1.2 else "INFO"
        print(
            f"  [{status}] Raw-space Fate-A DR: {dr:.2f}x  "
            f"(>1.2 means signal is present; model refines this)"
        )

    # 3. OOD distinctness
    all_pert_c = pert_cells.mean(0)
    ood_c = ood_cells.mean(0)
    sim = (all_pert_c @ ood_c) / (
        np.linalg.norm(all_pert_c) * np.linalg.norm(ood_c) + 1e-8
    )
    status = "PASS" if sim < 0.97 else "FAIL"
    print(
        f"  [{status}] OOD vs training cosine similarity: {sim:.4f}  "
        f"(target < 0.97)"
    )

    # 4. Sparsity
    sp = 100 * (home_cells == 0).mean()
    status = "PASS" if 60 < sp < 90 else "WARN"
    print(f"  [{status}] Homeostatic sparsity: {sp:.1f}%  (target 60-90%)")

    print("")


# ---------------------------------------------------------------------------
# Step 8: Save
# ---------------------------------------------------------------------------


def save_dataset(
    output_dir,
    home_cells,
    home_labels,
    home_digit_ids,
    pert_cells,
    pert_labels,
    pert_ids,
    pert_str,
    ood_cells,
    ood_labels,
    W,
    config,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "proxy_dataset.npz"
    meta_path = output_dir / "proxy_dataset_metadata.json"

    np.savez(
        npz_path,
        homeostatic_cells=home_cells,
        homeostatic_labels=home_labels,
        homeostatic_digit_ids=home_digit_ids,
        perturbation_cells=pert_cells,
        perturbation_labels=pert_labels,
        perturbation_ids=pert_ids,
        perturbation_strengths=pert_str,
        ood_cells=ood_cells,
        ood_labels=ood_labels,
        projection_matrix=W,
        gene_dim_metadata=np.array(config["gene_dim"], dtype=np.int32),
    )

    sparsity = float(100 * (home_cells == 0).mean())
    metadata = {
        "room_digits": ROOM_DIGITS,
        "label_to_digit": {str(i): d for i, d in enumerate(ROOM_DIGITS)},
        "digit_to_label": {str(d): i for i, d in enumerate(ROOM_DIGITS)},
        "perturbation_names": {str(k): v for k, v in PERTURBATION_NAMES.items()},
        "fate_destinations": {str(k): v for k, v in FATE_DESTINATIONS.items()},
        "progenitor_digit": PROGENITOR_DIGIT,
        "ood_target_digit": OOD_DIGIT,
        "dose_strengths": DOSE_STRENGTHS,
        "regimes": {
            "0": "Regime 1 -- fate-determining (discrete room switch)",
            "1": "Regime 1 -- fate-determining (different destination)",
            "2": "Regime 2 -- in-room modulation (continuous displacement)",
            "3": "Regime 3 -- partial priming (probabilistic commitment)",
            "4": "Regime 1+2 -- combinatorial",
        },
        "stats": {
            "n_homeostatic": int(len(home_cells)),
            "n_perturbed": int(len(pert_cells)),
            "n_ood": int(len(ood_cells)),
            "gene_dim": int(config["gene_dim"]),
            "n_classes": len(ROOM_DIGITS),
            "sparsity_percent": round(sparsity, 1),
            "n_perturbation_types": len(PERTURBATION_NAMES),
            "n_dose_levels": len(DOSE_STRENGTHS),
        },
        "vcc_analogs": {
            "homeostatic_cells": "unperturbed controls (Phase A scaffold)",
            "perturbation_cells": "Perturb-seq post-perturbation profiles",
            "ood_cells": "VCC held-out test perturbations (H1 hESC)",
            "homeostatic_labels": "cell type annotation",
            "perturbation_ids": "sgRNA target gene identity",
            "perturbation_strengths": "knockdown efficiency / drug dose",
            "fate_destinations": "ground-truth post-perturbation cell type",
        },
        "discrimination_ratio_guide": {
            "formula": "DR = dist(predicted, wrong_room) / dist(predicted, correct_room)",
            "target": "> 2.0  =>  Linearization Trap avoided",
            "pert_id_0": f"correct={FATE_A_DIGIT}, wrong={FATE_B_DIGIT}",
            "pert_id_1": f"correct={FATE_B_DIGIT}, wrong={FATE_A_DIGIT}",
            "pert_id_2": f"correct={FATE_A_DIGIT} (in-room, so DR may be lower)",
            "pert_id_3": "no single correct room (probabilistic)",
            "ood": "no ground truth; use nearest training-room distance as proxy",
        },
        "generation_config": config,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = npz_path.stat().st_size / 1e6
    print("=" * 60)
    print(f"Saved: {npz_path}  ({size_mb:.1f} MB)")
    print(f"Saved: {meta_path}")
    print("=" * 60)
    print(f"Homeostatic : {len(home_cells):>7,}  ({len(ROOM_DIGITS)} rooms)")
    print(
        f"Perturbed   : {len(pert_cells):>7,}  ({len(PERTURBATION_NAMES)} types "
        f"x {len(DOSE_STRENGTHS)} doses)"
    )
    print(f"OOD test    : {len(ood_cells):>7,}")
    print(f"Gene dim    : {config['gene_dim']:>7,}")
    print(f"Sparsity    : {sparsity:>6.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build MNIST proxy dataset for Geo-JEPA VCC development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gene_dim",
        type=int,
        default=2000,
        help="Projected dimensionality (default: 2000)",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=800,
        help="Cells per digit class (default: 800)",
    )
    parser.add_argument(
        "--no_nb_noise",
        action="store_true",
        help="Skip negative binomial noise for faster build",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    config = {
        "gene_dim": args.gene_dim,
        "n_per_class": args.n_per_class,
        "add_nb_noise": not args.no_nb_noise,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    print("=" * 60)
    print("Geo-JEPA MNIST Proxy Dataset Builder")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")

    rng = np.random.default_rng(config["seed"])

    X_raw, y_raw = load_mnist()

    print(f"\nBuilding projection matrix (784 -> {config['gene_dim']}D)...")
    W = make_projection_matrix(config["gene_dim"], rng)
    print(f"  Shape: {W.shape},  nonzero: {100*(W!=0).mean():.1f}%")

    home_cells, home_labels, home_digit_ids, digit_means = build_homeostatic_cells(
        X_raw, y_raw, W, rng, config["n_per_class"], config["add_nb_noise"]
    )

    pert_cells, pert_labels, pert_ids, pert_str = build_perturbation_cells(
        X_raw, y_raw, W, rng, digit_means, config["n_per_class"], config["add_nb_noise"]
    )

    ood_cells, ood_labels = build_ood_cells(
        X_raw, y_raw, W, rng, digit_means, config["n_per_class"], config["add_nb_noise"]
    )

    run_sanity_checks(
        home_cells, home_labels, pert_cells, pert_ids, pert_str, ood_cells
    )

    save_dataset(
        config["output_dir"],
        home_cells,
        home_labels,
        home_digit_ids,
        pert_cells,
        pert_labels,
        pert_ids,
        pert_str,
        ood_cells,
        ood_labels,
        W,
        config,
    )


if __name__ == "__main__":
    main()
