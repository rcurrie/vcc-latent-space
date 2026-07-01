"""S3: matched per-neighborhood delta targets (training-time).

For each perturbation we want a SET of gene-space deltas {Δ_nb}, not one
pseudo-bulk delta — that set's spread is what the neighborhood experiment is
about. We get it by organizing both perturbed and control cells into overlapping
KNN neighborhoods (the S2 primitive), then for each perturbed neighborhood:

    Δ_nb = pseudobulk(perturbed nb) − pseudobulk(matched control nb)

The control neighborhood is matched to the perturbed one by **cell-cycle phase
composition** — the cheap covariate. S1 found phase explains only ~3-4% of
within-pert variance and the discrete calling is near-degenerate (almost no G1),
so this matching is weak by construction; it's the plan's "cheap path only for
now". The spread that S4 needs lives in the neighborhoods regardless of how good
the matching is. A richer embedding-based matching is the obvious later upgrade.

Everything is in gene space (log1p CP10k). Pseudo-bulk is count-corrected
(sum raw UMIs -> CP10k -> log1p) via lewm.neighborhoods.

`build_matched_deltas` also returns the control neighborhood pseudo-bulks, which
S4 uses at inference: apply the predicted delta to control neighborhoods to emit
a population.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from lewm.data import normalize
from lewm.neighborhoods import build_knn_neighborhoods, neighborhood_pseudobulks

# Tirosh/Regev-lab cell-cycle markers (human symbols); canonical home for the
# project. (scripts/v2/diagnostic_gaussianity.py keeps its own copy from S1 —
# worth de-duplicating against this module later.)
TIROSH_S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
]
TIROSH_G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]
PHASES = ["G1", "S", "G2M"]


def score_phase(x_norm: np.ndarray, var_names: list[str]) -> np.ndarray:
    """Cell-cycle phase per cell via scanpy on log1p(CP10k) data.

    x_norm : (n_cells, n_genes) log1p(CP10k). Returns (n_cells,) str labels in
    {G1, S, G2M}. Markers are intersected with var_names (missing aliases drop).
    """
    var_set = set(var_names)
    s_genes = [g for g in TIROSH_S_GENES if g in var_set]
    g2m_genes = [g for g in TIROSH_G2M_GENES if g in var_set]
    adata = ad.AnnData(X=x_norm.astype(np.float32), var=pd.DataFrame(index=list(var_names)))
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    return adata.obs["phase"].to_numpy().astype(str)


def _phase_comp(labels: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Fraction of a neighborhood's cells in each phase (vector over PHASES)."""
    lab = labels[members]
    return np.array([(lab == p).mean() for p in PHASES], dtype=np.float64)


@dataclass
class PertTargets:
    pert: str
    deltas: np.ndarray              # (n_nb, G) matched per-neighborhood deltas
    pert_pb: np.ndarray             # (n_nb, G) perturbed neighborhood pseudo-bulks
    pert_nb_z: np.ndarray           # (n_nb, embed_dim) perturbed neighborhood latent centroids
    matched_ctrl_idx: np.ndarray    # (n_nb,) index into control neighborhoods
    phase_l1_matched: float         # mean |comp_pert - comp_matched_ctrl|
    phase_l1_random: float          # same, but control matched at random (baseline)


@torch.no_grad()
def _embed(encoder, X_csr, idx, device, bs=512) -> np.ndarray:
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(normalize(X_csr[idx[s:s + bs]].toarray())).to(device)
        out.append(encoder(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_matched_deltas(
    *,
    encoder,
    device,
    split,
    pert_ids: list[int],
    k: int = 50,
    n_ctrl: int = 5000,
    n_pert_max: int = 1500,
    min_pert_cells: int = 200,
    prop: float = 0.1,
    seed: int = 0,
    match_by: str = "phase",
):
    """Build matched per-neighborhood deltas for the given perturbations.

    Returns (targets, control_nb_pb) where targets is a list[PertTargets] and
    control_nb_pb is (n_ctrl_nb, G) — the control neighborhood pseudo-bulks S4
    applies the predicted delta to at inference.
    """
    rng = np.random.default_rng(seed)
    var_names = list(split.var_names)

    # ---- control side (built once) ----
    cidx = np.where(split.control_mask)[0]
    cidx = np.sort(rng.choice(cidx, size=min(n_ctrl, len(cidx)), replace=False))
    z_ctrl = _embed(encoder, split.X, cidx, device)
    x_ctrl_norm = normalize(split.X[cidx].toarray())
    ctrl_phase = score_phase(x_ctrl_norm, var_names)
    ctrl_nbhds = build_knn_neighborhoods(z_ctrl, k=k, prop=prop, seed=seed)
    ctrl_nb_pb = neighborhood_pseudobulks(split.X[cidx], ctrl_nbhds)          # (Cnb, G)
    ctrl_nb_comp = np.stack([_phase_comp(ctrl_phase, nb) for nb in ctrl_nbhds])  # (Cnb, 3)
    ctrl_nb_z = np.stack([z_ctrl[nb].mean(axis=0) for nb in ctrl_nbhds])      # (Cnb, D)

    targets: list[PertTargets] = []
    for pid in pert_ids:
        pos = np.where(split.pert_ids == pid)[0]
        if len(pos) < min_pert_cells:
            continue
        if len(pos) > n_pert_max:
            pos = np.sort(rng.choice(pos, size=n_pert_max, replace=False))
        z_p = _embed(encoder, split.X, pos, device)
        x_p_norm = normalize(split.X[pos].toarray())
        p_phase = score_phase(x_p_norm, var_names)
        p_nbhds = build_knn_neighborhoods(z_p, k=k, prop=prop, seed=seed)
        p_nb_pb = neighborhood_pseudobulks(split.X[pos], p_nbhds)             # (Pnb, G)
        p_nb_z = np.stack([z_p[nb].mean(axis=0) for nb in p_nbhds])           # (Pnb, D)
        p_nb_comp = np.stack([_phase_comp(p_phase, nb) for nb in p_nbhds])    # (Pnb, 3)

        # Match each perturbed nb to a control nb: by phase composition (cheap,
        # leaves baseline cell-state in the delta) or by latent position (cancels
        # baseline state, isolating the perturbation effect).
        if match_by == "embedding":
            d = ((p_nb_z[:, None, :] - ctrl_nb_z[None, :, :]) ** 2).sum(-1)   # (Pnb, Cnb)
        else:
            d = np.abs(p_nb_comp[:, None, :] - ctrl_nb_comp[None, :, :]).sum(-1)
        matched = d.argmin(axis=1)
        deltas = p_nb_pb - ctrl_nb_pb[matched]                               # (Pnb, G)

        l1_matched = float(np.abs(p_nb_comp - ctrl_nb_comp[matched]).sum(-1).mean())
        rand_match = rng.integers(0, len(ctrl_nbhds), size=len(p_nbhds))
        l1_random = float(np.abs(p_nb_comp - ctrl_nb_comp[rand_match]).sum(-1).mean())

        targets.append(PertTargets(
            pert=split.pert_vocab[pid], deltas=deltas, pert_pb=p_nb_pb,
            pert_nb_z=p_nb_z, matched_ctrl_idx=matched,
            phase_l1_matched=l1_matched, phase_l1_random=l1_random,
        ))

    return targets, ctrl_nb_pb
