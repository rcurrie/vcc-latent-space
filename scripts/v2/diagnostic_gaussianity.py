"""Gaussianity diagnostic — Option 3 from the LeJEPA-identifiability discussion.

Tests whether our A1 encoder's output distribution is actually Gaussian (which
would support the latent-space approach with our current prior) vs. whether it
shows non-Gaussian structure (which would confirm the paper's diagnosis: we've
been enforcing the wrong prior on a non-Gaussian latent manifold).

Three concrete tests:

  1. **Marginal Gaussianity of encoder output (controls).** Per-dimension
     normality test via D'Agostino's K² (sample size limit-friendly) on each
     of the 256 dimensions. Reports the *fraction of dims that fail
     normality* at p<0.01. If <5% fail → encoder output is well-Gaussianized.
     If >>5% fail → biology is leaking non-Gaussian structure through despite
     MCR².

  2. **Per-perturbation Gaussianity.** For a few perturbations with enough
     cells, repeat the test. If individual classes are also Gaussian-ish,
     then the encoder is producing a mixture-of-Gaussians, with each class
     near-Gaussian. If perturbations are systematically NOT Gaussian within,
     the encoder isn't even achieving its local goal.

  3. **Encoder vs. random-projection baseline.** Project the raw log1p(CP10k)
     gene data through a random Gaussian linear projection to the same 256
     dim. Test Gaussianity of that. Random projection of high-dim data is
     "naturally Gaussian" by CLT — so this is the null baseline. If our
     learned encoder is MORE Gaussian than random projection, we are
     actively enforcing more Gaussianity than the data naturally has — which
     is the paper's failure mode.

Also reports inter-class vs intra-class variance ratio (a sanity check on
whether perturbations are even separable in latent space).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
from scipy import stats

from lewm.v2.data import load_split
from lewm.v2.models import (
    MLPEncoder,
    ProteinActionEmbedV2,
    ResidualPerturbationPredictor as PerturbationPredictor,
)
from lewm.v2.train_phase_a import select_device
from lewm.data import normalize


# Tirosh/Regev-lab cell-cycle markers (human symbols). 43 S-phase + 54 G2M.
# Standard list shipped with the scanpy cell-cycle tutorial. Some symbols are
# aliases that may be absent from the VCC panel — we intersect with var_names
# and log how many matched, so a missing alias just drops out of the score.
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


def phase_variance_explained(M: np.ndarray, labels: np.ndarray) -> dict:
    """One-way variance decomposition of M by group `labels` (eta²).

    M : (n, d) per-cell features (latent z or gene log1p-CP10k).
    labels : (n,) categorical phase labels.

    Returns a dict with the fraction of total (intra-pert) sum-of-squares
    explained by between-phase-group means, pooled over dims:
        frac = sum_d between_ss_d / sum_d total_ss_d.
    Uses population SS (ddof=0), matching how the 12.68 intra-var is computed
    (np.var default). `mean_total_var_per_dim` lets the caller cross-check that
    this pert's total intra-var matches the Test-4 quantity.
    """
    n, d = M.shape
    mu = M.mean(axis=0)
    total_ss_per_dim = ((M - mu) ** 2).sum(axis=0)
    between_ss_per_dim = np.zeros(d, dtype=np.float64)
    for g in np.unique(labels):
        mask = labels == g
        ng = int(mask.sum())
        if ng == 0:
            continue
        mu_g = M[mask].mean(axis=0)
        between_ss_per_dim += ng * (mu_g - mu) ** 2
    total_ss = float(total_ss_per_dim.sum())
    between_ss = float(between_ss_per_dim.sum())
    return {
        "frac": between_ss / max(total_ss, 1e-12),
        "between_ss": between_ss,
        "total_ss": total_ss,
        "mean_total_var_per_dim": float(total_ss / d / max(n, 1)),
    }


def score_variance_explained(M: np.ndarray, cov: np.ndarray) -> dict:
    """Multivariate R² of regressing M on continuous covariates `cov`.

    M : (n, d) features. cov : (n, k) continuous covariates (S/G2M scores).
    Fits M ≈ [1, cov] @ B by least squares and reports the pooled
    1 - SS_res/SS_tot over dims. Threshold-free alternative to the discrete
    phase-bin eta², immune to G1/S/G2M mis-calling.
    """
    n, d = M.shape
    A = np.concatenate([np.ones((n, 1)), cov], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, M, rcond=None)
    pred = A @ coef
    mu = M.mean(axis=0)
    ss_res = float(((M - pred) ** 2).sum())
    ss_tot = float(((M - mu) ** 2).sum())
    return {
        "frac": 1.0 - ss_res / max(ss_tot, 1e-12),
        "ss_resid_reduction": ss_tot - ss_res,
        "ss_tot": ss_tot,
    }


def normality_fail_rate(Z: np.ndarray, alpha: float = 0.01) -> dict:
    """Per-dim D'Agostino's K² normality test on (n_samples, d) array.

    Returns the fraction of dimensions where we REJECT normality at
    significance level `alpha`. Under a true normal distribution this is
    `alpha` by construction (5% at alpha=0.05, 1% at alpha=0.01).
    """
    n, d = Z.shape
    if n < 8:
        return {"fail_rate": float("nan"), "n_dims": d, "note": "too few samples"}
    p_values = np.zeros(d)
    for j in range(d):
        try:
            _, p = stats.normaltest(Z[:, j])
        except Exception:
            p = 0.0
        p_values[j] = p
    fail_count = int((p_values < alpha).sum())
    return {
        "fail_rate": float(fail_count / d),
        "fail_count": fail_count,
        "n_dims": int(d),
        "alpha": float(alpha),
        "median_p": float(np.median(p_values)),
        "min_p": float(np.min(p_values)),
    }


def moments(Z: np.ndarray) -> dict:
    """Per-dim skewness and kurtosis (excess), then averaged."""
    if Z.shape[0] < 4:
        return {"mean_skew": float("nan"), "mean_kurt": float("nan")}
    skew = stats.skew(Z, axis=0, bias=False)        # 0 for Gaussian
    kurt = stats.kurtosis(Z, axis=0, bias=False)    # 0 for Gaussian (excess)
    return {
        "mean_abs_skew": float(np.mean(np.abs(skew))),
        "mean_abs_kurt": float(np.mean(np.abs(kurt))),
        "max_abs_skew": float(np.max(np.abs(skew))),
        "max_abs_kurt": float(np.max(np.abs(kurt))),
    }


@torch.no_grad()
def encode_cells(encoder, X_csr, indices, device, batch_size: int = 512) -> np.ndarray:
    """Encode a set of cells in batches. Returns (n, 256) ndarray."""
    encoder.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        x_dense = X_csr[idx].toarray()
        x = torch.from_numpy(normalize(x_dense)).to(device)
        z = encoder(x).cpu().numpy()
        out.append(z)
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-checkpoint", default="results/v2/A1_phase_b/checkpoint.pt")
    ap.add_argument("--protein-panel-path", default="data/vcc/v2_gene_esm2_panel_pca1280.pt")
    ap.add_argument("--n-controls", type=int, default=5000)
    ap.add_argument("--n-perts-to-test", type=int, default=10)
    ap.add_argument("--min-cells-per-pert", type=int, default=100)
    ap.add_argument("--out", default="results/v2/A1_phase_c/gaussianity_diagnostic.json")
    ap.add_argument("--phase-max-cells", type=int, default=1000,
                    help="max cells per pert for the S1 cell-cycle headroom test")
    ap.add_argument("--phase-out", default="results/nbhd/S1_phase_headroom.json")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}")

    split = load_split("train")

    # Load A1 encoder.
    ckpt = torch.load(args.phase_b_checkpoint, weights_only=False, map_location=device)
    cfg_b = ckpt["config"]
    encoder = MLPEncoder(
        gene_dim=split.n_genes,
        embed_dim=cfg_b["embed_dim"],
        hidden_dim=cfg_b["hidden_dim"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    print(f"loaded encoder from {args.phase_b_checkpoint}")

    rng = np.random.default_rng(0)

    # -------- 1. Control distribution Gaussianity --------
    print("\n=== Test 1: Marginal Gaussianity of encoder output on controls ===")
    ctrl_idx = np.where(split.control_mask)[0]
    sample_idx = rng.choice(ctrl_idx, size=min(args.n_controls, len(ctrl_idx)), replace=False)
    z_ctrl = encode_cells(encoder, split.X, sample_idx, device)
    print(f"  encoded {len(sample_idx)} controls → z shape {z_ctrl.shape}")
    result_ctrl = normality_fail_rate(z_ctrl)
    result_ctrl_moments = moments(z_ctrl)
    print(f"  D'Agostino K² fail rate (p<0.01, expected ~1%): "
          f"{result_ctrl['fail_rate']*100:.1f}% "
          f"({result_ctrl['fail_count']}/{result_ctrl['n_dims']} dims)")
    print(f"  median p-value: {result_ctrl['median_p']:.4f}  "
          f"(uniform under H0; very small means highly non-Gaussian)")
    print(f"  mean |skew|={result_ctrl_moments['mean_abs_skew']:.3f}  "
          f"(0 for Gaussian)")
    print(f"  mean |excess kurt|={result_ctrl_moments['mean_abs_kurt']:.3f}  "
          f"(0 for Gaussian)")

    # -------- 2. Per-perturbation Gaussianity --------
    print("\n=== Test 2: Per-perturbation Gaussianity ===")
    # Pick perturbations with the most cells.
    pert_counts = []
    for pid in range(1, split.n_perts):
        n = int((split.pert_ids == pid).sum())
        if n >= args.min_cells_per_pert:
            pert_counts.append((pid, n))
    pert_counts.sort(key=lambda x: -x[1])
    chosen_perts = pert_counts[:args.n_perts_to_test]
    print(f"  testing {len(chosen_perts)} perts (most cells available)")
    per_pert_results = []
    for pid, n in chosen_perts:
        gname = split.pert_vocab[pid]
        pidx = np.where(split.pert_ids == pid)[0]
        if len(pidx) > 1000:
            pidx = rng.choice(pidx, size=1000, replace=False)
        z_p = encode_cells(encoder, split.X, pidx, device)
        r = normality_fail_rate(z_p)
        m = moments(z_p)
        per_pert_results.append({
            "pert": gname,
            "n_cells": int(len(pidx)),
            "fail_rate": r["fail_rate"],
            "median_p": r["median_p"],
            "mean_abs_skew": m["mean_abs_skew"],
            "mean_abs_kurt": m["mean_abs_kurt"],
        })
        print(f"    {gname:>10s} (n={len(pidx)}): "
              f"fail={r['fail_rate']*100:5.1f}%  "
              f"|skew|={m['mean_abs_skew']:.2f}  "
              f"|kurt|={m['mean_abs_kurt']:.2f}")

    mean_pert_fail = float(np.mean([r["fail_rate"] for r in per_pert_results]))
    print(f"  mean per-pert fail rate: {mean_pert_fail*100:.1f}%")

    # -------- 3. Random-projection baseline --------
    print("\n=== Test 3: Random-projection baseline (the killer test) ===")
    print("  Project raw log1p(CP10k) gene data via a random 18080 → 256 linear")
    print("  projection. CLT predicts each dim should already look quasi-Gaussian.")
    rand_seed = 42
    rand_W = np.random.default_rng(rand_seed).standard_normal(
        size=(split.n_genes, cfg_b["embed_dim"])
    ).astype(np.float32) / np.sqrt(split.n_genes)
    # Re-use the same control sample.
    x_dense = split.X[sample_idx].toarray().astype(np.float32)
    x_norm = normalize(x_dense)
    z_rand = x_norm @ rand_W                                # (n, 256)
    print(f"  random-projected {len(sample_idx)} controls → z shape {z_rand.shape}")
    result_rand = normality_fail_rate(z_rand)
    result_rand_moments = moments(z_rand)
    print(f"  fail rate: {result_rand['fail_rate']*100:.1f}%  "
          f"(vs learned encoder {result_ctrl['fail_rate']*100:.1f}%)")
    print(f"  median p-value: {result_rand['median_p']:.4f}  "
          f"(vs learned {result_ctrl['median_p']:.4f})")
    print(f"  mean |skew|={result_rand_moments['mean_abs_skew']:.3f}  "
          f"(vs learned {result_ctrl_moments['mean_abs_skew']:.3f})")
    print(f"  mean |excess kurt|={result_rand_moments['mean_abs_kurt']:.3f}  "
          f"(vs learned {result_ctrl_moments['mean_abs_kurt']:.3f})")

    # -------- 4. Inter-class vs intra-class variance --------
    print("\n=== Test 4: Inter-class vs intra-class variance (mixture structure) ===")
    centroids = []
    intra_vars = []
    for r_entry in per_pert_results:
        pid_for_name = [pid for pid, n in chosen_perts if split.pert_vocab[pid] == r_entry["pert"]][0]
        pidx = np.where(split.pert_ids == pid_for_name)[0]
        if len(pidx) > 1000:
            pidx = rng.choice(pidx, size=1000, replace=False)
        z_p = encode_cells(encoder, split.X, pidx, device)
        centroids.append(z_p.mean(axis=0))
        intra_vars.append(z_p.var(axis=0).mean())
    centroids = np.array(centroids)
    inter_var = centroids.var(axis=0).mean()
    intra_var = float(np.mean(intra_vars))
    ratio = float(inter_var / max(intra_var, 1e-9))
    print(f"  inter-class variance (across pert centroids): {inter_var:.4f}")
    print(f"  intra-class variance (within pert):           {intra_var:.4f}")
    print(f"  ratio (inter/intra): {ratio:.3f}  "
          f"(>>1 → strongly clustered; ≈0 → indistinguishable from one Gaussian)")

    # -------- 5. Cell-cycle phase headroom (S1) --------
    # How much of the within-pert variance (the 12.68) is just cell-cycle
    # phase? If phase explains ~all of it, mid-k KNN neighborhoods ≈ phase bins
    # and H1 headroom is low. Measured in BOTH latent space (consistent with
    # how 12.68 is defined) and gene space (closer to what DES sees).
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    print("\n=== Test 5: Cell-cycle phase headroom (S1) ===")
    var_set = set(split.var_names)
    s_present = [g for g in TIROSH_S_GENES if g in var_set]
    g2m_present = [g for g in TIROSH_G2M_GENES if g in var_set]
    print(f"  Tirosh markers matched to panel: "
          f"S {len(s_present)}/{len(TIROSH_S_GENES)}, "
          f"G2M {len(g2m_present)}/{len(TIROSH_G2M_GENES)}")

    # Deterministic per-pert sampling for this test (independent of Test 2/4).
    phase_rng = np.random.default_rng(123)
    pert_idx_lists = []
    pert_pids = []
    for pid, _n in chosen_perts:
        pidx = np.where(split.pert_ids == pid)[0]
        if len(pidx) > args.phase_max_cells:
            pidx = phase_rng.choice(pidx, size=args.phase_max_cells, replace=False)
        pert_idx_lists.append(pidx)
        pert_pids.append(pid)
    all_idx = np.concatenate(pert_idx_lists)
    offsets = np.cumsum([0] + [len(p) for p in pert_idx_lists])

    # Gene-space log1p(CP10k) for all cells, scored for phase in one pass so the
    # scanpy expression-bin reference is shared across perts.
    x_norm_all = normalize(split.X[all_idx].toarray())
    adata = ad.AnnData(
        X=x_norm_all,
        var=pd.DataFrame(index=list(split.var_names)),
    )
    sc.tl.score_genes_cell_cycle(
        adata, s_genes=s_present, g2m_genes=g2m_present
    )
    phase_all = adata.obs["phase"].to_numpy().astype(str)
    s_score = adata.obs["S_score"].to_numpy().astype(np.float64)
    g2m_score = adata.obs["G2M_score"].to_numpy().astype(np.float64)
    z_all = encode_cells(encoder, split.X, all_idx, device)

    phase_counts = {str(p): int((phase_all == p).sum()) for p in np.unique(phase_all)}
    g1_frac = phase_counts.get("G1", 0) / len(phase_all)
    print(f"  phase distribution (all {len(phase_all)} cells): {phase_counts}")
    if g1_frac < 0.05:
        print(f"  ⚠  only {g1_frac*100:.2f}% of cells called G1 — discrete-phase calling")
        print("     is likely mis-calibrated; trust the continuous S/G2M-score measure")
        print("     below, which is immune to the G1/S/G2M thresholding.")

    # Discrete (phase-bin) and continuous (S/G2M-score regression) measures.
    # The continuous one regresses each feature dim on [1, S_score, G2M_score]
    # and reports 1 - SS_res/SS_tot — a threshold-free upper-ish bound on how
    # much within-pert variance the cell-cycle axis can explain.
    per_pert_phase = []
    num_lat = den_lat = 0.0
    num_gene = den_gene = 0.0
    cnum_lat = cden_lat = 0.0
    cnum_gene = cden_gene = 0.0
    intra_var_latent_check = []
    for i, pid in enumerate(pert_pids):
        sl = slice(offsets[i], offsets[i + 1])
        labels = phase_all[sl]
        cov = np.stack([s_score[sl], g2m_score[sl]], axis=1)
        lat = phase_variance_explained(z_all[sl], labels)
        gene = phase_variance_explained(x_norm_all[sl], labels)
        clat = score_variance_explained(z_all[sl], cov)
        cgene = score_variance_explained(x_norm_all[sl], cov)
        num_lat += lat["between_ss"]; den_lat += lat["total_ss"]
        num_gene += gene["between_ss"]; den_gene += gene["total_ss"]
        cnum_lat += clat["ss_resid_reduction"]; cden_lat += clat["ss_tot"]
        cnum_gene += cgene["ss_resid_reduction"]; cden_gene += cgene["ss_tot"]
        intra_var_latent_check.append(lat["mean_total_var_per_dim"])
        per_pert_phase.append({
            "pert": split.pert_vocab[pid],
            "n_cells": int(offsets[i + 1] - offsets[i]),
            "phase_counts": {str(p): int((labels == p).sum()) for p in np.unique(labels)},
            "frac_latent": lat["frac"],
            "frac_gene": gene["frac"],
            "frac_latent_score": clat["frac"],
            "frac_gene_score": cgene["frac"],
        })
        print(f"    {split.pert_vocab[pid]:>10s} (n={offsets[i+1]-offsets[i]}): "
              f"discrete latent={lat['frac']*100:4.1f}% gene={gene['frac']*100:4.1f}% | "
              f"score latent={clat['frac']*100:4.1f}% gene={cgene['frac']*100:4.1f}%")

    pooled_frac_latent = float(num_lat / max(den_lat, 1e-12))
    pooled_frac_gene = float(num_gene / max(den_gene, 1e-12))
    pooled_frac_latent_score = float(cnum_lat / max(cden_lat, 1e-12))
    pooled_frac_gene_score = float(cnum_gene / max(cden_gene, 1e-12))
    mean_frac_latent = float(np.mean([p["frac_latent"] for p in per_pert_phase]))
    mean_frac_gene = float(np.mean([p["frac_gene"] for p in per_pert_phase]))
    intra_var_latent_mean = float(np.mean(intra_var_latent_check))

    print(f"  pooled DISCRETE phase-explained: "
          f"latent={pooled_frac_latent*100:.1f}%  gene={pooled_frac_gene*100:.1f}%")
    print(f"  pooled CONTINUOUS S/G2M-score-explained: "
          f"latent={pooled_frac_latent_score*100:.1f}%  gene={pooled_frac_gene_score*100:.1f}%")
    print(f"  cross-check: mean latent intra-var/dim here = {intra_var_latent_mean:.2f} "
          f"(Test 4 reported {intra_var:.2f}; should be the same ballpark)")

    # -------- Interpretation --------
    print("\n=== Interpretation ===")
    if result_ctrl["fail_rate"] < 0.05:
        if result_rand["fail_rate"] >= result_ctrl["fail_rate"]:
            print("  ✓ Encoder output IS approximately Gaussian.")
            if result_rand["fail_rate"] > result_ctrl["fail_rate"] * 2:
                print("  ⚠  But random projection is MORE non-Gaussian — encoder is")
                print("     actively *enforcing* Gaussianity beyond what the data has.")
                print("     This is the paper's failure mode: we may have thrown away")
                print("     non-Gaussian structure that mattered.")
            else:
                print("  ✓ Random projection is comparably Gaussian — encoder hasn't")
                print("     over-enforced. The paper's worst case doesn't apply directly.")
        else:
            print("  Unusual: random projection is MORE Gaussian than learned encoder.")
    else:
        print(f"  ⚠  Encoder output is NOT Gaussian "
              f"({result_ctrl['fail_rate']*100:.1f}% of dims reject normality).")
        print("     Despite our MCR² prior, biology is leaking through. The encoder")
        print("     is in a constrained-trade-off solution that's neither faithful to")
        print("     data nor isotropic-Gaussian.")

    if ratio < 0.1:
        print(f"\n  ⚠  Inter/intra variance ratio is {ratio:.3f} — perturbations are")
        print("     barely separable in latent space. The encoder isn't producing")
        print("     a useful mixture structure for downstream prediction.")
    else:
        print(f"\n  Inter/intra ratio {ratio:.3f}: perts have some clustering structure.")

    # Use the continuous score measure for the verdict (more trustworthy given
    # the degenerate discrete phase calling); take the max of the two as a
    # conservative upper bound on phase's share.
    phase_share = max(pooled_frac_latent_score, pooled_frac_latent)
    if phase_share > 0.7:
        print(f"\n  ⚠  Cell-cycle explains ~{phase_share*100:.0f}% of within-pert latent")
        print("     variance — neighborhoods would mostly recover phase bins.")
        print("     H1 headroom is LOW. Proceed with low expectations (not a kill).")
    else:
        print(f"\n  Cell-cycle explains only ~{phase_share*100:.0f}% of within-pert latent")
        print("     variance (continuous score measure; discrete bins agree). Substantial")
        print("     non-phase structure remains for neighborhoods to exploit — H1 headroom")
        print("     is plausible and is NOT dominated by cell-cycle phase.")

    # -------- Persist --------
    payload = {
        "checkpoint": args.phase_b_checkpoint,
        "n_controls": int(len(sample_idx)),
        "control_test": {**result_ctrl, **result_ctrl_moments},
        "per_pert_tests": per_pert_results,
        "mean_pert_fail_rate": mean_pert_fail,
        "random_projection_test": {**result_rand, **result_rand_moments},
        "inter_intra_variance_ratio": ratio,
        "inter_var": float(inter_var),
        "intra_var": float(intra_var),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out_path}")

    # S1 headroom payload (separate file for the neighborhood experiment).
    phase_payload = {
        "checkpoint": args.phase_b_checkpoint,
        "n_perts_tested": len(per_pert_phase),
        "phase_max_cells": args.phase_max_cells,
        "tirosh_markers_matched": {
            "s": len(s_present), "s_total": len(TIROSH_S_GENES),
            "g2m": len(g2m_present), "g2m_total": len(TIROSH_G2M_GENES),
        },
        "phase_distribution": phase_counts,
        "g1_fraction": g1_frac,
        "discrete_phase_calling_suspect": bool(g1_frac < 0.05),
        "pooled_frac_phase_explained_latent": pooled_frac_latent,
        "pooled_frac_phase_explained_gene": pooled_frac_gene,
        "pooled_frac_score_explained_latent": pooled_frac_latent_score,
        "pooled_frac_score_explained_gene": pooled_frac_gene_score,
        "mean_per_pert_frac_latent": mean_frac_latent,
        "mean_per_pert_frac_gene": mean_frac_gene,
        "latent_intra_var_per_dim_check": intra_var_latent_mean,
        "test4_intra_var": float(intra_var),
        "per_pert": per_pert_phase,
    }
    phase_out = Path(args.phase_out)
    phase_out.parent.mkdir(parents=True, exist_ok=True)
    phase_out.write_text(json.dumps(phase_payload, indent=2, default=float))
    print(f"wrote {phase_out}")


if __name__ == "__main__":
    main()
