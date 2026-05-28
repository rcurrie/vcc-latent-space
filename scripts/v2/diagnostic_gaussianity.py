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


if __name__ == "__main__":
    main()
