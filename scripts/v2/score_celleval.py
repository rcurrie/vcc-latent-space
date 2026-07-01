"""Score the hybrid simple-head model with the OFFICIAL cell-eval metrics.

This is the RESCOPE step for the neighborhood experiment (docs/des_neighborhood_plan.md).
S0 found that the in-repo DES collapses each perturbation to a *mean* profile before
scoring, so any within-population spread is invisible to it — the neighborhood thesis
can't pay off against that metric. cell-eval's DES (`overlap_at_N`) instead runs a
real differential-expression test (via pdex) on the predicted *population* vs the
predicted controls, so it rewards spread. Using it requires two changes, both here:

  1. Inference emits a POPULATION, not a single mean. We apply the per-perturbation
     predicted delta to each of N sampled control cells (delta is constant per pert for
     the simple head, so the predicted spread is the control population's spread — an
     honest floor that the neighborhood work, S2+, will lift by making delta vary per
     neighborhood).
  2. Predicted values are clamped to >= 0. A signed delta can push a downregulated gene
     below zero in log1p space; cell-eval's lognorm validation rejects negatives.

Everything stays in log1p(CP10k) space (what the model predicts and what cell-eval
detects as already-lognorm, skipping its own re-normalization).

    uv run python scripts/v2/score_celleval.py --split val
    uv run python scripts/v2/score_celleval.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cell_eval import MetricsEvaluator

from lewm.v2.data import load_split, normalize
from lewm.v2.models import MLPEncoder, ProteinActionEmbedV2
from lewm.v2.train_phase_a import select_device

CONTROL = "non-targeting"


class SimpleDeltaHead(nn.Module):
    """Mirror of the simple head defined inside train_hybrid.train_hybrid:
    a zero-initialized Linear(action_dim -> gene_dim) that ignores z_ctrl."""

    def __init__(self, action_dim: int, gene_dim: int):
        super().__init__()
        self.lin = nn.Linear(action_dim, gene_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.lin(action)


def build_anndata(blocks: list[tuple[str, np.ndarray]], var_names: list[str]) -> ad.AnnData:
    """Stack (target_label, (n, G) float32) blocks into one AnnData with obs['target']."""
    X = np.vstack([b for _, b in blocks]).astype(np.float32)
    targets = np.concatenate([[lbl] * b.shape[0] for lbl, b in blocks])
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=list(var_names)))
    adata.obs["target"] = targets
    return adata


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/v2/hybrid_simple_sanity/checkpoint.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n-ctrl-base", type=int, default=2000,
                    help="pool of training control cells to build predicted populations from")
    ap.add_argument("--n-pred-per-pert", type=int, default=256,
                    help="predicted cells emitted per perturbation (and predicted controls)")
    ap.add_argument("--n-real-per-pert", type=int, default=500,
                    help="real cells per perturbation fed to cell-eval (cap)")
    ap.add_argument("--n-real-ctrl", type=int, default=2000)
    ap.add_argument("--base", choices=["cells", "neighborhoods"], default="cells",
                    help="population base: control cells (RESCOPE floor) or control "
                         "neighborhood pseudo-bulks (S4 spread test)")
    ap.add_argument("--k", type=int, default=50, help="neighborhood size (--base neighborhoods)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}  split: {args.split}")
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    head = cfg.get("head_type") or ("simple" if cfg.get("use_simple_head") else "adaln")
    if head != "simple":
        raise NotImplementedError(
            f"this scorer only supports the simple head; checkpoint is '{head}'"
        )

    # --- control base population (predicted cells are this base + delta) ---
    #   cells         : sampled training control cells (RESCOPE floor)
    #   neighborhoods : control NEIGHBORHOOD pseudo-bulks on the frozen encoder
    #                   (S4 — tighter, neighborhood-level spread the DES can reward)
    train_split = load_split("train")
    var_names = list(train_split.var_names)
    var_to_col = {g: i for i, g in enumerate(var_names)}
    ctrl_idx = np.where(train_split.control_mask)[0]
    pick = np.sort(rng.choice(ctrl_idx, size=min(args.n_ctrl_base, len(ctrl_idx)), replace=False))
    if args.base == "cells":
        ctrl_pool = normalize(train_split.X[pick].toarray())      # (P, G) log1p CP10k
    else:
        from lewm.neighborhoods import build_knn_neighborhoods, neighborhood_pseudobulks
        enc = MLPEncoder(train_split.n_genes, embed_dim=cfg["embed_dim"],
                         hidden_dim=cfg["hidden_dim"]).to(device)
        enc.load_state_dict(ckpt["encoder"]); enc.eval()
        z = []
        for s in range(0, len(pick), 512):
            xb = torch.from_numpy(normalize(train_split.X[pick[s:s+512]].toarray())).to(device)
            z.append(enc(xb).cpu().numpy())
        z = np.concatenate(z, axis=0)
        nbhds = build_knn_neighborhoods(z, k=args.k, prop=0.1, seed=args.seed)
        ctrl_pool = neighborhood_pseudobulks(train_split.X[pick], nbhds)  # (n_nb, G)
    del train_split
    P = ctrl_pool.shape[0]
    print(f"control base ({args.base}): {ctrl_pool.shape}")

    # --- model (simple head ignores the encoder, so we skip loading it) ---
    panel = torch.load(cfg["protein_panel_path"], weights_only=False, map_location="cpu")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"], coverage=panel["coverage"],
        action_dim=cfg["action_dim"],
    ).to(device)
    action_embed.load_state_dict(ckpt["action_embed"])
    action_embed.eval()
    delta_head = SimpleDeltaHead(cfg["action_dim"], len(var_names)).to(device)
    delta_head.load_state_dict(ckpt["delta_head"])
    delta_head.eval()

    # --- eval split ---
    eval_split = load_split(args.split)
    if list(eval_split.var_names) != var_names:
        raise RuntimeError("eval var_names != train var_names")
    eval_perts = sorted(set(int(p) for p in eval_split.pert_ids if p != 0))

    pred_blocks: list[tuple[str, np.ndarray]] = []
    real_blocks: list[tuple[str, np.ndarray]] = []
    n_used = 0
    for pid in eval_perts:
        gene = eval_split.pert_vocab[pid]
        col = var_to_col.get(gene, -1)
        if col == -1:
            continue  # target gene not in panel — can't form a delta

        gene_idx = torch.tensor([col], dtype=torch.long, device=device)
        delta = delta_head(action_embed(gene_idx)).squeeze(0).cpu().numpy()  # (G,)

        # Predicted population = control base + delta, clamped >= 0. Neighborhoods
        # mode uses ALL control-neighborhood pseudo-bulks; cells mode samples N.
        if args.base == "neighborhoods":
            x_pred = np.clip(ctrl_pool + delta, 0.0, None)
        else:
            sub = rng.choice(P, size=args.n_pred_per_pert, replace=args.n_pred_per_pert > P)
            x_pred = np.clip(ctrl_pool[sub] + delta, 0.0, None)
        pred_blocks.append((gene, x_pred))

        # Real population for this pert (capped).
        pos = np.where(eval_split.pert_ids == pid)[0]
        if len(pos) > args.n_real_per_pert:
            pos = rng.choice(pos, size=args.n_real_per_pert, replace=False)
        real_blocks.append((gene, normalize(eval_split.X[pos].toarray())))
        n_used += 1

    # Controls in both pred and real (needed for the per-side DE reference).
    if args.base == "neighborhoods":
        pred_ctrl = ctrl_pool
    else:
        pred_ctrl = ctrl_pool[rng.choice(P, size=args.n_pred_per_pert, replace=args.n_pred_per_pert > P)]
    pred_blocks.append((CONTROL, pred_ctrl))
    rc = np.where(eval_split.control_mask)[0]
    rc = rng.choice(rc, size=min(args.n_real_ctrl, len(rc)), replace=False)
    real_blocks.append((CONTROL, normalize(eval_split.X[rc].toarray())))

    print(f"perts scored: {n_used}  building AnnData...")
    adata_pred = build_anndata(pred_blocks, var_names)
    adata_real = build_anndata(real_blocks, var_names)
    print(f"pred: {adata_pred.shape}  real: {adata_real.shape}")

    suffix = args.split if args.base == "cells" else f"{args.split}_{args.base}_k{args.k}"
    out_dir = Path(args.out or f"results/nbhd/celleval_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = MetricsEvaluator(
        adata_pred, adata_real,
        pert_col="target", control_pert=CONTROL,
        outdir=str(out_dir),
    )
    per_pert, agg = ev.compute(profile="vcc", write_csv=True)

    means = {r["statistic"]: r for r in agg.iter_rows(named=True)}["mean"]
    des = float(means["overlap_at_N"])
    pds = float(means["discrimination_score_l1"])
    mae = float(means["mae"])
    print("\n=== OFFICIAL cell-eval (vcc profile) ===")
    print(f"  perts:  {n_used}")
    print(f"  DES (overlap_at_N):            {des:.4f}   (centroid-approx baseline: 0.060)")
    print(f"  PDS (discrimination_score_l1): {pds:.4f}   (centroid-approx baseline: 0.538)")
    print(f"  MAE:                           {mae:.4f}")

    summary = {
        "split": args.split,
        "base": args.base,
        "k": args.k if args.base == "neighborhoods" else None,
        "checkpoint": args.checkpoint,
        "n_perts": n_used,
        "pred_pop_size": int(P),
        "n_real_per_pert": args.n_real_per_pert,
        "des_overlap_at_N": des,
        "pds_discrimination_score_l1": pds,
        "mae": mae,
        "note": f"official cell-eval; population base={args.base}; delta clamped >=0",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/summary.json (+ cell-eval CSVs)")


if __name__ == "__main__":
    main()
