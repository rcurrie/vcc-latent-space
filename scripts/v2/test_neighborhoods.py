"""S2 gate: unit-test the neighborhood primitive (src/lewm/neighborhoods.py).

Standalone smoke-test (the repo has no pytest harness). Verifies:

  1. Pseudo-bulk gate (the plan's go/no-go): the count-corrected pseudo-bulk of
     an ALL-CELLS neighborhood reproduces the existing global control centroid.
     Note the existing centroid is a mean-of-logs; the count pseudo-bulk is
     sum->CP10k->log1p. These are NOT algebraically equal (log1p is concave,
     and depth-weighting differs), so the gate is a high-correlation / small-L1
     tolerance, not bit-equality. We print both and assert correlation > 0.99.

  2. Self-consistency: neighborhood_pseudobulks over a single all-cells
     neighborhood equals the direct pseudobulk_lognorm of the same cells
     (catches indexing bugs). Bit-equal.

  3. Overlapping-KNN structure: neighborhoods have size k+1, indices are valid,
     and they actually overlap (some cell belongs to >1 neighborhood).

    uv run python scripts/v2/test_neighborhoods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch

from lewm.data import normalize
from lewm.neighborhoods import (
    build_knn_neighborhoods,
    neighborhood_pseudobulks,
    pseudobulk_lognorm,
)
from lewm.v2.data import load_split
from lewm.v2.models import MLPEncoder
from lewm.v2.train_phase_a import select_device

ENCODER_CKPT = "results/v2/phase_a/checkpoint.pt"
N_SAMPLE = 5000
K = 50


@torch.no_grad()
def embed(encoder, X_csr, idx, device, bs=512):
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(normalize(X_csr[idx[s:s + bs]].toarray())).to(device)
        out.append(encoder(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    device = select_device()
    print(f"device: {device}")
    rng = np.random.default_rng(0)

    split = load_split("val")
    ctrl = np.where(split.control_mask)[0]
    pick = np.sort(rng.choice(ctrl, size=min(N_SAMPLE, len(ctrl)), replace=False))
    raw = split.X[pick]                                  # (N, G) sparse raw UMIs
    print(f"sampled {raw.shape[0]} control cells x {raw.shape[1]} genes")

    # ---- 1. Pseudo-bulk gate ----
    count_pb = pseudobulk_lognorm(raw)                   # sum -> CP10k -> log1p
    centroid_meanlogs = normalize(raw.toarray()).mean(axis=0)  # existing global centroid
    corr = float(np.corrcoef(count_pb, centroid_meanlogs)[0, 1])
    l1 = float(np.abs(count_pb - centroid_meanlogs).sum())
    max_abs = float(np.abs(count_pb - centroid_meanlogs).max())
    mean_abs = float(np.abs(count_pb - centroid_meanlogs).mean())
    print("\n[1] count pseudo-bulk vs existing mean-of-logs centroid:")
    print(f"    pearson corr : {corr:.5f}")
    print(f"    L1 (sum)     : {l1:.3f}   mean|Δ|: {mean_abs:.5f}   max|Δ|: {max_abs:.4f}")

    # ---- 2. Self-consistency (bit-equal) ----
    nb_pb = neighborhood_pseudobulks(raw, [np.arange(raw.shape[0])])[0]
    selfconsistent = np.allclose(nb_pb, count_pb, atol=1e-6)
    print(f"\n[2] all-cells-via-neighborhoods == direct pseudobulk: {selfconsistent}")

    # ---- 3. Overlapping-KNN structure ----
    emb = MLPEncoder(split.n_genes, embed_dim=256, hidden_dim=512).to(device)
    ck = torch.load(ENCODER_CKPT, weights_only=False, map_location=device)
    emb.load_state_dict(ck["encoder"]); emb.eval()
    Z = embed(emb, split.X, pick, device)
    nbhds = build_knn_neighborhoods(Z, k=K, prop=0.1, seed=0)
    sizes = np.array([len(nb) for nb in nbhds])
    membership = np.bincount(np.concatenate(nbhds), minlength=len(pick))
    max_member = int(membership.max())
    all_valid = bool((np.concatenate(nbhds) < len(pick)).all())
    print(f"\n[3] neighborhoods: {len(nbhds)}  size(min/max)={sizes.min()}/{sizes.max()} "
          f"(expect {K + 1})")
    print(f"    cells covered: {(membership > 0).sum()}/{len(pick)}  "
          f"max memberships per cell: {max_member}  indices valid: {all_valid}")

    # ---- gate ----
    ok = (corr > 0.99) and selfconsistent and bool((sizes == K + 1).all()) \
        and (max_member > 1) and all_valid
    print(f"\nS2 GATE: {'PASS' if ok else 'FAIL'}")

    import json
    out = Path("results/nbhd/S2_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_cells": int(raw.shape[0]), "k": K,
        "pseudobulk_vs_centroid_corr": corr,
        "pseudobulk_vs_centroid_mean_abs": mean_abs,
        "pseudobulk_vs_centroid_max_abs": max_abs,
        "self_consistent": bool(selfconsistent),
        "n_neighborhoods": len(nbhds),
        "neighborhood_size": int(sizes.max()),
        "max_memberships_per_cell": max_member,
        "cells_covered": int((membership > 0).sum()),
        "gate_pass": bool(ok),
    }, indent=2))
    print(f"wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
