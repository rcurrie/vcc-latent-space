"""Phase 1 smoke test.

Validates the data path end-to-end:
  1. Load training h5ad into memory
  2. Build VCCDataset + StratifiedPerturbationSampler
  3. Iterate one epoch, timing batches and embedding through a randomly
     initialized MLP on MPS
  4. Hold out 10 perturbations as internal val and verify split sanity

Run:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `from lewm.data import ...` without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lewm.data import (
    VCCDataset,
    StratifiedPerturbationSampler,
    collate_dense,
    load_split,
    make_internal_val_split,
)


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    print("\n[1] loading training split ...")
    t0 = time.perf_counter()
    split = load_split("train")
    print(
        f"    loaded in {time.perf_counter() - t0:.1f}s: "
        f"{split.n_cells} cells x {split.n_genes} genes, "
        f"{split.n_perts} perturbations (incl control)"
    )
    csr = split.X
    csr_mem_gb = (csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) / 1e9
    print(f"    X CSR memory: {csr_mem_gb:.2f}GB, nnz: {csr.nnz / 1e6:.1f}M")
    n_control = int(split.control_mask.sum())
    print(f"    {n_control} non-targeting cells, {split.n_cells - n_control} perturbed")

    print("\n[2] holding out 10 perturbations for internal val ...")
    train_idx, val_idx, holdout_names = make_internal_val_split(split, n_holdout_perts=10)
    print(f"    train: {len(train_idx)} cells, val: {len(val_idx)} cells")
    print(f"    holdout perts: {holdout_names}")
    assert split.control_mask[train_idx].sum() == split.control_mask.sum(), (
        "all controls should be in train"
    )

    print("\n[3] building dataset + stratified sampler ...")
    train_dataset = VCCDataset(split, indices=train_idx)
    sampler = StratifiedPerturbationSampler(
        train_dataset,
        batch_size=512,
        n_perts_per_batch=8,
        control_fraction=0.25,
        n_batches=20,  # short smoke run, not full epoch
        seed=0,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_dense,
    )
    print(
        f"    batch_size={sampler.batch_size_actual} "
        f"(n_control={sampler.n_control}, "
        f"n_perts_per_batch={sampler.n_perts_per_batch}, "
        f"n_per_pert={sampler.n_per_pert})"
    )
    print(f"    n_batches: {sampler.n_batches}")

    print("\n[4] randomly initialized MLP forward pass ...")
    encoder = nn.Sequential(
        nn.Linear(split.n_genes, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"    encoder params: {n_params/1e6:.1f}M")

    print("\n[5] iterating ...")
    t0 = time.perf_counter()
    n_seen = 0
    pert_id_counts = np.zeros(split.n_perts, dtype=np.int64)
    for i, (x, pert_id, batch_id, is_control) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        z = encoder(x)
        n_seen += x.shape[0]
        for p in pert_id.numpy():
            pert_id_counts[p] += 1
        if i < 3 or i == sampler.n_batches - 1:
            ctrl_in_batch = int(is_control.sum())
            unique_perts = int(len(set(pert_id.tolist())))
            print(
                f"    batch {i:3d}: x={tuple(x.shape)}, z={tuple(z.shape)}, "
                f"controls={ctrl_in_batch}, unique_perts={unique_perts}, "
                f"x.mean={x.mean().item():.3f}, x.std={x.std().item():.3f}"
            )
    elapsed = time.perf_counter() - t0
    n_ctrl_seen = int(pert_id_counts[0])
    print(f"\n    {sampler.n_batches} batches, {n_seen} cells, {elapsed:.1f}s "
          f"({elapsed/sampler.n_batches*1000:.0f}ms/batch)")
    print(f"    cells seen: {n_seen}; controls: {n_ctrl_seen}; "
          f"non-control perts encountered: {(pert_id_counts[1:] > 0).sum()}")

    print("\nsmoke test ok")


if __name__ == "__main__":
    main()
