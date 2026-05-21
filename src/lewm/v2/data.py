"""VCC data loading + augmentation for v2.

We reuse v1's load_split / VCCSplit / collate / StratifiedPerturbationSampler
because the underlying split files and pert vocabulary are stable. v2 adds:

  - BinomialSubsample : drop UMI counts at rate (1-p) before normalization.
                        Operates on integer-valued count vectors (the h5ad's
                        adata.X is integer-valued float32 — verified Phase 0).
                        Mathematically equivalent to sequencing at depth p·D.
                        Used to generate paired views (x, τ(x)) for the
                        augmentation-invariance loss.

  - V2Dataset         : yields a single cell as (x_view1, x_view2, pert_id,
                        batch_id, is_control), where view1 and view2 are
                        independent binomial-subsampled-then-log1p-CP10k views
                        of the same raw count vector. Setting tau=1.0 disables
                        subsampling and returns x_view1 == x_view2 == log1p(CP10k(x)),
                        useful for eval and Phase C.

The internal-val split (15 held-out perturbations carved from the 150 training
perts) lives in JSON at data/vcc/v2_internal_val_split.json; see splits.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Re-export v1 primitives we keep using unchanged.
from lewm.data import (  # noqa: F401  (used by callers)
    CONTROL_LABEL,
    VCCSplit,
    collate_dense,
    load_split,
    normalize,
    StratifiedPerturbationSampler,
)


def binomial_subsample(
    counts: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial subsample of an integer count vector.

    For each gene g, c'_g ~ Binomial(c_g, p). Equivalent to sequencing the
    same cell at depth p * D where D = sum(counts). Mass is conserved in
    expectation: E[c'_g] = p * c_g.

    Parameters
    ----------
    counts : (G,) or (B, G) float/int array. Assumed integer-valued.
    p      : retention probability in (0, 1].
    rng    : numpy Generator.

    Returns
    -------
    same-shape array, integer-valued (returned as float32 for downstream
    log1p / sparse compatibility).
    """
    if p >= 1.0:
        return counts.astype(np.float32, copy=False)
    if p <= 0.0:
        return np.zeros_like(counts, dtype=np.float32)
    # rng.binomial accepts an integer-valued ndarray for `n` and broadcasts.
    n = counts.astype(np.int64, copy=False)
    sub = rng.binomial(n=n, p=p).astype(np.float32, copy=False)
    return sub


@dataclass
class AugmentConfig:
    """Binomial-subsample augmentation config.

    tau : retention probability per view. tau=1.0 disables subsampling
          (returns raw counts unchanged).
    paired_views : if True, return two independently subsampled views.
                   Used for the augmentation-invariance loss.
    """
    tau: float = 0.5
    paired_views: bool = True


class V2Dataset(Dataset):
    """Per-cell dataset returning paired binomial-subsampled views.

    Each item:
      x1, x2 : (G,) float32 tensors. log1p(CP10k(binomial_subsample(raw))).
               If aug.paired_views is False, x1 == x2.
               If aug.tau == 1.0, both views are the same deterministic
               log1p(CP10k(raw)) — useful for eval and Phase C.
      pert_id, batch_id, is_control : ints / bool

    Subsample happens at the count level, BEFORE log1p(CP10k). This is the
    biologically faithful augmentation (matches the actual sequencing-depth
    noise process) and is the augmentation choice advocated by the v2 sketch.
    """

    def __init__(
        self,
        split: VCCSplit,
        indices: np.ndarray | None = None,
        aug: AugmentConfig | None = None,
        seed: int = 0,
    ):
        self.split = split
        self.indices = (
            np.arange(split.n_cells, dtype=np.int64) if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.aug = aug or AugmentConfig()
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def _view(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sub = binomial_subsample(raw, self.aug.tau, rng)
        return normalize(sub[None, :])[0]

    def __getitem__(self, i: int):
        cell_idx = int(self.indices[i])
        raw = self.split.X[cell_idx].toarray().ravel().astype(np.float32)

        # Fresh-entropy RNG per __getitem__ call. With num_workers > 0 each
        # worker process draws independent system entropy, and within a
        # worker successive calls also draw fresh entropy, so we never
        # repeat augmentations across epochs. Trades bit-reproducibility
        # for actually-different augmentation each time the same cell is
        # sampled. ~3us cost per call.
        rng = np.random.default_rng()

        if self.aug.tau >= 1.0 or not self.aug.paired_views:
            v = self._view(raw, rng)
            x1 = x2 = torch.from_numpy(v)
        else:
            v1 = self._view(raw, rng)
            v2 = self._view(raw, rng)
            x1 = torch.from_numpy(v1)
            x2 = torch.from_numpy(v2)

        return (
            x1,
            x2,
            int(self.split.pert_ids[cell_idx]),
            int(self.split.batch_ids[cell_idx]),
            bool(self.split.pert_ids[cell_idx] == 0),
        )


def collate_v2(batch):
    """Stack paired-view items from V2Dataset."""
    x1s, x2s, pert_ids, batch_ids, is_control = zip(*batch)
    return (
        torch.stack(x1s, dim=0),
        torch.stack(x2s, dim=0),
        torch.tensor(pert_ids, dtype=torch.long),
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(is_control, dtype=torch.bool),
    )
