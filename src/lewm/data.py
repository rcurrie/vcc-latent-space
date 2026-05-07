"""VCC 2025 data loading.

The h5ad files contain raw UMI counts (sparse CSR). After loading, we keep
the CSR in memory and dense-ify per batch. Normalization (log1p of CP10k) is
applied on the fly inside the dataset.

Memory budget on M4 / 32GB:
    train: ~15.5GB CSR (1.93B nnz)
    val  : ~7GB
    test : ~12GB
    -> load one split at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, Sampler

DATA_DIR = Path("data/vcc")
SPLIT_FILES = {
    "train": "adata_Training.h5ad",
    "val": "adata_Validation.h5ad",
    "test": "adata_Test.h5ad",
}
CONTROL_LABEL = "non-targeting"


@dataclass
class VCCSplit:
    """A loaded VCC split: CSR expression + per-cell metadata.

    target_genes : str array, length n_cells. The gene name targeted by
                   the perturbation, or "non-targeting" for controls.
    pert_ids     : int array. Encodes target_genes via pert_vocab.
    pert_vocab   : list of str. pert_vocab[pert_ids[i]] == target_genes[i].
                   Index 0 is reserved for the control class.
    batch_ids    : int array. Encodes obs.batch (plate / capture batch).
    var_names    : list of str, length n_genes. Gene names matching X columns.
    """
    X: sp.csr_matrix          # (n_cells, n_genes), float32, raw UMI counts
    target_genes: np.ndarray  # (n_cells,) object/str
    pert_ids: np.ndarray      # (n_cells,) int64
    pert_vocab: list[str]     # control at index 0
    batch_ids: np.ndarray     # (n_cells,) int64
    batch_vocab: list[str]
    var_names: list[str]

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    @property
    def n_perts(self) -> int:
        return len(self.pert_vocab)

    @property
    def control_mask(self) -> np.ndarray:
        return self.pert_ids == 0


def load_split(
    split: str,
    data_dir: Path | str = DATA_DIR,
    pert_vocab: list[str] | None = None,
) -> VCCSplit:
    """Load one VCC split into memory.

    Parameters
    ----------
    split: 'train' | 'val' | 'test'
    pert_vocab: optional. If given, encode target_genes against this vocab
        instead of building a fresh one. Use this to keep pert IDs consistent
        across splits at submission time. Unknown genes get id -1.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}, expected one of {list(SPLIT_FILES)}")

    path = Path(data_dir) / SPLIT_FILES[split]
    a = ad.read_h5ad(path)
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()
    X = X.astype(np.float32, copy=False)

    target_genes = a.obs["target_gene"].astype(str).to_numpy()
    if pert_vocab is None:
        unique = sorted(g for g in set(target_genes) if g != CONTROL_LABEL)
        pert_vocab = [CONTROL_LABEL] + unique
    pert_to_id = {g: i for i, g in enumerate(pert_vocab)}
    pert_ids = np.array(
        [pert_to_id.get(g, -1) for g in target_genes], dtype=np.int64
    )
    if (pert_ids == -1).any():
        n_unk = int((pert_ids == -1).sum())
        unique_unk = sorted(set(target_genes[pert_ids == -1]))
        print(
            f"  warning: {n_unk} cells have target_gene not in vocab "
            f"({len(unique_unk)} unique unknowns: {unique_unk[:5]}...)"
        )

    batch_strs = a.obs["batch"].astype(str).to_numpy()
    batch_vocab = sorted(set(batch_strs))
    batch_to_id = {b: i for i, b in enumerate(batch_vocab)}
    batch_ids = np.array([batch_to_id[b] for b in batch_strs], dtype=np.int64)

    var_names = list(a.var_names.astype(str))

    return VCCSplit(
        X=X,
        target_genes=target_genes,
        pert_ids=pert_ids,
        pert_vocab=pert_vocab,
        batch_ids=batch_ids,
        batch_vocab=batch_vocab,
        var_names=var_names,
    )


def normalize(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """log1p(CP10k) normalization. counts: (B, G) dense raw UMIs."""
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)  # guard against empty cells
    scaled = counts * (target_sum / row_sums)
    return np.log1p(scaled, dtype=np.float32)


class VCCDataset(Dataset):
    """Per-cell dataset over a VCCSplit.

    __getitem__(i) returns:
        x         : (n_genes,) float32 tensor, normalized expression
        pert_id   : int
        batch_id  : int
        is_control: bool
    """

    def __init__(
        self,
        split: VCCSplit,
        indices: np.ndarray | None = None,
        normalize_on_get: bool = True,
    ):
        self.split = split
        self.indices = (
            np.arange(split.n_cells, dtype=np.int64) if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.normalize_on_get = normalize_on_get

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        cell_idx = int(self.indices[i])
        row = self.split.X[cell_idx].toarray().ravel()
        if self.normalize_on_get:
            row = normalize(row[None, :])[0]
        return (
            torch.from_numpy(row),
            int(self.split.pert_ids[cell_idx]),
            int(self.split.batch_ids[cell_idx]),
            bool(self.split.pert_ids[cell_idx] == 0),
        )


def collate_dense(batch):
    """Default collate that stacks dense tensors and returns int64 metadata."""
    xs, pert_ids, batch_ids, is_control = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.tensor(pert_ids, dtype=torch.long),
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(is_control, dtype=torch.bool),
    )


class StratifiedPerturbationSampler(Sampler[list[int]]):
    """Yield batches that contain a mix of controls and several perturbations.

    Each batch contains:
        - n_control_per_batch cells drawn from controls (with replacement)
        - the rest split evenly across n_perts_per_batch perturbations

    This avoids pathological batches (e.g. all-one-perturbation) and gives
    SIGReg a meaningful sample to test for Gaussianity.

    Indexes are positions in the underlying VCCDataset.indices, not raw cell
    indices, so wrap a VCCDataset and use this sampler with batch_sampler=...
    """

    def __init__(
        self,
        dataset: VCCDataset,
        batch_size: int = 512,
        n_perts_per_batch: int = 8,
        control_fraction: float = 0.25,
        n_batches: int | None = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_perts_per_batch = n_perts_per_batch
        self.n_control = max(1, int(round(batch_size * control_fraction)))
        self.n_per_pert = (batch_size - self.n_control) // n_perts_per_batch
        self.batch_size_actual = self.n_control + self.n_per_pert * n_perts_per_batch
        self.seed = seed

        # Group dataset positions by perturbation id
        pert_ids = self.dataset.split.pert_ids[self.dataset.indices]
        self._pos_by_pert: dict[int, np.ndarray] = {}
        for p in np.unique(pert_ids):
            self._pos_by_pert[int(p)] = np.where(pert_ids == p)[0].astype(np.int64)

        self._control_pos = self._pos_by_pert.get(0, np.empty(0, dtype=np.int64))
        self._noncontrol_perts = sorted(p for p in self._pos_by_pert if p != 0)

        if len(self._control_pos) == 0:
            raise ValueError("no control cells in dataset; SIGReg sampler needs them")
        if len(self._noncontrol_perts) < n_perts_per_batch:
            raise ValueError(
                f"only {len(self._noncontrol_perts)} non-control perts in dataset; "
                f"need at least {n_perts_per_batch}"
            )

        if n_batches is None:
            # Default: one epoch ≈ enough batches to cover the noncontrol cells
            n_noncontrol = sum(
                len(self._pos_by_pert[p]) for p in self._noncontrol_perts
            )
            n_batches = max(1, n_noncontrol // (self.n_per_pert * n_perts_per_batch))
        self.n_batches = n_batches

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_batches):
            ctrl = rng.choice(self._control_pos, size=self.n_control, replace=False)
            chosen_perts = rng.choice(
                self._noncontrol_perts, size=self.n_perts_per_batch, replace=False
            )
            parts = [ctrl]
            for p in chosen_perts:
                pool = self._pos_by_pert[int(p)]
                replace = len(pool) < self.n_per_pert
                parts.append(rng.choice(pool, size=self.n_per_pert, replace=replace))
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.tolist()


def make_internal_val_split(
    split: VCCSplit,
    n_holdout_perts: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hold out N perturbations from a training split for internal eval.

    Returns (train_indices, val_indices, holdout_pert_names).
    Controls always stay in train_indices.
    """
    rng = np.random.default_rng(seed)
    noncontrol_perts = [p for p in range(1, split.n_perts)]
    if n_holdout_perts >= len(noncontrol_perts):
        raise ValueError(
            f"can't hold out {n_holdout_perts} perts from {len(noncontrol_perts)}"
        )
    holdout_pert_ids = rng.choice(noncontrol_perts, size=n_holdout_perts, replace=False)
    holdout_set = set(int(p) for p in holdout_pert_ids)
    val_mask = np.array([p in holdout_set for p in split.pert_ids])
    val_indices = np.where(val_mask)[0].astype(np.int64)
    train_indices = np.where(~val_mask)[0].astype(np.int64)
    holdout_names = [split.pert_vocab[i] for i in sorted(holdout_set)]
    return train_indices, val_indices, holdout_names
