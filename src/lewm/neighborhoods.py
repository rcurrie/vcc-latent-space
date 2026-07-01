"""Overlapping-KNN neighborhoods + count-corrected pseudo-bulk (S2).

The aggregation primitive for the neighborhood experiment (docs/des_neighborhood_plan.md).
Two pieces, both pure (no training, no model dependency):

  build_knn_neighborhoods : Milo-style overlapping-KNN neighborhoods on a frozen
                            embedding. Lifts the *aggregation primitive* from
                            milopy.make_nhoods — NOT Milo's NB-GLM differential-
                            abundance test. Neighborhoods overlap (a cell can
                            belong to several), which is the bias/variance knob `k`
                            controls between per-cell (k=1) and global (k=N).

  pseudobulk_lognorm      : count-corrected pseudo-bulk for one neighborhood —
                            SUM raw UMIs across cells -> CP10k -> log1p. This is
                            the biologically correct aggregation; we never take a
                            mean-of-logs (Jensen-biased, depth-blind).

The encoder that produces the embeddings stays frozen and external: callers pass
in an (n_cells, embed_dim) array. S7 will swap MCR² embeddings for PCA here to
test whether the SSL encoder is load-bearing.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def build_knn_neighborhoods(
    embeddings: np.ndarray,
    k: int = 50,
    prop: float = 0.1,
    refine: bool = True,
    seed: int = 0,
) -> list[np.ndarray]:
    """Overlapping-KNN neighborhoods on `embeddings` (n_cells, embed_dim).

    Mirrors milopy.make_nhoods: sample a fraction `prop` of cells as index
    cells; optionally refine each index to the neighborhood member nearest the
    neighborhood mean (Milo's "graph" refinement — picks index cells in dense
    regions and de-duplicates redundant neighborhoods); each neighborhood is
    then that index cell plus its k nearest neighbors.

    Parameters
    ----------
    k       : neighbors per neighborhood (neighborhood size is k+1 incl. self).
    prop    : fraction of cells sampled as index cells.
    refine  : apply Milo's index refinement.
    seed    : RNG seed for index sampling.

    Returns
    -------
    list of int arrays (each length k+1), one per neighborhood. Overlapping.
    """
    n = embeddings.shape[0]
    k_eff = min(k, n - 1)
    nn_model = NearestNeighbors(n_neighbors=k_eff + 1).fit(embeddings)  # +1 = self
    knn_idx = nn_model.kneighbors(embeddings, return_distance=False)    # (n, k+1)

    rng = np.random.default_rng(seed)
    n_index = max(1, int(round(prop * n)))
    index_cells = rng.choice(n, size=n_index, replace=False)

    if refine:
        refined = np.empty(len(index_cells), dtype=np.int64)
        for i, ic in enumerate(index_cells):
            nbrs = knn_idx[ic]
            centroid = embeddings[nbrs].mean(axis=0)
            d = ((embeddings[nbrs] - centroid) ** 2).sum(axis=1)
            refined[i] = nbrs[int(d.argmin())]
        index_cells = np.unique(refined)

    return [knn_idx[ic] for ic in index_cells]


def pseudobulk_lognorm(
    counts_raw: np.ndarray | sp.spmatrix,
    target_sum: float = 1e4,
) -> np.ndarray:
    """Count-corrected pseudo-bulk of one neighborhood.

    SUM raw UMIs across cells -> CP10k -> log1p. Depth-correct and Jensen-free,
    unlike a mean of per-cell log1p(CP10k) vectors.

    counts_raw : (n_cells, n_genes) raw UMI counts (dense or sparse).
    Returns    : (n_genes,) float32 = log1p(target_sum * gene_sum / total_sum).
    """
    if sp.issparse(counts_raw):
        summed = np.asarray(counts_raw.sum(axis=0)).ravel()
    else:
        summed = np.asarray(counts_raw).sum(axis=0)
    total = max(float(summed.sum()), 1.0)
    cp = summed * (target_sum / total)
    return np.log1p(cp).astype(np.float32)


def neighborhood_pseudobulks(
    counts_raw: np.ndarray | sp.spmatrix,
    neighborhoods: list[np.ndarray],
    target_sum: float = 1e4,
) -> np.ndarray:
    """Stack count-corrected pseudo-bulks over a list of neighborhoods.

    Returns (n_neighborhoods, n_genes) float32.
    """
    return np.stack(
        [pseudobulk_lognorm(counts_raw[nb], target_sum) for nb in neighborhoods],
        axis=0,
    )
