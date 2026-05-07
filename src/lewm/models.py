"""Model components for the LeWM scRNA-seq world model.

Architecture (mirrors legacy/lewm_scrna.py with adaptations for VCC):

  Encoder        : gene_dim -> embed_dim, BatchNorm projector, no L2 norm
                   Operates in R^d to match SIGReg's isotropic Gaussian target.
  ActionEmbed    : gene index -> action embedding, via a learned MLP over
                   precomputed per-gene features. Generalizes to unseen perts
                   because the gene panel is shared across train/val/test.
  AdaLNBlock     : transformer block with AdaLN modulation (zero-initialized
                   so action conditioning starts as identity).
  PerturbPred    : stack of AdaLN blocks. (z_cell, action) -> z_post.
  JEPAPredictor  : simple MLP for the homeostatic gene-set masking task.
  Decoder        : embed_dim -> gene_dim, for VCC submission generation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """gene_dim -> embed_dim with BatchNorm projector. No L2 normalization.

    SIGReg targets isotropic Gaussian in R^d, so we leave the embeddings
    unconstrained on the sphere.
    """
    def __init__(self, gene_dim: int, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # 1-layer MLP + BN projector matches the LeWM paper's design.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


class JEPAPredictor(nn.Module):
    """Simple MLP predictor for the JEPA gene-set masking task.

    In Phase 2.1 (homeostatic pretraining), the encoder sees a randomly-
    masked subset of genes and the JEPAPredictor predicts the embedding
    of the complementary masked subset.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ActionEmbed(nn.Module):
    """Map a target gene's column index to an action embedding via gene features.

    gene_features: (n_genes, n_features) tensor, frozen. For each gene we
    precompute statistics from control cells (mean expression, dispersion,
    fraction expressing). At test time, perturbations of unseen genes plug
    into the same MLP and produce a real action embedding.

    Falls back to a zero embedding only when gene_idx == -1 (unknown).
    """
    def __init__(
        self,
        gene_features: torch.Tensor,
        action_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # Clone explicitly so the buffer owns its memory (defensive against
        # torch.from_numpy aliasing on the caller's side).
        self.register_buffer("gene_features", gene_features.detach().clone().float().contiguous())
        n_features = gene_features.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @property
    def action_dim(self) -> int:
        return self.proj[-1].out_features

    def forward(self, gene_idx: torch.Tensor) -> torch.Tensor:
        # gene_idx may contain -1 (unknown / control). Replace with a safe
        # value, look up features, and zero out the unknown rows afterwards.
        safe_idx = gene_idx.clamp(min=0)
        feats = self.gene_features[safe_idx]                           # (B, F)
        emb = self.proj(feats)                                         # (B, A)
        mask = (gene_idx >= 0).float().unsqueeze(-1)
        return emb * mask


class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization.

    Action embedding modulates each layer via learned scale/shift, both
    initialized to zero so the block starts as identity-like ("no
    perturbation effect"). Joint training learns deviations.
    """
    def __init__(self, embed_dim: int, n_heads: int, action_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # AdaLN: action_emb -> (scale1, shift1, scale2, shift2), zero-init.
        self.adaln = nn.Linear(action_dim, embed_dim * 4)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2 = self.adaln(action_emb).chunk(4, dim=-1)

        h = self.norm1(x) * (1 + s1) + b1
        h = h.unsqueeze(1)                          # (B, 1, D) for MHA
        h, _ = self.attn(h, h, h)
        x = x + h.squeeze(1)

        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.mlp(h)
        return x


class PerturbationPredictor(nn.Module):
    """World model predictor: (z_source, gene_idx) -> z_post.

    Stack of AdaLN-conditioned transformer blocks. Action conditioning is
    zero-initialized so the predictor starts as near-identity and gradually
    learns how each perturbation shifts the representation.
    """
    def __init__(
        self,
        embed_dim: int,
        action_embed: ActionEmbed,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.action_embed = action_embed
        action_dim = action_embed.action_dim
        # Project action features into the AdaLN conditioning space.
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, n_heads, embed_dim) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z_source: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        action = self.action_embed(gene_idx)
        action = self.action_proj(action)
        x = z_source
        for block in self.blocks:
            x = block(x, action)
        return self.out_norm(x)


class Decoder(nn.Module):
    """embed_dim -> gene_dim, predicts log1p(CP10k)-normalized expression.

    Outputs are non-negative (softplus or relu) so they round-trip cleanly
    through cell-eval's expectations.
    """
    def __init__(self, embed_dim: int, gene_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z))


def gene_set_mask(
    x: torch.Tensor,
    context_ratio: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition genes into context (visible) and target (masked) views.

    Returns two zero-padded views of x; together they cover all genes.
    The same mask is used for the whole batch (per-batch random partition).
    """
    gene_dim = x.shape[1]
    n_context = int(gene_dim * context_ratio)
    perm = torch.randperm(gene_dim, device=x.device)
    x_context = torch.zeros_like(x)
    x_context[:, perm[:n_context]] = x[:, perm[:n_context]]
    x_target = torch.zeros_like(x)
    x_target[:, perm[n_context:]] = x[:, perm[n_context:]]
    return x_context, x_target


def compute_gene_features(
    X_csr,
    control_indices,
    log_normalize_target_sum: float = 1e4,
) -> torch.Tensor:
    """Precompute per-gene features from control cells.

    Three features per gene:
      0. log1p of mean expression (CP10k-normalized)
      1. log of (variance / mean) — overdispersion proxy
      2. fraction of control cells with nonzero expression

    Parameters
    ----------
    X_csr : scipy.sparse.csr_matrix of raw counts (n_cells, n_genes)
    control_indices : np.ndarray of int positions for control cells

    Returns
    -------
    features : torch.Tensor (n_genes, 3), float32
    """
    import numpy as np
    import scipy.sparse as sp

    Xc = X_csr[control_indices]                      # (n_ctrl, n_genes), CSR
    n_ctrl = Xc.shape[0]

    # CP10k-normalize per cell, then take per-gene mean/var without densifying
    row_sums = np.asarray(Xc.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1.0)
    scale = log_normalize_target_sum / row_sums      # (n_ctrl,)

    # Scale rows of Xc by `scale`. Build a sparse diag * Xc.
    Xs = sp.diags(scale) @ Xc                        # (n_ctrl, n_genes), CSR
    # We want per-gene mean and variance of log1p(Xs). Densifying log1p of
    # nonzeros only is correct because log1p(0) = 0.
    Xs_log = Xs.copy()
    Xs_log.data = np.log1p(Xs_log.data, dtype=np.float32)

    mean = np.asarray(Xs_log.sum(axis=0)).ravel() / n_ctrl
    sq_mean = np.asarray(Xs_log.multiply(Xs_log).sum(axis=0)).ravel() / n_ctrl
    var = np.maximum(sq_mean - mean ** 2, 0.0)

    # Fraction expressing: count nonzero entries per column
    nnz_per_gene = np.asarray((Xc != 0).sum(axis=0)).ravel()
    frac_expr = nnz_per_gene / n_ctrl

    feat0 = np.log1p(mean).astype(np.float32)
    feat1 = np.log(np.maximum(var / np.maximum(mean, 1e-6), 1e-6)).astype(np.float32)
    feat2 = frac_expr.astype(np.float32)

    feats = np.stack([feat0, feat1, feat2], axis=1)  # (n_genes, 3)
    return torch.tensor(feats, dtype=torch.float32)
