"""Model components for LeWM-scRNA v2.

Reuses v1 building blocks where the design is unchanged (MLPEncoder,
JEPAPredictor, AdaLNBlock, PerturbationPredictor, gene_set_mask) and
introduces three v2-specific pieces:

  ProteinActionEmbedV2 — single-linear projection (P -> action_dim) instead
                         of the v1 MLP. Defaults to action_dim=64 on a
                         PCA-reduced 1280-dim ESM2 panel. Per-gene fallback
                         for non-protein-coding panel members preserved.

  ActionConditionedDecoder — AdaLN-zero conditioned decoder, trained in
                             Phase C with the rest of the model frozen.
                             Mirrors the predictor's conditioning so the
                             decoder can produce pert-specific gene-space
                             outputs (the v1 DES gap).

The Phase C decoder lives here even though it does NOT participate in
representation learning, to keep the model namespace tidy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# v1 modules we reuse verbatim.
from lewm.models import (  # noqa: F401
    AdaLNBlock,
    JEPAPredictor,
    MLPEncoder,
    PerturbationPredictor,
    gene_set_mask,
)


class ResidualPerturbationPredictor(nn.Module):
    """Predictor that explicitly returns `z_source + delta`.

    Wraps the v1 PerturbationPredictor (AdaLN-conditioned transformer) but
    adds a zero-initialized output projection so the predictor truly starts
    at identity. The trainable part has to learn only the *delta from
    baseline* — the perturbation effect — rather than the absolute target
    location.

    Why this matters:
      - v1 PerturbationPredictor called itself "AdaLN-zero" but only
        zero-init'd the AdaLN scale/shift parameters, not an output gate.
        At init, the attention residual inside each block contributes a
        non-zero shift; the predictor was *near-identity* but not exactly
        identity, and the predicted z_post lived on the same scale as
        z_target (~5–10 in latent space).
      - With explicit `z_source + delta` and a zero-init out_proj, the
        delta starts as zero. The loss at init is `||z_source − z_target||²`
        — the natural "predict no change" baseline. Each gradient step
        moves the delta toward `z_target − z_source`, a much
        smaller-magnitude target.
      - Standard ResNet-style argument: residual reparameterizations have
        better optimization dynamics when the identity is a reasonable
        starting solution.

    Note: the wrapper has the same forward signature as the v1
    PerturbationPredictor, so it's a drop-in replacement in Phase B/C code.
    """

    def __init__(
        self,
        embed_dim: int,
        action_embed,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        # action_embed is owned by self.core only — don't double-register
        # at the wrapper level or state_dict gets duplicate keys.
        self.core = PerturbationPredictor(
            embed_dim=embed_dim,
            action_embed=action_embed,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        # Zero-init residual head. At init, out_proj(x) == 0 for all x, so
        # forward returns z_source unchanged.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @property
    def action_embed(self):
        # Pass-through accessor for callers that expect predictor.action_embed
        # (e.g. Phase C decoder lookup).
        return self.core.action_embed

    def forward(
        self,
        z_source: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        core_out = self.core(z_source, gene_idx)
        delta = self.out_proj(core_out)
        # Detach z_source on the additive identity path so the encoder doesn't
        # see "make z_source == z_target" pressure via the residual skip — that
        # would collapse the control and perturbed distributions into the same
        # cluster (observed empirically: PR=158 → PR=3 in epoch 1 without this
        # detach). The encoder still gets shaped by L_pred via core_out's
        # dependence on z_source (and by L_inv + L_mcr2 directly), but not
        # via the magnitude-dominant identity skip.
        return z_source.detach() + delta


class ProteinActionEmbedV2(nn.Module):
    """Single-linear projection of a (PCA-reduced) ESM2 panel to action_dim.

    Removes v1's 2-layer MLP. The motivation: with the panel already PCA-
    reduced from 5120 -> 1280 (or directly using a smaller ESM2 model),
    the dimensionality-reduction work is done up front; only a single
    learned linear remap remains. This reduces memorization capacity
    (~82k params at 1280 -> 64) and tests whether the protein-similarity
    geometry survives the simpler projection.

    Falls back to a per-gene learned embedding for panel rows where
    coverage[gene_idx] is False (~1.7% of genes — non-protein-coding).

    Buffers:
      protein_embeddings : (n_genes, P) frozen.
      coverage           : (n_genes,) bool frozen.
    """
    def __init__(
        self,
        protein_embeddings: torch.Tensor,
        coverage: torch.Tensor,
        action_dim: int = 64,
        layernorm_inputs: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "protein_embeddings",
            protein_embeddings.detach().clone().float().contiguous(),
        )
        self.register_buffer(
            "coverage",
            coverage.detach().clone().bool().contiguous(),
        )
        n_genes, P = protein_embeddings.shape

        # Optional LayerNorm over the ESM2 dimension. UCE-brain applies a
        # learned LayerNorm to ESM2 rows before use; we use an unlearned
        # zero-mean/unit-var LN as a defensive default since the PCA output
        # has variance concentrated in early components. Cheap.
        self.input_ln = nn.LayerNorm(P, elementwise_affine=False) if layernorm_inputs else nn.Identity()

        # Single linear: P -> action_dim
        self.proj = nn.Linear(P, action_dim)

        # Per-gene fallback for uncovered panel rows.
        self.fallback = nn.Embedding(n_genes, action_dim)
        nn.init.zeros_(self.fallback.weight)

        self._action_dim = action_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(self, gene_idx: torch.Tensor) -> torch.Tensor:
        safe_idx = gene_idx.clamp(min=0)
        prot = self.protein_embeddings[safe_idx]               # (B, P)
        prot = self.input_ln(prot)
        emb_main = self.proj(prot)                              # (B, A)
        emb_fall = self.fallback(safe_idx)                      # (B, A)

        covered = self.coverage[safe_idx].float().unsqueeze(-1)
        emb = covered * emb_main + (1.0 - covered) * emb_fall

        valid = (gene_idx >= 0).float().unsqueeze(-1)
        return emb * valid


class ActionConditionedDecoder(nn.Module):
    """Phase C decoder: (z, action_emb) -> log1p(CP10k) gene-space output.

    AdaLN-zero modulation by the action so the decoder can specialize per
    perturbation. Trained ONLY after the world model is frozen — never
    influences representation learning. Softplus output keeps predictions
    non-negative.

    Architecturally a 2-block AdaLN MLP at embed_dim, followed by a linear
    projection to gene_dim. This is intentionally similar to the predictor
    so we can compare A5 ablations (concat-conditioning, no-conditioning)
    on the same scaffold.
    """
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        gene_dim: int,
        hidden_dim: int = 1024,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _AdaLNDecBlock(hidden_dim, action_dim) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, gene_dim)

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        for blk in self.blocks:
            h = blk(h, action_emb)
        return F.softplus(self.out_proj(h))


class _AdaLNDecBlock(nn.Module):
    """AdaLN-zero MLP block, residual. Used inside ActionConditionedDecoder."""
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # AdaLN: action -> (scale, shift), zero-initialized for identity start.
        self.adaln = nn.Linear(action_dim, hidden_dim * 2)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        s, b = self.adaln(action_emb).chunk(2, dim=-1)
        h = self.norm(x) * (1 + s) + b
        return x + self.mlp(h)
