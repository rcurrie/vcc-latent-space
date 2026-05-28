"""Phase C: post-hoc decoder training on a frozen Phase B world model.

The encoder, action_embed, and predictor are frozen. We train an
ActionConditionedDecoder to map (z, action) → x where x is the log1p(CP10k)
gene-space expression vector. The decoder cannot influence representation
learning — it's purely a post-hoc decoder for evaluation/interpretation
(the v2 sketch's Phase C).

Training data:
  - controls + 135 training perts (held-out 15 perts stay out per the
    v2_internal_val_split.json)
  - For each batch: pair each perturbed cell with a random control as
    predictor source.
  - Compute BOTH z_actual = encoder(x) AND z_post = predictor(z_source, action),
    then train decoder on both → x. This makes the decoder robust to both
    "the encoder gives me the right latent" and "the predictor gives me the
    right latent" — only the second is used at eval time.

Output:
  results/v2/phase_c/checkpoint.pt — decoder state + ref to Phase B checkpoint
  results/v2/phase_c/metrics.jsonl — per-epoch decoder MSE
  results/v2/phase_c/val_scores.json — final VCC PDS/DES/MAE
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    AugmentConfig,
    StratifiedPerturbationSampler,
    V2Dataset,
    collate_v2,
    load_split,
)
from .eval_gene import score_v2_against_split
from .models import (
    ActionConditionedDecoder,
    MLPEncoder,
    PerturbationPredictor,
    ProteinActionEmbedV2,
)
from .splits import load_internal_val_split, partition_indices_for_internal_val
from .train_phase_a import select_device


DEFAULT_PHASE_B_CKPT = "results/v2/phase_b/checkpoint.pt"
DEFAULT_PCA_PANEL_PATH = "data/vcc/v2_gene_esm2_panel_pca1280.pt"


@dataclass
class PhaseCConfig:
    # data
    batch_size: int = 512
    n_perts_per_batch: int = 8
    control_fraction: float = 0.25
    # model
    embed_dim: int = 256
    action_dim: int = 64
    decoder_hidden: int = 1024
    decoder_blocks: int = 2
    # losses
    z_actual_weight: float = 1.0
    z_post_weight: float = 1.0
    # optimization
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # paths
    phase_b_checkpoint: str = DEFAULT_PHASE_B_CKPT
    protein_panel_path: str = DEFAULT_PCA_PANEL_PATH
    out_dir: str = "results/v2/phase_c"
    # eval
    score_val_every: int = 0       # 0 = once at end of training
    val_n_pred_per_pert: int = 256
    val_deg_top_k: int = 100
    # data loading
    num_workers: int = 4
    seed: int = 0


def _build_pert_to_gene_idx(split) -> torch.Tensor:
    var_to_col = {g: i for i, g in enumerate(split.var_names)}
    out = np.full(len(split.pert_vocab), -1, dtype=np.int64)
    for pid, gname in enumerate(split.pert_vocab):
        if pid == 0:
            continue
        out[pid] = var_to_col.get(gname, -1)
    return torch.from_numpy(out)


def _freeze(*modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()


def train_phase_c(cfg: PhaseCConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device()
    print(f"device: {device}")

    # ---------------------------------------------------------------- data
    split = load_split("train")
    iv = load_internal_val_split()
    holdout_pert_names = iv["holdout_pert_names"]
    train_idx, _ = partition_indices_for_internal_val(split, holdout_pert_names)
    print(f"train cells: {len(train_idx)} (holdout {len(holdout_pert_names)} perts excluded)")

    # tau=1.0 — no augmentation. Phase C is regression, not SSL.
    aug = AugmentConfig(tau=1.0, paired_views=False)
    train_ds = V2Dataset(split, indices=train_idx, aug=aug, seed=cfg.seed)
    sampler = StratifiedPerturbationSampler(
        train_ds,
        batch_size=cfg.batch_size,
        n_perts_per_batch=cfg.n_perts_per_batch,
        control_fraction=cfg.control_fraction,
        seed=cfg.seed,
    )
    fork_ctx = mp.get_context("fork") if cfg.num_workers > 0 else None
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_v2,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=fork_ctx,
    )
    print(f"batches/epoch: {len(sampler)} | batch_size_actual={sampler.batch_size_actual}")

    # ---------------------------------------------------------------- models
    encoder = MLPEncoder(gene_dim=split.n_genes, embed_dim=cfg.embed_dim).to(device)
    panel = torch.load(cfg.protein_panel_path, weights_only=False, map_location="cpu")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"],
        coverage=panel["coverage"],
        action_dim=cfg.action_dim,
    ).to(device)
    predictor = PerturbationPredictor(
        embed_dim=cfg.embed_dim,
        action_embed=action_embed,
        n_layers=4, n_heads=4,
    ).to(device)

    ckpt = torch.load(cfg.phase_b_checkpoint, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    action_embed.load_state_dict(ckpt["action_embed"])
    predictor.load_state_dict(ckpt["predictor"])
    print(f"loaded Phase B checkpoint from {cfg.phase_b_checkpoint}")
    print(f"  Phase B final PDS={ckpt.get('final_metrics', {}).get('latent_pds', ckpt.get('pds', 'n/a'))}")

    # Freeze everything from Phase B. Decoder is the only trainable component.
    _freeze(encoder, action_embed, predictor)

    decoder = ActionConditionedDecoder(
        embed_dim=cfg.embed_dim,
        action_dim=cfg.action_dim,
        gene_dim=split.n_genes,
        hidden_dim=cfg.decoder_hidden,
        n_blocks=cfg.decoder_blocks,
    ).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"decoder params: {n_dec/1e6:.2f}M (only trainable component)")

    optim = torch.optim.AdamW(
        decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    pert_to_gene_idx = _build_pert_to_gene_idx(split).to(device)

    # ---------------------------------------------------------------- train
    total_time = 0.0
    final_metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        decoder.train()
        t0 = time.perf_counter()
        sums = {"dec_actual": 0.0, "dec_post": 0.0, "total": 0.0}
        n_batches = 0

        for x1, _x2, pert_id, _batch, is_control in loader:
            x1 = x1.to(device)
            pert_id = pert_id.to(device)
            is_control = is_control.to(device)
            gene_idx = pert_to_gene_idx[pert_id]

            B = x1.shape[0]
            ctrl_pos = torch.where(is_control)[0]
            if len(ctrl_pos) == 0:
                continue
            source_idx = torch.where(
                is_control,
                torch.arange(B, device=device),
                ctrl_pos[torch.randint(len(ctrl_pos), (B,), device=device)],
            )

            # Frozen forward pass — no grad through encoder/predictor.
            with torch.no_grad():
                z_actual = encoder(x1)
                z_source = z_actual[source_idx]
                z_post = predictor(z_source, gene_idx)
                action_vec = action_embed(gene_idx)

            # Decoder learns both targets.
            x_hat_actual = decoder(z_actual, action_vec)
            x_hat_post = decoder(z_post, action_vec)
            L_actual = F.mse_loss(x_hat_actual, x1)
            L_post = F.mse_loss(x_hat_post, x1)
            L = cfg.z_actual_weight * L_actual + cfg.z_post_weight * L_post

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
            optim.step()

            sums["dec_actual"] += float(L_actual.detach())
            sums["dec_post"] += float(L_post.detach())
            sums["total"] += float(L.detach())
            n_batches += 1

        sched.step()
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        avg = {k: v / n_batches for k, v in sums.items()}
        avg["epoch"] = epoch
        avg["lr"] = optim.param_groups[0]["lr"]
        avg["epoch_time_s"] = epoch_time

        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(avg, default=float) + "\n")

        print(
            f"[C {epoch:3d}/{cfg.epochs}] "
            f"L={avg['total']:.4f} dec_actual={avg['dec_actual']:.4f} "
            f"dec_post={avg['dec_post']:.4f} ({epoch_time:.1f}s)"
        )
        final_metrics = avg

    # ---------------------------------------------------------------- save
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "phase_b_checkpoint_ref": cfg.phase_b_checkpoint,
            "config": asdict(cfg),
            "final_metrics": final_metrics,
            "total_time_s": total_time,
        },
        str(ckpt_path),
    )
    print(f"saved decoder checkpoint to {ckpt_path} "
          f"({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    print(f"total train time: {total_time:.1f}s")

    # ---------------------------------------------------------------- score val file
    print(f"\nScoring against official validation file ...")
    # Build val's own pert_vocab — val perts are different genes from train
    # by design; using train's vocab would collapse them all to pid=-1.
    # The gene PANEL (var_names) is shared, which is what var_to_col uses.
    val_split = load_split("val")
    print(f"  val cells: {val_split.n_cells} | val perts (unique): {val_split.n_perts - 1}")
    scores = score_v2_against_split(
        encoder=encoder, predictor=predictor, action_embed=action_embed,
        decoder=decoder,
        train_split=split, eval_split=val_split,
        device=device,
        n_pred_per_pert=cfg.val_n_pred_per_pert,
        deg_top_k=cfg.val_deg_top_k,
        seed=cfg.seed,
    )
    print(f"\nVCC validation scores:")
    print(f"  n_perts:  {scores.n_perts}")
    print(f"  PDS:      {scores.pds:.4f}   (v1 best: 0.544 / chance: 0.500)")
    print(f"  DES:      {scores.des:.4f}   (v1 best: 0.076)")
    print(f"  MAE:      {scores.mae_logcp10k:.4f}   (v1: 0.014-0.015)")
    print(f"  pred_emb_mse: {scores.pred_emb_mse:.4f}")

    val_path = out_dir / "val_scores.json"
    val_path.write_text(json.dumps(
        {
            "n_perts": scores.n_perts,
            "pds": scores.pds,
            "des": scores.des,
            "mae_logcp10k": scores.mae_logcp10k,
            "pred_emb_mse": scores.pred_emb_mse,
            "per_pert_mae": scores.per_pert_mae,
            "per_pert_des": scores.per_pert_des,
        },
        indent=2, default=float,
    ))
    print(f"wrote {val_path}")

    return {**final_metrics, "val_pds": scores.pds, "val_des": scores.des,
            "val_mae": scores.mae_logcp10k}
