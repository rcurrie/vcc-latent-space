"""Phase 2 training loops for the VCC LeWM world model.

Two phases share infrastructure but have different objectives:

  Phase 2.1 (homeostatic) :
      Train encoder + JEPAPredictor on control cells only.
      JEPA gene-set masking + SIGReg. Establishes a non-degenerate
      latent space before perturbations are introduced.

  Phase 2.2 (perturbation):
      Train encoder + PerturbationPredictor + Decoder jointly on the
      full stratified dataset. Encoder unfrozen; SIGReg keeps it from
      collapsing. Perturbations are conditioning actions; the decoder
      maps z back to gene space for VCC submissions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    VCCDataset,
    StratifiedPerturbationSampler,
    collate_dense,
    load_split,
    make_internal_val_split,
    normalize,
    CONTROL_LABEL,
)
from .losses import sigreg_loss, contrastive_centroid_loss
from .models import (
    ActionEmbed,
    Decoder,
    JEPAPredictor,
    MLPEncoder,
    PerturbationPredictor,
    compute_gene_features,
    gene_set_mask,
)
from .optimizers import PopRiskAdamW


@dataclass
class TrainConfig:
    # data
    batch_size: int = 512
    n_perts_per_batch: int = 8
    control_fraction: float = 0.25
    n_holdout_perts: int = 10
    # model
    embed_dim: int = 256
    hidden_dim: int = 512
    jepa_hidden: int = 256
    action_dim: int = 64
    adaln_layers: int = 4
    adaln_heads: int = 4
    decoder_hidden: int = 1024
    # objective
    sigreg_weight: float = 1.0
    sigreg_projections: int = 64
    decoder_weight: float = 1.0
    context_ratio: float = 0.75
    # Phase 3 #1: contrastive auxiliary loss on perturbation centroids.
    # Pushes z_pred for pert g toward the actual centroid of g and away
    # from other perts' centroids. Targets mean collapse directly.
    contrastive_weight: float = 0.0   # 0.0 = disabled (default for backward compat)
    contrastive_temperature: float = 1.0
    # optimization
    phase1_epochs: int = 30
    phase1_lr: float = 1e-3
    phase2_epochs: int = 40
    phase2_predictor_lr: float = 3e-4
    phase2_encoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # logging
    log_every: int = 50
    eval_every: int = 2  # epochs
    out_dir: str = "results/vcc"
    seed: int = 0
    # population-risk gate (Litman & Guo 2026)
    use_population_gate: bool = False
    gate_alpha: float = 1.0       # 1.0 = fresh-batch boundary; b/(n-b) = formal
    gate_lambda_pop: float = 0.0  # soft-gate denom scale; 0 = mostly-binary
    gate_rho: float = 0.99        # variance EMA decay


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_metrics(out_dir: Path, **kwargs) -> None:
    with open(out_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(kwargs, default=float) + "\n")


def _build_optimizer(param_groups, cfg: TrainConfig):
    """Build either AdamW (default) or PopRiskAdamW depending on cfg flag.

    param_groups is the same structure expected by torch.optim — either a
    list of params or a list of {'params': ..., 'lr': ...} dicts.
    """
    if cfg.use_population_gate:
        return PopRiskAdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
            rho=cfg.gate_rho,
            alpha=cfg.gate_alpha,
            lambda_pop=cfg.gate_lambda_pop,
        )
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train_phase1(
    encoder: MLPEncoder,
    jepa_predictor: JEPAPredictor,
    control_dataset: VCCDataset,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Phase 2.1: homeostatic pretraining on controls.

    Trains encoder + JEPA predictor with gene-set masking + SIGReg.
    """
    encoder.train(); jepa_predictor.train()
    params = list(encoder.parameters()) + list(jepa_predictor.parameters())
    optim = _build_optimizer(
        [{"params": params, "lr": cfg.phase1_lr}], cfg,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.phase1_epochs)

    # Plain DataLoader with shuffling for Phase 1 (no perturbation balancing
    # needed since everything is control here).
    loader = DataLoader(
        control_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_dense,
        drop_last=True,
    )

    print(f"\nPhase 2.1: homeostatic pretraining on {len(control_dataset)} control cells")
    print(f"  encoder + jepa params: {sum(p.numel() for p in params)/1e6:.2f}M")

    total_time = 0.0
    for epoch in range(1, cfg.phase1_epochs + 1):
        t0 = time.perf_counter()
        ep_pred = ep_sig = 0.0
        n_batches = 0
        for x, _pert, _batch, _ctrl in loader:
            x = x.to(device)
            x_ctx, x_tgt = gene_set_mask(x, cfg.context_ratio)
            z_ctx = encoder(x_ctx)
            with torch.no_grad():
                z_tgt = encoder(x_tgt)
            z_pred = jepa_predictor(z_ctx)

            pred_loss = F.mse_loss(z_pred, z_tgt.detach())
            sig_loss = sigreg_loss(z_ctx, n_projections=cfg.sigreg_projections)
            loss = pred_loss + cfg.sigreg_weight * sig_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            ep_pred += pred_loss.item()
            ep_sig += sig_loss.item()
            n_batches += 1
        sched.step()
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        avg_pred = ep_pred / n_batches
        avg_sig = ep_sig / n_batches
        lr_now = sched.get_last_lr()[0]
        gate_stats = optim.gate_stats() if isinstance(optim, PopRiskAdamW) else {}
        gate_str = (
            f" | q_mean={gate_stats['mean_q']:.2f} q_kill={gate_stats['frac_killed']:.2f}"
            if gate_stats else ""
        )
        print(
            f"  ep {epoch:3d}/{cfg.phase1_epochs} | pred={avg_pred:.4f} | "
            f"sigreg={avg_sig:.5f} | lr={lr_now:.2e}{gate_str} | {elapsed:.1f}s"
        )
        log_metrics(
            out_dir, phase="2.1", epoch=epoch, pred=avg_pred, sigreg=avg_sig,
            lr=lr_now, time_s=elapsed, **gate_stats,
        )

    print(f"Phase 2.1 done in {total_time:.1f}s ({total_time/cfg.phase1_epochs:.1f}s/epoch)")
    return {"phase1_time_s": total_time}


def train_phase2(
    encoder: MLPEncoder,
    pert_predictor: PerturbationPredictor,
    decoder: Decoder,
    train_dataset: VCCDataset,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    eval_fn=None,  # called as eval_fn(epoch)
) -> dict:
    """Phase 2.2: joint encoder + predictor + decoder training on all cells.

    Each batch from the stratified sampler contains a mix of control cells
    (pert_id == 0) and perturbed cells. We pair each perturbed cell with a
    random control cell from the *same batch* as its source, train the
    predictor to map (z_control, gene_idx) -> z_perturbed, and decode back
    to gene space. Control cells are paired with themselves (action = -1,
    AdaLN passthrough), so they train the encoder + decoder as an autoencoder
    while contributing to SIGReg.

    Loss components:
        pred : MSE between predicted z_post and z(actual perturbed cell)
        dec  : MSE between predicted gene expression and actual cell
        sig  : SIGReg on the encoder outputs to prevent collapse
    """
    encoder.train(); pert_predictor.train(); decoder.train()
    enc_params = list(encoder.parameters())
    pred_params = list(pert_predictor.parameters()) + list(decoder.parameters())
    optim = _build_optimizer(
        [
            {"params": enc_params, "lr": cfg.phase2_encoder_lr},
            {"params": pred_params, "lr": cfg.phase2_predictor_lr},
        ],
        cfg,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.phase2_epochs)

    sampler = StratifiedPerturbationSampler(
        train_dataset,
        batch_size=cfg.batch_size,
        n_perts_per_batch=cfg.n_perts_per_batch,
        control_fraction=cfg.control_fraction,
        seed=cfg.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_dense,
    )

    n_params = (
        sum(p.numel() for p in enc_params)
        + sum(p.numel() for p in pred_params)
    )
    print(f"\nPhase 2.2: joint training on {len(train_dataset)} cells")
    print(f"  total trainable params: {n_params/1e6:.2f}M")
    print(f"  batches/epoch: {len(sampler)}, batch_size: {sampler.batch_size_actual}")

    # We need to map dataset perturbation IDs (vocab-indexed) to gene column
    # indices in the var_names panel for ActionEmbed.
    pert_to_gene_idx = _build_pert_to_gene_idx_map(train_dataset.split)
    pert_to_gene_idx_t = torch.tensor(pert_to_gene_idx, dtype=torch.long, device=device)

    total_time = 0.0
    use_contrastive = cfg.contrastive_weight > 0.0
    for epoch in range(1, cfg.phase2_epochs + 1):
        t0 = time.perf_counter()
        ep_pred = ep_dec = ep_sig = ep_con = 0.0
        ep_logit_gap = 0.0
        n_batches = 0
        for x, pert_id, _batch, is_control in loader:
            x = x.to(device)
            pert_id = pert_id.to(device)
            is_control = is_control.to(device)
            gene_idx = pert_to_gene_idx_t[pert_id]              # (B,)

            # Pair each cell with a control source. Controls source themselves;
            # perturbed cells source a random control from this batch.
            ctrl_pos = torch.where(is_control)[0]
            if len(ctrl_pos) == 0:
                continue                                          # skip pathological batches
            B = x.shape[0]
            source_idx = torch.where(
                is_control,
                torch.arange(B, device=device),
                ctrl_pos[torch.randint(len(ctrl_pos), (B,), device=device)],
            )
            x_source = x[source_idx]

            z_source = encoder(x_source)
            z_target = encoder(x)
            z_post = pert_predictor(z_source, gene_idx)
            x_hat = decoder(z_post)

            pred_loss = F.mse_loss(z_post, z_target.detach())
            dec_loss = F.mse_loss(x_hat, x)
            sig_loss = sigreg_loss(z_target, n_projections=cfg.sigreg_projections)
            loss = pred_loss + cfg.decoder_weight * dec_loss + cfg.sigreg_weight * sig_loss

            if use_contrastive:
                con_loss, con_diag = contrastive_centroid_loss(
                    z_post, z_target, pert_id, is_control,
                    temperature=cfg.contrastive_temperature,
                )
                loss = loss + cfg.contrastive_weight * con_loss
                ep_con += con_loss.item()
                ep_logit_gap += con_diag.get("mean_logit_gap", 0.0)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + pred_params, cfg.grad_clip)
            optim.step()

            ep_pred += pred_loss.item()
            ep_dec += dec_loss.item()
            ep_sig += sig_loss.item()
            n_batches += 1
        sched.step()
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        avg_pred = ep_pred / n_batches
        avg_dec = ep_dec / n_batches
        avg_sig = ep_sig / n_batches
        avg_con = ep_con / n_batches if use_contrastive else 0.0
        avg_gap = ep_logit_gap / n_batches if use_contrastive else 0.0
        lr_pred = sched.get_last_lr()[1]
        gate_stats = optim.gate_stats() if isinstance(optim, PopRiskAdamW) else {}
        gate_str = (
            f" | q_mean={gate_stats['mean_q']:.2f} q_kill={gate_stats['frac_killed']:.2f}"
            if gate_stats else ""
        )
        con_str = (
            f" | con={avg_con:.3f} gap={avg_gap:+.2f}"
            if use_contrastive else ""
        )
        print(
            f"  ep {epoch:3d}/{cfg.phase2_epochs} | "
            f"pred={avg_pred:.4f} | dec={avg_dec:.4f} | sigreg={avg_sig:.5f}"
            f"{con_str} | lr_pred={lr_pred:.2e}{gate_str} | {elapsed:.1f}s"
        )
        extras = {}
        if use_contrastive:
            extras["con"] = avg_con
            extras["logit_gap"] = avg_gap
        log_metrics(
            out_dir, phase="2.2", epoch=epoch, pred=avg_pred, dec=avg_dec,
            sigreg=avg_sig, lr_pred=lr_pred, time_s=elapsed, **gate_stats, **extras,
        )
        if eval_fn is not None and epoch % cfg.eval_every == 0:
            eval_fn(epoch)

    print(f"Phase 2.2 done in {total_time:.1f}s ({total_time/cfg.phase2_epochs:.1f}s/epoch)")
    return {"phase2_time_s": total_time}


def _build_pert_to_gene_idx_map(split) -> np.ndarray:
    """Map perturbation IDs (vocab indices) to var_names column indices.

    Result[pert_id] = column index of that gene in X, or -1 if pert is
    'non-targeting' or the gene is not in the panel.
    """
    var_to_col = {g: i for i, g in enumerate(split.var_names)}
    out = np.full(len(split.pert_vocab), -1, dtype=np.int64)
    for pid, gname in enumerate(split.pert_vocab):
        if gname == CONTROL_LABEL:
            out[pid] = -1
        else:
            out[pid] = var_to_col.get(gname, -1)
    return out


def build_models(split, cfg: TrainConfig, device: torch.device):
    """Construct encoder, JEPA pred, ActionEmbed, PerturbationPred, Decoder.

    Gene features for ActionEmbed are precomputed from this split's controls.
    """
    encoder = MLPEncoder(split.n_genes, cfg.embed_dim, cfg.hidden_dim).to(device)
    jepa = JEPAPredictor(cfg.embed_dim, cfg.jepa_hidden).to(device)
    print("computing per-gene features from controls ...")
    ctrl_idx = np.where(split.control_mask)[0]
    gene_feats = compute_gene_features(split.X, ctrl_idx).to(device)
    print(f"  gene_features: {tuple(gene_feats.shape)}")
    action_embed = ActionEmbed(
        gene_feats, action_dim=cfg.action_dim, hidden_dim=cfg.action_dim,
    ).to(device)
    pert_predictor = PerturbationPredictor(
        cfg.embed_dim, action_embed, n_layers=cfg.adaln_layers, n_heads=cfg.adaln_heads,
    ).to(device)
    decoder = Decoder(cfg.embed_dim, split.n_genes, cfg.decoder_hidden).to(device)
    return encoder, jepa, pert_predictor, decoder


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").write_text("")  # reset
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device()
    print(f"device: {device}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("\nloading training split ...")
    split = load_split("train")
    print(f"  {split.n_cells} cells x {split.n_genes} genes, {split.n_perts} perts")

    encoder, jepa, pert_predictor, decoder = build_models(split, cfg, device)

    # ---- Phase 2.1: homeostatic pretraining on controls only --------------
    ctrl_idx = np.where(split.control_mask)[0]
    ctrl_dataset = VCCDataset(split, indices=ctrl_idx)
    p1_stats = train_phase1(encoder, jepa, ctrl_dataset, cfg, device, out_dir)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "jepa": jepa.state_dict(),
            "config": asdict(cfg),
        },
        out_dir / "phase1_checkpoint.pt",
    )
    print(f"saved phase1 checkpoint to {out_dir / 'phase1_checkpoint.pt'}")

    # ---- Phase 2.2: joint training on all train cells (minus internal val) ---
    train_idx, val_idx, holdout_names = make_internal_val_split(
        split, n_holdout_perts=cfg.n_holdout_perts, seed=cfg.seed,
    )
    print(f"\ninternal val: {len(holdout_names)} held-out perts ({holdout_names})")
    train_dataset = VCCDataset(split, indices=train_idx)
    val_dataset = VCCDataset(split, indices=val_idx)

    from .eval import internal_val_metrics  # late import to avoid cycle

    def eval_fn(epoch: int) -> None:
        metrics = internal_val_metrics(
            encoder, pert_predictor, decoder, val_dataset, split, device, cfg,
        )
        print(f"    [internal val @ ep {epoch}] {metrics}")
        log_metrics(out_dir, phase="2.2", event="val", epoch=epoch, **metrics)

    p2_stats = train_phase2(
        encoder, pert_predictor, decoder, train_dataset, cfg, device, out_dir,
        eval_fn=eval_fn,
    )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "pert_predictor": pert_predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "config": asdict(cfg),
            "pert_vocab": split.pert_vocab,
            "var_names": split.var_names,
        },
        out_dir / "phase2_checkpoint.pt",
    )
    print(f"saved phase2 checkpoint to {out_dir / 'phase2_checkpoint.pt'}")
    print(
        f"\nphase1: {p1_stats['phase1_time_s']:.1f}s, "
        f"phase2: {p2_stats['phase2_time_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
