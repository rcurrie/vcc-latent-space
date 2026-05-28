"""Option B Hybrid: latent encoder + direct gene-space delta head.

The latent-space pipeline (v2/v3/v4/v5) hit a ceiling at Test PDS ≈ 0.53.
The diagnostic in scripts/v2/diagnostic_gaussianity.py showed the issue
isn't Gaussianity enforcement — it's that single-cell intra-perturbation
noise dominates the inter-perturbation centroid signal by ~67×. The
field's winners avoid this by aggregating to pseudo-bulk and predicting
delta-from-control directly in gene space.

This hybrid keeps the latent-space encoder as a *feature extractor* but
predicts gene-space deltas directly:

    z_control = encoder(x_control)          # frozen
    action    = action_embed(gene_idx)      # frozen or trainable
    delta_x   = delta_head(z_control, action)
    x_pert    = x_control + delta_x

Training: pseudo-bulk. Per batch, compute mean control + mean perturbed
per perturbation, train delta_head on `MSE(predicted_delta, true_delta)`.

Inference: pseudo-bulk. Average training controls, encode, predict delta
per eval perturbation, add to control mean. PDS/DES/MAE on the per-pert
gene-space means.

Why we still call it "hybrid": we reuse the SSL-pretrained encoder
(Phase A's invariance + MCR² recipe) as a feature representation, but
the prediction target lives in gene space. Decoder is gone.
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
from .models import (
    ActionConditionedDecoder,
    MLPEncoder,
    ProteinActionEmbedV2,
)
from .splits import load_internal_val_split, partition_indices_for_internal_val
from .train_phase_a import select_device


DEFAULT_PHASE_A_CKPT = "results/v2/phase_a/checkpoint.pt"
DEFAULT_PCA_PANEL_PATH = "data/vcc/v2_gene_esm2_panel_pca1280.pt"


@dataclass
class HybridConfig:
    # data
    batch_size: int = 512
    n_perts_per_batch: int = 8
    control_fraction: float = 0.25
    # model
    embed_dim: int = 256
    hidden_dim: int = 512
    action_dim: int = 64
    delta_hidden: int = 1024
    delta_blocks: int = 2
    # Capacity ablation. Head choices:
    #   "simple" : Linear(action_dim → gene_dim), action-only, ~1.2M params.
    #              ESM2-protein-similarity baseline. Test PDS=0.538 (10 ep).
    #   "concat" : Linear(embed_dim + action_dim → gene_dim), action + z_ctrl
    #              concatenated as input. ~5.8M params. Tests whether the SSL
    #              encoder's output adds gene-space-delta information beyond
    #              what ESM2 carries.
    #   "adaln"  : full 28.7M-param AdaLN delta head (catastrophic mean
    #              collapse — kept here only for reproducibility).
    head_type: str = "simple"
    use_simple_head: bool = True  # back-compat alias for head_type=="simple"
    # optimization
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # paths
    phase_a_checkpoint: str = DEFAULT_PHASE_A_CKPT
    protein_panel_path: str = DEFAULT_PCA_PANEL_PATH
    out_dir: str = "results/v2/hybrid"
    # eval cadence
    eval_every: int = 5
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


@torch.no_grad()
def score_hybrid_against_split(
    *,
    encoder,
    action_embed,
    delta_head: ActionConditionedDecoder,
    train_split,
    eval_split,
    device: torch.device,
    n_ctrl_for_ref: int = 1024,
    deg_top_k: int = 100,
    seed: int = 0,
):
    from lewm.data import normalize
    # encoder is frozen — keep it in eval throughout (don't restore .train() at the
    # end). action_embed and delta_head are reset to .train() by caller next epoch.
    encoder.eval(); action_embed.eval(); delta_head.eval()
    if list(eval_split.var_names) != list(train_split.var_names):
        raise RuntimeError("eval split var_names != train split var_names")
    var_to_col = {g: i for i, g in enumerate(train_split.var_names)}

    rng = np.random.default_rng(seed)
    # Pseudo-bulk control reference: mean of n_ctrl_for_ref training controls.
    ctrl_idx = np.where(train_split.control_mask)[0]
    pick = rng.choice(ctrl_idx, size=min(n_ctrl_for_ref, len(ctrl_idx)), replace=False)
    x_ctrl_dense = train_split.X[pick].toarray()
    x_ctrl_mean = torch.from_numpy(normalize(x_ctrl_dense).mean(axis=0)).to(device)  # (G,)
    z_ctrl_mean = encoder(x_ctrl_mean.unsqueeze(0))                                  # (1, D)

    eval_perts_present = sorted(set(int(p) for p in eval_split.pert_ids if p != 0))

    actual_means = {}
    pred_means = {}
    pert_names = {}
    for pid in eval_perts_present:
        gene_name = eval_split.pert_vocab[pid]
        col = var_to_col.get(gene_name, -1)
        if col == -1:
            continue
        pert_names[pid] = gene_name

        cell_pos = np.where(eval_split.pert_ids == pid)[0]
        x_actual_dense = eval_split.X[cell_pos].toarray()
        x_actual_mean = torch.from_numpy(normalize(x_actual_dense).mean(axis=0)).to(device)  # (G,)

        gene_idx = torch.tensor([col], dtype=torch.long, device=device)
        action = action_embed(gene_idx)                                              # (1, A)
        delta = delta_head(z_ctrl_mean, action).squeeze(0)                           # (G,)
        x_pred = x_ctrl_mean + delta

        actual_means[pid] = x_actual_mean
        pred_means[pid] = x_pred

    perts = list(actual_means.keys())
    n = len(perts)
    if n < 2:
        encoder.train(); action_embed.train(); delta_head.train()
        return {"n_perts": n, "pds": float("nan"), "des": float("nan"), "mae": float("nan")}

    pred_stack = torch.stack([pred_means[p] for p in perts], dim=0)
    actual_stack = torch.stack([actual_means[p] for p in perts], dim=0)

    # PDS: L1 in gene space.
    dist = (pred_stack.unsqueeze(1) - actual_stack.unsqueeze(0)).abs().sum(dim=-1)
    diag = dist.diag()
    rank_correct = (dist > diag.unsqueeze(1)).sum(dim=1)
    pds = float((rank_correct.float() / max(n - 1, 1)).mean().item())

    # DES: top-k DEG Jaccard vs control mean.
    ctrl_mean_cpu = x_ctrl_mean.cpu().numpy()
    des_vals = []
    mae_vals = []
    for p in perts:
        a_delta = actual_means[p].cpu().numpy() - ctrl_mean_cpu
        p_delta = pred_means[p].cpu().numpy() - ctrl_mean_cpu
        actual_up = set(np.argsort(-a_delta)[:deg_top_k])
        actual_dn = set(np.argsort(a_delta)[:deg_top_k])
        pred_up = set(np.argsort(-p_delta)[:deg_top_k])
        pred_dn = set(np.argsort(p_delta)[:deg_top_k])
        j_up = len(actual_up & pred_up) / max(len(actual_up | pred_up), 1)
        j_dn = len(actual_dn & pred_dn) / max(len(actual_dn | pred_dn), 1)
        des_vals.append(0.5 * (j_up + j_dn))
        mae_vals.append((pred_means[p] - actual_means[p]).abs().mean().item())

    # encoder STAYS in eval (frozen); only restore trainable modules.
    action_embed.train(); delta_head.train()
    return {
        "n_perts": n,
        "pds": pds,
        "des": float(np.mean(des_vals)),
        "mae": float(np.mean(mae_vals)),
    }


def train_hybrid(cfg: HybridConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device()
    print(f"device: {device}")

    split = load_split("train")
    iv = load_internal_val_split()
    holdout_pert_names = iv["holdout_pert_names"]
    train_idx, _ = partition_indices_for_internal_val(split, holdout_pert_names)
    print(f"train cells: {len(train_idx)} (holdout {len(holdout_pert_names)} perts excluded)")

    aug = AugmentConfig(tau=0.5, paired_views=False)
    train_ds = V2Dataset(split, indices=train_idx, aug=aug, seed=cfg.seed)
    sampler = StratifiedPerturbationSampler(
        train_ds, batch_size=cfg.batch_size,
        n_perts_per_batch=cfg.n_perts_per_batch,
        control_fraction=cfg.control_fraction, seed=cfg.seed,
    )
    fork_ctx = mp.get_context("fork") if cfg.num_workers > 0 else None
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_v2,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=fork_ctx,
    )

    # ---- models ----
    encoder = MLPEncoder(
        gene_dim=split.n_genes, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
    ).to(device)
    ckpt = torch.load(cfg.phase_a_checkpoint, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    print(f"loaded Phase A encoder from {cfg.phase_a_checkpoint}")
    _freeze(encoder)

    panel = torch.load(cfg.protein_panel_path, weights_only=False, map_location="cpu")
    action_embed = ProteinActionEmbedV2(
        protein_embeddings=panel["embeddings"], coverage=panel["coverage"],
        action_dim=cfg.action_dim,
    ).to(device)
    # action_embed is randomly initialized — we'll train its projection
    # alongside the delta head so the action representation is shaped for
    # the gene-space delta task.

    # Resolve head choice (head_type takes precedence; use_simple_head is back-compat).
    head_choice = cfg.head_type if hasattr(cfg, "head_type") else None
    if head_choice is None:
        head_choice = "simple" if cfg.use_simple_head else "adaln"

    if head_choice == "simple":
        # Tiny linear delta head. Ignores z_ctrl entirely; pure
        # action → gene-space delta. ~1.2M params if action_dim=64,
        # gene_dim=18080. Forces the model to use ESM2's protein-similarity
        # structure rather than memorizing 135 perts.
        class SimpleDeltaHead(torch.nn.Module):
            def __init__(self, action_dim, gene_dim):
                super().__init__()
                self.lin = torch.nn.Linear(action_dim, gene_dim)
                torch.nn.init.zeros_(self.lin.weight)
                torch.nn.init.zeros_(self.lin.bias)
            def forward(self, z_unused, action):
                return self.lin(action)
        delta_head = SimpleDeltaHead(cfg.action_dim, split.n_genes).to(device)
    elif head_choice == "concat":
        # Linear delta head taking [z_ctrl, action] concatenated. Tests whether
        # the SSL encoder's output carries gene-space-delta information beyond
        # ESM2 alone. ~5.8M params at embed_dim=256, action_dim=64,
        # gene_dim=18080.
        class ConcatDeltaHead(torch.nn.Module):
            def __init__(self, embed_dim, action_dim, gene_dim):
                super().__init__()
                self.lin = torch.nn.Linear(embed_dim + action_dim, gene_dim)
                torch.nn.init.zeros_(self.lin.weight)
                torch.nn.init.zeros_(self.lin.bias)
            def forward(self, z, action):
                # z is (1, embed_dim), action is (1, action_dim).
                h = torch.cat([z, action], dim=-1)
                return self.lin(h)
        delta_head = ConcatDeltaHead(
            cfg.embed_dim, cfg.action_dim, split.n_genes,
        ).to(device)
    else:
        delta_head = ActionConditionedDecoder(
            embed_dim=cfg.embed_dim, action_dim=cfg.action_dim,
            gene_dim=split.n_genes, hidden_dim=cfg.delta_hidden, n_blocks=cfg.delta_blocks,
        ).to(device)
        # AdaLN decoder applies softplus on output, which we DON'T want for a
        # signed delta. Override its forward.
        def delta_forward(z, a):
            h = delta_head.in_proj(z)
            for blk in delta_head.blocks:
                h = blk(h, a)
            return delta_head.out_proj(h)
        delta_head.forward = delta_forward

    n_params = sum(p.numel() for p in delta_head.parameters()) + sum(
        p.numel() for p in action_embed.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M "
          f"(delta_head + action_embed projection)")

    optim = torch.optim.AdamW(
        list(delta_head.parameters()) + list(action_embed.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    pert_to_gene_idx = _build_pert_to_gene_idx(split).to(device)

    # ---- train ----
    total_time = 0.0
    best_pds = -float("inf")
    best_epoch = -1
    final_metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        delta_head.train(); action_embed.train()
        encoder.eval()  # belt-and-braces: encoder is frozen and stays in eval
        t0 = time.perf_counter()
        sums = {"loss": 0.0, "n_perts_total": 0}
        n_batches = 0

        for x1, _x2, pert_id, _batch, is_control in loader:
            x1 = x1.to(device)
            pert_id = pert_id.to(device)
            is_control = is_control.to(device)

            ctrl_mask = is_control
            if ctrl_mask.sum() == 0:
                continue
            x_ctrl_mean = x1[ctrl_mask].mean(dim=0)                    # (G,)

            with torch.no_grad():
                z_ctrl_mean = encoder(x_ctrl_mean.unsqueeze(0))        # (1, D)

            # Per-perturbation pseudo-bulk targets.
            unique_perts = torch.unique(pert_id[~ctrl_mask])
            unique_perts = unique_perts[unique_perts != 0]
            if len(unique_perts) == 0:
                continue

            losses = []
            for pid_v in unique_perts:
                pid_int = int(pid_v.item())
                gi = int(pert_to_gene_idx[pid_int].item())
                if gi < 0:
                    continue
                p_mask = pert_id == pid_int
                if p_mask.sum() == 0:
                    continue
                x_pert_mean = x1[p_mask].mean(dim=0)                    # (G,)
                delta_target = x_pert_mean - x_ctrl_mean                # (G,)

                gene_idx = torch.tensor([gi], dtype=torch.long, device=device)
                action = action_embed(gene_idx)                         # (1, A)
                delta_pred = delta_head(z_ctrl_mean, action).squeeze(0) # (G,)

                losses.append(F.mse_loss(delta_pred, delta_target))

            if not losses:
                continue

            L = torch.stack(losses).mean()

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(
                list(delta_head.parameters()) + list(action_embed.parameters()),
                cfg.grad_clip,
            )
            optim.step()

            sums["loss"] += float(L.detach())
            sums["n_perts_total"] += len(losses)
            n_batches += 1

        sched.step()
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        avg = {"epoch": epoch, "lr": optim.param_groups[0]["lr"],
               "epoch_time_s": epoch_time,
               "loss": sums["loss"] / max(n_batches, 1),
               "perts_seen_per_batch": sums["n_perts_total"] / max(n_batches, 1)}

        # Eval on the 50-pert Validation file periodically.
        if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val_split = load_split("val")
            r = score_hybrid_against_split(
                encoder=encoder, action_embed=action_embed, delta_head=delta_head,
                train_split=split, eval_split=val_split, device=device,
                n_ctrl_for_ref=1024, deg_top_k=cfg.val_deg_top_k, seed=cfg.seed,
            )
            avg["val_pds"] = r["pds"]
            avg["val_des"] = r["des"]
            avg["val_mae"] = r["mae"]
            if r["pds"] > best_pds:
                best_pds = r["pds"]
                best_epoch = epoch

        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(avg, default=float) + "\n")

        line = f"[H {epoch:3d}/{cfg.epochs}] L={avg['loss']:.4f}"
        if "val_pds" in avg:
            line += (f" | val PDS={avg['val_pds']:.4f}"
                     f" DES={avg['val_des']:.4f} MAE={avg['val_mae']:.4f}")
        line += f" ({epoch_time:.1f}s)"
        print(line)
        final_metrics = avg

    # ---- save ----
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "action_embed": action_embed.state_dict(),
            "delta_head": delta_head.state_dict(),
            "config": asdict(cfg),
            "final_metrics": final_metrics,
            "total_time_s": total_time,
        },
        str(ckpt_path),
    )
    print(f"\nsaved checkpoint to {ckpt_path}")
    print(f"best val PDS: {best_pds:.4f} at ep{best_epoch}")
    print(f"total train time: {total_time:.1f}s")

    # ---- final Test score ----
    print(f"\nScoring against TEST file (100 disjoint perts)...")
    test_split = load_split("test")
    test_scores = score_hybrid_against_split(
        encoder=encoder, action_embed=action_embed, delta_head=delta_head,
        train_split=split, eval_split=test_split, device=device,
        n_ctrl_for_ref=1024, deg_top_k=cfg.val_deg_top_k, seed=cfg.seed,
    )
    print(f"  Test PDS:      {test_scores['pds']:.4f}   (A1 Test: 0.528, chance: 0.500)")
    print(f"  Test DES:      {test_scores['des']:.4f}   (A1 Test: 0.084)")
    print(f"  Test MAE:      {test_scores['mae']:.4f}   (A1 Test: 0.017)")

    (out_dir / "test_scores.json").write_text(
        json.dumps({"split": "test", **test_scores}, indent=2, default=float)
    )

    return {**final_metrics, "test_pds": test_scores["pds"],
            "test_des": test_scores["des"], "test_mae": test_scores["mae"]}
