"""A/B smoke for the population-risk gate.

Run Phase 2.1 (homeostatic on controls) for a few epochs with the gate
off, then on. Compare final pred MSE, SIGReg convergence, and gate stats.
This validates the gate doesn't break anything before we commit to a
full ~1h training run.

Run:
    python scripts/gate_ab_smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from lewm.data import VCCDataset, load_split
from lewm.models import JEPAPredictor, MLPEncoder
from lewm.train import TrainConfig, _build_optimizer, log_metrics, select_device
from lewm.losses import sigreg_loss
from lewm.models import gene_set_mask


def run_phase1_short(use_gate: bool, epochs: int, split, device, out_dir: Path,
                     gate_alpha: float = 1.0):
    cfg = TrainConfig(
        batch_size=512,
        embed_dim=128,
        hidden_dim=256,
        jepa_hidden=128,
        sigreg_projections=32,
        phase1_epochs=epochs,
        phase1_lr=1e-3,
        use_population_gate=use_gate,
        gate_alpha=gate_alpha,
        out_dir=str(out_dir),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").write_text("")

    torch.manual_seed(0)  # same init for both runs
    np.random.seed(0)

    encoder = MLPEncoder(split.n_genes, cfg.embed_dim, cfg.hidden_dim).to(device)
    jepa = JEPAPredictor(cfg.embed_dim, cfg.jepa_hidden).to(device)

    params = list(encoder.parameters()) + list(jepa.parameters())
    optim = _build_optimizer([{"params": params, "lr": cfg.phase1_lr}], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    ctrl_idx = np.where(split.control_mask)[0]
    ds = VCCDataset(split, indices=ctrl_idx)
    from lewm.data import collate_dense
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_dense, drop_last=True,
        generator=torch.Generator().manual_seed(0),
    )

    label = f"GATE-ON(a={gate_alpha:g})" if use_gate else "GATE-OFF"
    print(f"\n=== {label} ===")
    print(f"  encoder + jepa: {sum(p.numel() for p in params)/1e6:.2f}M params")

    epoch_times = []
    for epoch in range(1, epochs + 1):
        encoder.train(); jepa.train()
        t0 = time.perf_counter()
        ep_pred = ep_sig = 0.0
        n_b = 0
        for x, _, _, _ in loader:
            x = x.to(device)
            x_ctx, x_tgt = gene_set_mask(x, cfg.context_ratio)
            z_ctx = encoder(x_ctx)
            with torch.no_grad():
                z_tgt = encoder(x_tgt)
            z_pred = jepa(z_ctx)
            pred = torch.nn.functional.mse_loss(z_pred, z_tgt.detach())
            sig = sigreg_loss(z_ctx, n_projections=cfg.sigreg_projections)
            loss = pred + cfg.sigreg_weight * sig
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()
            ep_pred += pred.item(); ep_sig += sig.item(); n_b += 1
        sched.step()
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        avg_pred = ep_pred / n_b
        avg_sig = ep_sig / n_b
        gate = optim.gate_stats() if hasattr(optim, "gate_stats") else {}
        gs = (
            f" | q_mean={gate['mean_q']:.2f} q_kill={gate['frac_killed']:.2f}"
            if gate else ""
        )
        print(f"  ep{epoch}: pred={avg_pred:.4f} sig={avg_sig:.4f}{gs} ({elapsed:.1f}s)")

    return {
        "label": label.strip(),
        "final_pred": avg_pred,
        "final_sigreg": avg_sig,
        "mean_epoch_s": float(np.mean(epoch_times)),
        "gate_stats": gate,
    }


def main():
    device = select_device()
    print(f"device: {device}")
    split = load_split("train")
    print(f"split: {split.n_cells} cells x {split.n_genes} genes")

    # Sweep alpha values: fresh-batch (1.0), and a few softer choices.
    # b/(n-b) = 512/210708 ≈ 0.0024. The paper says either is valid; for
    # finite-dataset training the smaller alpha is the formally-correct one.
    alphas_to_try = [1.0, 0.1, 0.01]

    epochs = 4
    out = Path("results/vcc-ab")
    runs = [run_phase1_short(False, epochs, split, device, out / "off")]
    for a_val in alphas_to_try:
        runs.append(run_phase1_short(True, epochs, split, device,
                                     out / f"on_a{a_val:g}",
                                     gate_alpha=a_val))

    print("\n=== A/B summary ===")
    headers = [r["label"] for r in runs]
    print(f"{'metric':<22} " + " ".join(f"{h:>20}" for h in headers))
    for k in ["final_pred", "final_sigreg", "mean_epoch_s"]:
        vals = [r[k] for r in runs]
        print(f"{k:<22} " + " ".join(f"{v:>20.4f}" for v in vals))
    print()
    for r in runs:
        if r["gate_stats"]:
            g = r["gate_stats"]
            print(
                f"  {r['label']:<22} "
                f"q_mean={g['mean_q']:.3f} frac_open={g['frac_open']:.3f} "
                f"frac_killed={g['frac_killed']:.3f}"
            )


if __name__ == "__main__":
    main()
