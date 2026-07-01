"""S3 gate: build matched per-neighborhood delta targets and sanity-check them.

Runs lewm.v2.nbhd_targets.build_matched_deltas on a handful of high-count
training perturbations and checks the plan's S3 gate:

  (a) matched control phase composition aligns with perturbed (matched-L1 small,
      and no worse than random matching);
  (b) deltas are finite and non-degenerate (real spread across neighborhoods).

    uv run python scripts/v2/build_nbhd_targets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch

from lewm.v2.data import load_split
from lewm.v2.models import MLPEncoder
from lewm.v2.nbhd_targets import build_matched_deltas
from lewm.v2.train_phase_a import select_device

ENCODER_CKPT = "results/v2/phase_a/checkpoint.pt"
N_PERTS = 8
K = 50


def main():
    device = select_device()
    print(f"device: {device}")

    split = load_split("train")
    encoder = MLPEncoder(split.n_genes, embed_dim=256, hidden_dim=512).to(device)
    ck = torch.load(ENCODER_CKPT, weights_only=False, map_location=device)
    encoder.load_state_dict(ck["encoder"]); encoder.eval()

    # Highest-count non-control perts.
    counts = [(pid, int((split.pert_ids == pid).sum())) for pid in range(1, split.n_perts)]
    counts.sort(key=lambda x: -x[1])
    pert_ids = [pid for pid, _ in counts[:N_PERTS]]
    print(f"perts: {[split.pert_vocab[p] for p in pert_ids]}")

    targets, ctrl_nb_pb = build_matched_deltas(
        encoder=encoder, device=device, split=split, pert_ids=pert_ids, k=K, seed=0,
    )
    print(f"control neighborhoods: {ctrl_nb_pb.shape[0]}  (pseudo-bulk dim {ctrl_nb_pb.shape[1]})")

    per_pert = []
    all_finite = True
    for t in targets:
        finite = bool(np.isfinite(t.deltas).all())
        all_finite = all_finite and finite
        spread = float(t.deltas.std(axis=0).mean())          # mean per-gene std across nbhds
        mean_abs = float(np.abs(t.deltas).mean())
        per_pert.append({
            "pert": t.pert,
            "n_neighborhoods": int(t.deltas.shape[0]),
            "delta_spread": spread,
            "delta_mean_abs": mean_abs,
            "phase_l1_matched": t.phase_l1_matched,
            "phase_l1_random": t.phase_l1_random,
            "finite": finite,
        })
        print(f"  {t.pert:>10s}: nbhds={t.deltas.shape[0]:4d}  "
              f"spread={spread:.4f}  |Δ|={mean_abs:.4f}  "
              f"phaseL1 matched={t.phase_l1_matched:.3f} random={t.phase_l1_random:.3f}")

    mean_spread = float(np.mean([p["delta_spread"] for p in per_pert]))
    mean_l1_matched = float(np.mean([p["phase_l1_matched"] for p in per_pert]))
    mean_l1_random = float(np.mean([p["phase_l1_random"] for p in per_pert]))
    min_nbhds = min(p["n_neighborhoods"] for p in per_pert)

    # Gate: deltas finite + non-degenerate spread; phase match aligns (small L1,
    # and no worse than random — with S1's near-degenerate phase these are ~equal).
    nondegenerate = all_finite and mean_spread > 1e-3 and min_nbhds >= 2
    phase_aligns = mean_l1_matched <= mean_l1_random + 1e-6 and mean_l1_matched < 0.3
    ok = nondegenerate and phase_aligns

    print(f"\n  mean delta spread: {mean_spread:.4f}  (non-degenerate: {nondegenerate})")
    print(f"  phase-L1 matched {mean_l1_matched:.3f} vs random {mean_l1_random:.3f} "
          f"(aligns: {phase_aligns})")
    print(f"\nS3 GATE: {'PASS' if ok else 'FAIL'}")

    out = Path("results/nbhd/S3_targets_sanity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": K, "n_perts": len(per_pert),
        "n_control_neighborhoods": int(ctrl_nb_pb.shape[0]),
        "mean_delta_spread": mean_spread,
        "mean_phase_l1_matched": mean_l1_matched,
        "mean_phase_l1_random": mean_l1_random,
        "all_finite": all_finite,
        "gate_pass": bool(ok),
        "per_pert": per_pert,
        "note": "phase matching is weak by design (S1: phase near-degenerate); "
                "delta spread comes from the neighborhoods, not the matching.",
    }, indent=2))
    print(f"wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
