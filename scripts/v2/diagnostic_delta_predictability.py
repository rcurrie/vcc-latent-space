"""Gate for the delta-per-neighborhood idea: is per-neighborhood delta spread
structured by latent position, or is it noise?

The S4 spread-only test was flat because a constant per-pert delta produces no
perturbation-specific spread. The only live variant is a head delta = f(action,
z_nb) whose delta VARIES with the neighborhood's latent state. That can only work
if the per-neighborhood delta variation is actually predictable from z_nb (and,
for unseen perts, predictable by a model shared across perturbations).

This diagnostic measures that with cross-validated ridge R² (n_nb ~ 110 < 256
dims, so a plain fit would overfit — CV is mandatory), versus a shuffle floor,
for two targets and two levels:

  targets:
    delta   : Δ_nb = matched(perturbed nb) − control nb  (the head's real target)
    pert_pb : perturbed neighborhood pseudobulk          (heterogeneity w/o match noise)
  levels:
    within-pert    : per-pert 5-fold CV — is there ANY latent-correlated structure?
    cross-pert     : leave-perts-out CV with a SHARED z->deviation model —
                     does the structure transfer to unseen perturbations?

Read:
  - within-pert pert_pb R² ~ floor  -> no structure at all; idea is dead.
  - within-pert high but cross-pert ~ floor -> pert-idiosyncratic; won't generalize.
  - cross-pert delta R² meaningfully > floor -> real, transferable; a head is worth building.
  - magnitude matters: R²=0.05 means ~95% of the spread is noise (marginal DES gain).

    uv run python scripts/v2/diagnostic_delta_predictability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch

from lewm.v2.data import load_split
from lewm.v2.models import MLPEncoder
from lewm.v2.nbhd_targets import build_matched_deltas
from lewm.v2.splits import load_internal_val_split
from lewm.v2.train_phase_a import select_device

ENCODER_CKPT = "results/v2/phase_a/checkpoint.pt"
K = 50
LAM = 10.0
N_SPLITS = 5


def _ridge_pred(Ztr, Ytr, Zte, lam):
    """Standardize Z on train stats, ridge-fit Ztr->Ytr (Y already centered), predict Zte."""
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr = (Ztr - zmu) / zsd
    Zte = (Zte - zmu) / zsd
    d = Ztr.shape[1]
    B = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(d), Ztr.T @ Ytr)
    return Zte @ B


def within_pert_r2(Z, Y, lam=LAM, n_splits=N_SPLITS, rng=None):
    """Per-pert K-fold CV ridge R² of z_nb -> (Y - mean). Pooled over genes."""
    n = Z.shape[0]
    if n < n_splits + 2:
        return np.nan
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    ss_res = ss_tot = 0.0
    for i in range(n_splits):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        ymu = Y[tr].mean(0)
        pred = _ridge_pred(Z[tr], Y[tr] - ymu, Z[te], lam) + ymu
        ss_res += float(((Y[te] - pred) ** 2).sum())
        ss_tot += float(((Y[te] - ymu) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def cross_pert_r2(Z, Yc, groups, lam=LAM, n_splits=N_SPLITS, rng=None):
    """Leave-perts-out CV: shared z -> per-pert-centered deviation. R² vs predicting 0."""
    uniq = np.unique(groups)
    if rng is not None:
        rng.shuffle(uniq)
    gfolds = np.array_split(uniq, n_splits)
    ss_res = ss_tot = 0.0
    for i in range(n_splits):
        te = np.isin(groups, gfolds[i])
        tr = ~te
        pred = _ridge_pred(Z[tr], Yc[tr], Z[te], lam)
        ss_res += float(((Yc[te] - pred) ** 2).sum())
        ss_tot += float((Yc[te] ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-by", choices=["phase", "embedding"], default="phase")
    args = ap.parse_args()

    device = select_device()
    print(f"device: {device}  match_by: {args.match_by}")
    rng = np.random.default_rng(0)

    split = load_split("train")
    encoder = MLPEncoder(split.n_genes, embed_dim=256, hidden_dim=512).to(device)
    ck = torch.load(ENCODER_CKPT, weights_only=False, map_location=device)
    encoder.load_state_dict(ck["encoder"]); encoder.eval()

    holdout = set(load_internal_val_split()["holdout_pert_names"])
    pert_ids = [pid for pid in range(1, split.n_perts) if split.pert_vocab[pid] not in holdout]
    print(f"building targets for {len(pert_ids)} training perts...")
    targets, _ = build_matched_deltas(
        encoder=encoder, device=device, split=split, pert_ids=pert_ids, k=K, seed=0,
        match_by=args.match_by,
    )
    print(f"  got {len(targets)} perts")

    # ---- within-pert (real + shuffle floor), for both targets ----
    wp = {"delta": [], "delta_shuf": [], "pert_pb": [], "pert_pb_shuf": []}
    for t in targets:
        Z = t.pert_nb_z.astype(np.float64)
        for name, Y in (("delta", t.deltas.astype(np.float64)),
                        ("pert_pb", t.pert_pb.astype(np.float64))):
            wp[name].append(within_pert_r2(Z, Y, rng=None))
            perm = rng.permutation(Z.shape[0])           # break z<->Y pairing
            wp[name + "_shuf"].append(within_pert_r2(Z[perm], Y, rng=None))

    def m(key):
        return float(np.nanmean(wp[key]))

    # ---- cross-pert (shared model, leave-perts-out) ----
    Zall = np.concatenate([t.pert_nb_z for t in targets]).astype(np.float64)
    groups = np.concatenate([[i] * t.pert_nb_z.shape[0] for i, t in enumerate(targets)])
    Yd = np.concatenate([t.deltas - t.deltas.mean(0) for t in targets]).astype(np.float64)
    Yp = np.concatenate([t.pert_pb - t.pert_pb.mean(0) for t in targets]).astype(np.float64)
    shuf = rng.permutation(Zall.shape[0])
    cp = {
        "delta": cross_pert_r2(Zall, Yd, groups, rng=rng),
        "delta_shuf": cross_pert_r2(Zall[shuf], Yd, groups, rng=rng),
        "pert_pb": cross_pert_r2(Zall, Yp, groups, rng=rng),
        "pert_pb_shuf": cross_pert_r2(Zall[shuf], Yp, groups, rng=rng),
    }

    print("\n=== within-pert CV R² (z_nb -> per-neighborhood target) ===")
    print(f"  delta   : {m('delta'):+.4f}   (shuffle floor {m('delta_shuf'):+.4f})")
    print(f"  pert_pb : {m('pert_pb'):+.4f}   (shuffle floor {m('pert_pb_shuf'):+.4f})")
    print("\n=== cross-pert CV R² (shared model, leave-perts-out) ===")
    print(f"  delta   : {cp['delta']:+.4f}   (shuffle floor {cp['delta_shuf']:+.4f})")
    print(f"  pert_pb : {cp['pert_pb']:+.4f}   (shuffle floor {cp['pert_pb_shuf']:+.4f})")

    # Verdict keys on the DELTA (the perturbation effect), NOT pert_pb — pert_pb
    # R² is trivial autocorrelation (z_nb encodes the same cells' expression).
    # The decisive run is --match-by embedding, which cancels baseline cell-state
    # so the delta isolates the perturbation effect. We require genuine signal:
    # within-pert delta R² > 0.10 (per-pert heterogeneity a head could emit) AND
    # cross-pert delta R² > 0.10 (transfers to unseen perts).
    within_delta = m("delta")
    cross_delta = cp["delta"]
    if within_delta < 0.10:
        verdict = (f"DEAD: within-pert delta R²={within_delta:.3f} ~ 0 — for a given "
                   f"perturbation the delta does not vary with neighborhood state; "
                   f"a z-conditioned head has ~no real per-pert spread to emit.")
    elif cross_delta < 0.10:
        verdict = (f"DEAD for the goal: within-pert delta R²={within_delta:.3f} but "
                   f"cross-pert R²={cross_delta:.3f} — does not transfer to unseen perts.")
    else:
        verdict = (f"ALIVE: within-pert {within_delta:.3f} and cross-pert {cross_delta:.3f} "
                   f"delta R² — a z-conditioned head is worth building.")
    if args.match_by != "embedding":
        verdict += " (NOTE: phase matching leaves baseline cell-state in the delta — "
        verdict += "rerun --match-by embedding for the decisive number.)"
    print(f"\nVERDICT: {verdict}")

    out = Path(f"results/nbhd/S4b_delta_predictability_{args.match_by}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "match_by": args.match_by,
        "k": K, "lam": LAM, "n_splits": N_SPLITS, "n_perts": len(targets),
        "within_pert": {k: m(k) for k in wp},
        "cross_pert": cp,
        "verdict": verdict,
    }, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
