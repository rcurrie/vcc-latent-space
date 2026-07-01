# Negative result: neighborhood aggregation does not lift DES

**TL;DR.** We tested whether overlapping-KNN neighborhood pseudo-bulk + a spread-aware
loss could lift the hybrid simple-head's Differential Expression Score (DES) without
sinking PDS. It does not. The perturbation signal our data carries is a *mean* shift;
the within-perturbation *spread* that neighborhoods would organize is ~98% noise, so
there is nothing for a spread-based method to exploit. Along the way we wired in the
official `cell-eval` scorer, which re-baselined the model on real leaderboard metrics.

The experiment plan is [des_neighborhood_plan.md](des_neighborhood_plan.md); every step
wrote metrics to `results/nbhd/`.

## The thesis under test

`k` (neighborhood size) is a bias/variance dial between per-cell (k=1) and the global
centroid (k=N). The hope: mid-`k` overlapping-KNN pseudo-bulk on the frozen SSL encoder,
plus a DES-aware loss, would lift DES off its floor by giving predictions realistic
within-perturbation spread that a DE test could reward.

## What we found, step by step

**S0 — the in-repo DES was blind to spread.** Tracing the scorer showed both the latent
and hybrid DES collapse each perturbation to a *mean* profile before scoring (top-K DEG
Jaccard on the mean delta). Spread is invisible to it. So the thesis could not pay off
without changing the scorer. We resolved this by wiring in the official `cell-eval`
package (`overlap_at_N` DES, a real pdex differential-expression test on the predicted
population) and adding population-emitting inference — see
[scripts/v2/score_celleval.py](../scripts/v2/score_celleval.py).

**Official cell-eval re-baseline (hybrid simple-head).** With real leaderboard metrics:

| split | DES (`overlap_at_N`) | PDS (`discrimination_score_l1`) | MAE |
|---|---|---|---|
| Val (50 perts) | 0.077 | 0.552 | 0.021 |
| Test (100 perts) | 0.055 | 0.556 | 0.020 |

PDS agrees with the old full-panel approximation (0.556 vs 0.538), a good sanity check.

**S1 — cell-cycle phase explains almost none of the within-pert variance.** Extending the
gaussianity diagnostic with `sc.tl.score_genes_cell_cycle`: phase explains ~3% (discrete
bins) to ~4% (continuous S/G2M-score regression) of the 12.68 intra-perturbation latent
variance. Neighborhoods would not just be recovering phase bins — but the headroom is
small. (`results/nbhd/S1_phase_headroom.json`)

**S2 — the primitive works.** [src/lewm/neighborhoods.py](../src/lewm/neighborhoods.py):
overlapping-KNN neighborhoods + count-corrected pseudo-bulk (sum raw UMIs → CP10k →
log1p). All-cells pseudo-bulk reproduces the global centroid at corr 0.9997.

**S3 — matched delta targets are non-degenerate.** Per-neighborhood matched deltas
{Δ_nb} build cleanly (~110 neighborhoods/pert, spread 0.034). Phase-matching looks
near-perfect (matched-L1 0.002 vs 0.906 random) precisely *because* phase is so
degenerate (S1) that any composition has a near-identical control match — the matching
removes a small confound, not signal.

**S4 — the core spread-only test: FAIL.** Train the simple head on the per-neighborhood
delta sets; at inference apply the predicted delta to control neighborhoods to emit a
population; score with official cell-eval (k=50):

| config | Val DES | Val PDS | Test DES | Test PDS |
|---|---|---|---|---|
| baseline ckpt + cells (floor) | 0.077 | 0.552 | 0.055 | 0.556 |
| baseline ckpt + neighborhoods | 0.078 | 0.551 | 0.059 | 0.544 |
| **S4 (nbhd-trained + neighborhoods)** | 0.073 | 0.554 | 0.052 | 0.531 |

DES does not move up. Neighborhood inference alone gives at most a +0.003 Test bump that
is flat on Val and costs PDS; retraining on neighborhood targets actively hurts. The gate
("DES up AND PDS ≥ ~0.53") fails.

**S4b — the last live variant, killed cleanly.** The only version with a chance was a
head whose delta *varies* with neighborhood latent state, `delta = f(action, z_nb)`. That
requires the per-neighborhood delta variation to be predictable from `z_nb`. Cross-validated
ridge R² (`scripts/v2/diagnostic_delta_predictability.py`):

| target / control matching | within-pert R² | cross-pert R² |
|---|---|---|
| delta, phase-matched | +0.018 | +0.234 |
| **delta, embedding-matched** | **+0.006** | **+0.049** |
| pert_pb (autocorrelation control) | +0.75 | +0.63 |

The phase-matched 0.23 was baseline cell-state leaking into the delta through imperfect
matching. With embedding-based matching (control neighborhood at nearest latent position,
cancelling baseline state), genuine perturbation-effect predictability collapses to
0.5–5% — noise. `pert_pb`'s high R² is trivial autocorrelation (`z_nb` is the encoder
embedding of the same cells). Within-pert delta R² ≈ 0 is decisive: for a given
perturbation, the delta does not vary with neighborhood state, so a state-conditioned head
has essentially no real per-perturbation spread to emit.

## Why it fails (the through-line)

Every diagnostic points the same way: the within-perturbation structure is dominated by
noise. The encoder's inter/intra latent variance ratio is **0.015** (intra-class variance
~67× the between-class signal), phase explains ~3% of it, and the delta's genuine
state-dependence is ~0. The perturbation signal is a mean shift; there is no structured
spread for neighborhood aggregation to organize. A constant delta is, to first order, the
right model — which is exactly why the hybrid simple-head already captured what was there.

## What this does and does not say

It kills the hypothesis that **spread/heterogeneity** is a lever for DES on this data. It
says nothing about improving the **mean** delta prediction (better action representation,
encoder, or head) — the direction the VCC winners pursued and the natural next avenue if
DES is still the target.

## What we keep

- Official `cell-eval` wired in and the model re-baselined on real metrics (clears the
  standing "never ran cell-eval" follow-up).
- Reusable neighborhood primitive and the full negative-result chain in `results/nbhd/`.
