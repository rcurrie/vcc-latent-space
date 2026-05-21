"""One-time: carve 15 perturbations out of the 150 training perts and freeze.

Writes data/vcc/v2_internal_val_split.json. Re-running with the existing file
present will refuse unless --overwrite is passed (the split is meant to be
stable across the entire v2 lifecycle).

    uv run python scripts/v2/freeze_internal_val_split.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lewm.v2.splits import (
    DEFAULT_N_HOLDOUT,
    DEFAULT_PATH,
    DEFAULT_SEED,
    freeze_internal_val_split,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-holdout", type=int, default=DEFAULT_N_HOLDOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    payload = freeze_internal_val_split(
        out_path=args.out,
        n_holdout=args.n_holdout,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.out}")
    print(f"  n_holdout: {payload['n_holdout']} / {payload['n_training_perts_total']}")
    print(f"  seed: {payload['seed']}")
    print(f"  perts: {payload['holdout_pert_names']}")


if __name__ == "__main__":
    main()
