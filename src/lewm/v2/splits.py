"""Internal-val split management for v2.

We carve 15 perturbations out of the 150 non-control training perts and
freeze the choice in a JSON file at data/vcc/v2_internal_val_split.json.
This is the held-out slice used during Phase B for Latent-PDS / MMD
monitoring without touching the official 50-pert validation file.

Discipline:
  - Once frozen, never re-roll without bumping the JSON's `seed` and version.
  - Controls always remain in `train` (they are not perturbations).
  - The official 50-pert validation file is a separate split file
    (adata_Validation.h5ad); this module never references it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lewm.data import load_split

DEFAULT_PATH = Path("data/vcc/v2_internal_val_split.json")
DEFAULT_N_HOLDOUT = 15
DEFAULT_SEED = 1742


def freeze_internal_val_split(
    out_path: Path = DEFAULT_PATH,
    n_holdout: int = DEFAULT_N_HOLDOUT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict:
    """Carve held-out perturbations from the training split and persist.

    Picks `n_holdout` pert names uniformly at random from the 150 non-control
    training perts. Writes the chosen names + metadata to JSON. Returns the
    written dict.

    The persisted file is the source of truth at train/eval time — load it,
    don't re-roll.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass overwrite=True to replace")

    split = load_split("train")
    # pert_vocab[0] is the control label; skip it.
    noncontrol_perts = split.pert_vocab[1:]
    if n_holdout >= len(noncontrol_perts):
        raise ValueError(
            f"n_holdout={n_holdout} >= {len(noncontrol_perts)} available training perts"
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(noncontrol_perts), size=n_holdout, replace=False)
    chosen = sorted(noncontrol_perts[i] for i in chosen_idx)

    payload = {
        "version": 1,
        "seed": int(seed),
        "n_holdout": int(n_holdout),
        "n_training_perts_total": len(noncontrol_perts),
        "holdout_pert_names": chosen,
        "notes": (
            "Carved from training split's non-control perts. Frozen — "
            "do not re-roll without bumping version."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_internal_val_split(path: Path = DEFAULT_PATH) -> dict:
    """Load the persisted internal-val split."""
    return json.loads(Path(path).read_text())


def partition_indices_for_internal_val(
    split,
    holdout_pert_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Given a loaded VCCSplit + held-out pert names, return (train_idx, val_idx).

    Controls always go to train_idx. Cells whose target_gene is in
    holdout_pert_names go to val_idx; all other cells go to train_idx.
    """
    holdout_set = set(holdout_pert_names)
    is_holdout = np.array([g in holdout_set for g in split.target_genes])
    val_idx = np.where(is_holdout)[0].astype(np.int64)
    train_idx = np.where(~is_holdout)[0].astype(np.int64)
    return train_idx, val_idx
