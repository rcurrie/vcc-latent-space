"""Extract ESM2 protein embeddings for the VCC 18,080-gene panel.

Output:
  data/vcc/gene_esm2_panel.pt — dict with:
    embeddings : (n_genes, 5120) float32 tensor, in var_names order
    coverage   : (n_genes,) bool tensor; True where the gene had an ESM2 row
    var_names  : list[str] — the panel ordering (sanity-check anchor)
    n_covered  : int
    embed_dim  : int

The 1.7% of panel genes without an ESM2 row (mostly antisense / lincRNA /
pseudogenes, plus the lone training-pert symbol mismatch TAZ -> TAFAZZIN)
get a zero vector; downstream ActionEmbed handles them via a learned
unknown-gene fallback embedding.

Run once:
    python scripts/build_esm2_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lewm.data import load_split

ESM2_PATH = Path("data/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
OUT_PATH = Path("data/vcc/gene_esm2_panel.pt")

# A small set of gene-symbol aliases the ESM2 table doesn't cover under our names.
# Mapping is "panel symbol" -> "ESM2 symbol that has the right protein."
# Curated by hand for the perturbation set. Add more here if missing perts cause
# coverage drops.
ALIASES = {
    "TAZ": "TAFAZZIN",   # TAZ symbol officially renamed to TAFAZZIN by HGNC.
}


def main():
    print(f"loading panel from training split ...")
    split = load_split("train")
    panel = list(split.var_names)
    print(f"  {len(panel)} genes")

    print(f"loading ESM2 embeddings: {ESM2_PATH}")
    t0 = time.perf_counter()
    esm = torch.load(str(ESM2_PATH), weights_only=False, map_location="cpu")
    print(f"  {len(esm)} entries, loaded in {time.perf_counter() - t0:.1f}s")
    embed_dim = next(iter(esm.values())).shape[0]
    print(f"  embedding dim: {embed_dim}")

    out = torch.zeros(len(panel), embed_dim, dtype=torch.float32)
    coverage = torch.zeros(len(panel), dtype=torch.bool)

    n_alias_used = 0
    for i, sym in enumerate(panel):
        if sym in esm:
            out[i] = esm[sym].float()
            coverage[i] = True
        elif sym in ALIASES and ALIASES[sym] in esm:
            out[i] = esm[ALIASES[sym]].float()
            coverage[i] = True
            n_alias_used += 1

    n_covered = int(coverage.sum().item())
    print(f"\ncoverage:")
    print(f"  direct match    : {n_covered - n_alias_used}/{len(panel)}")
    print(f"  via aliases     : {n_alias_used}/{len(panel)}")
    print(f"  total covered   : {n_covered}/{len(panel)} ({100*n_covered/len(panel):.1f}%)")
    print(f"  zero placeholder: {len(panel) - n_covered}")

    # Sanity check: all perturbations must be covered (training, val, test).
    print(f"\nperturbation coverage:")
    total_perts = 0
    total_uncovered = []
    for name in ["train", "val", "test"]:
        s = load_split(name) if name != "train" else split
        s_panel_idx = {g: i for i, g in enumerate(panel)}
        perts = [p for p in s.pert_vocab if p != "non-targeting"]
        uncovered = [p for p in perts if (p not in s_panel_idx or not coverage[s_panel_idx[p]])]
        print(f"  {name}: {len(perts) - len(uncovered)}/{len(perts)} covered")
        if uncovered:
            print(f"    UNCOVERED: {uncovered}")
        total_perts += len(perts)
        total_uncovered.extend([(name, p) for p in uncovered])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": out,
        "coverage": coverage,
        "var_names": panel,
        "n_covered": n_covered,
        "embed_dim": embed_dim,
    }
    torch.save(payload, str(OUT_PATH))
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nwrote {OUT_PATH} ({size_mb:.1f} MB)")

    if total_uncovered:
        print(f"\nWARNING: {len(total_uncovered)} perturbations are uncovered. "
              f"Their action_emb will fall back to the learned 'unknown' embedding.")
    else:
        print(f"\nOK: all {total_perts} perturbations across train/val/test have ESM2 embeddings.")


if __name__ == "__main__":
    main()
