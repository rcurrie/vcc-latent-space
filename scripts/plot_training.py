"""Plot training curves and validation scores from results/vcc/metrics.jsonl.

Run after training to produce summary figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_phase(rows: list[dict]) -> dict:
    out: dict[str, list] = {"2.1": [], "2.2": [], "val": []}
    for r in rows:
        if r.get("event") == "val":
            out["val"].append(r)
        elif r.get("phase") == "2.1":
            out["2.1"].append(r)
        elif r.get("phase") == "2.2":
            out["2.2"].append(r)
    return out


def plot(out_dir: Path) -> None:
    rows = load_metrics(out_dir / "metrics.jsonl")
    if not rows:
        print(f"no metrics in {out_dir}/metrics.jsonl")
        return
    g = split_phase(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Phase 2.1 losses
    ax = axes[0][0]
    ep1 = [r["epoch"] for r in g["2.1"] if "epoch" in r]
    if ep1:
        ax.plot(ep1, [r["pred"] for r in g["2.1"] if "epoch" in r], label="JEPA pred (MSE)")
        ax2 = ax.twinx()
        ax2.plot(ep1, [r["sigreg"] for r in g["2.1"] if "epoch" in r],
                 color="tab:orange", label="SIGReg")
        ax.set_xlabel("epoch"); ax.set_ylabel("pred MSE")
        ax2.set_ylabel("SIGReg", color="tab:orange")
        ax.set_title("Phase 2.1 — homeostatic on controls")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    else:
        ax.set_title("Phase 2.1 (no data)")

    # Phase 2.2 losses
    ax = axes[0][1]
    ep2 = [r["epoch"] for r in g["2.2"] if "epoch" in r]
    if ep2:
        ax.plot(ep2, [r["pred"] for r in g["2.2"] if "epoch" in r], label="pred (MSE)")
        ax.plot(ep2, [r["dec"] for r in g["2.2"] if "epoch" in r], label="decoder (MSE)")
        ax2 = ax.twinx()
        ax2.plot(ep2, [r["sigreg"] for r in g["2.2"] if "epoch" in r],
                 color="tab:green", label="SIGReg")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax2.set_ylabel("SIGReg", color="tab:green")
        ax.set_title("Phase 2.2 — joint training")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    else:
        ax.set_title("Phase 2.2 (no data)")

    # Internal val: pred vs ctrl baselines
    ax = axes[1][0]
    val_ep = [r["epoch"] for r in g["val"]]
    if val_ep:
        ax.plot(val_ep, [r["pred_emb_mse"] for r in g["val"]], label="pred z MSE")
        ax.plot(val_ep, [r["ctrl_emb_mse"] for r in g["val"]],
                "--", label="control z baseline")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE in embedding space")
        ax.set_title("Internal val (held-out perts) — embedding")
        ax.legend()
    else:
        ax.set_title("Internal val (no data yet)")

    # Internal val: gene-space MSE + DR
    ax = axes[1][1]
    if val_ep:
        ax.plot(val_ep, [r["pred_gene_mse"] for r in g["val"]], label="pred gene MSE")
        ax.plot(val_ep, [r["ctrl_gene_mse"] for r in g["val"]],
                "--", label="control gene baseline")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE log1p(CP10k)")
        ax2 = ax.twinx()
        ax2.plot(val_ep, [r["pert_dr"] for r in g["val"]],
                 color="tab:red", marker="o", label="pert DR")
        ax2.axhline(1.0, color="tab:red", ls=":", alpha=0.4)
        ax2.set_ylabel("perturbation DR (>1 = useful)", color="tab:red")
        ax.set_title("Internal val — gene space + DR")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    else:
        ax.set_title("Internal val (no data yet)")

    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/vcc", type=Path)
    args = ap.parse_args()
    plot(args.out_dir)


if __name__ == "__main__":
    main()
