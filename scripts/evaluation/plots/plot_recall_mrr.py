"""
Plot Recall@K and MRR from results/analysis/recall_mrr.json.
Style aligned with other evaluation plots (neutral palette, clean grid).
Output: results/plots/recall_mrr.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent.parent.parent.parent
    metrics_path = root / "results" / "metrics" / "recall_mrr.json"
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with metrics_path.open() as f:
        data = json.load(f)

    recall_at_k = data.get("recall_at_k", {})
    mrr = data.get("mrr", 0)

    ks = sorted(int(k) for k in recall_at_k.keys())
    recalls = [recall_at_k[str(k)] for k in ks]

    plt.style.use("default")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2E86AB", "#4FA3BF", "#6FB9CF", "#8FD0DF"]
    bars = ax.bar([str(k) for k in ks], recalls, color=colors[: len(ks)], edgecolor="#1a1a1a", linewidth=0.8, alpha=0.9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@K", fontsize=11)
    ax.set_xlabel("K", fontsize=11)
    ax.set_title("Recall@K and MRR (citation positives)", fontsize=13, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#1a1a1a")

    ax.text(
        0.02,
        0.94,
        f"MRR: {mrr:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor="#999999"),
    )

    plt.tight_layout()
    out_dir = root / "results" / "plots" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recall_mrr.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

