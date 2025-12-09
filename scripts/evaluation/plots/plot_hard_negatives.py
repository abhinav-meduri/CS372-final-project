"""Summarize hard negative analysis results without saving plots."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def summarize_hard_negatives(results_file: Path):
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    with results_file.open() as f:
        results = json.load(f)
    pytorch_metrics = results.get("pytorch", {})
    prob_stats = results.get("pytorch_probability_stats", {})

    cm = np.array(pytorch_metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    acc = pytorch_metrics.get("accuracy", 0.0)
    f1 = pytorch_metrics.get("f1", 0.0)
    roc = pytorch_metrics.get("roc_auc", 0.0)

    print("Hard negatives summary (no plots written):")
    print(f"  Accuracy={acc:.4f} F1={f1:.4f} ROC-AUC={roc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    if prob_stats:
        print(
            f"  Prob stats: min={prob_stats.get('min', 0):.4f} "
            f"mean={prob_stats.get('mean', 0):.4f} "
            f"median={prob_stats.get('median', 0):.4f} "
            f"max={prob_stats.get('max', 0):.4f}"
        )


def main():
    results_file = Path("results/analysis/hard_negatives_analysis.json")
    summarize_hard_negatives(results_file)


if __name__ == "__main__":
    main()
