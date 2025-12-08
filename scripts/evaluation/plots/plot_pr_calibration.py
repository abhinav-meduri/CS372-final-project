"""
Plot PR curve and reliability (calibration) diagram for the PyTorch model.
Outputs:
- results/plots/eval/pr_curve.png
- results/plots/eval/calibration.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from src.app.pytorch_classifier import PyTorchPatentClassifier


def main():
    root = Path(__file__).resolve().parent.parent.parent.parent
    plots_dir = root / "results" / "plots" / "eval"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_test = np.load(root / "data" / "features" / "test_features_v2.X.npy")
    y_test = np.load(root / "data" / "features" / "test_features_v2.y.npy")
    if X_test.shape[1] == 13:
        idx_keep = [i for i in range(13) if i not in [0, 1, 6]]
        X_test = X_test[:, idx_keep]

    # Load model
    clf = PyTorchPatentClassifier()
    clf.load(root / "models" / "pytorch_nn")
    probs = clf.predict_proba(X_test)[:, 1]

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    plt.style.use("default")
    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="#2E86AB", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (PyTorch model)")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Calibration bins from saved metrics if available
    bins_path = root / "results" / "metrics" / "additional_metrics.json"
    bins = None
    if bins_path.exists():
        with bins_path.open() as f:
            meta = json.load(f)
            bins = meta.get("calibration_bins", None)

    if bins:
        bin_centers = [np.mean(b["range"]) for b in bins]
        pred_mean = [b.get("pred_mean", 0) for b in bins]
        true_mean = [b.get("true_mean", 0) for b in bins]

        plt.figure(figsize=(6.5, 5))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.plot(bin_centers, pred_mean, "o-", color="#2E86AB", label="Predicted mean")
        plt.plot(bin_centers, true_mean, "s-", color="#A23B72", label="True mean")
        plt.xlabel("Predicted probability bin center")
        plt.ylabel("Fraction positive")
        plt.title("Calibration (Reliability) Curve")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "calibration.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()

