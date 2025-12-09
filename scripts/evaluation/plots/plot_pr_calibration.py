"""Compute PR curve and calibration summary for the PyTorch model (no plots saved)."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from src.app.pytorch_classifier import PyTorchPatentClassifier


def main():
    root = Path(__file__).resolve().parent.parent.parent.parent
    X_test = np.load(root/"data"/"features"/"test_features_v2.X.npy")
    y_test = np.load(root/"data"/"features"/"test_features_v2.y.npy")
    if X_test.shape[1] == 13:
        idx_keep = [i for i in range(13) if i not in [0, 1, 6]]
        X_test = X_test[:, idx_keep]

    clf = PyTorchPatentClassifier()
    clf.load(root/"models"/"pytorch_nn")
    probs = clf.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    bins_path = root/"results"/"metrics"/"additional_metrics.json"
    bins = None
    if bins_path.exists():
        with bins_path.open() as f:
            meta = json.load(f)
            bins = meta.get("calibration_bins", None)

    print("PR/Calibration summary (no plots written):")
    print(f"  PR AUC: {pr_auc:.4f}")
    if bins:
        bin_centers = [np.mean(b["range"]) for b in bins]
        pred_mean = [b.get("pred_mean", 0) for b in bins]
        true_mean = [b.get("true_mean", 0) for b in bins]
        paired = list(zip(bin_centers, pred_mean, true_mean))
        print("  Calibration bins (center, pred_mean, true_mean):")
        for c, p, t in paired:
            print(f"    {c:.2f}: pred={p:.3f} true={t:.3f}")
    else:
        print("  No calibration bins found.")


if __name__ == "__main__":
    main()

