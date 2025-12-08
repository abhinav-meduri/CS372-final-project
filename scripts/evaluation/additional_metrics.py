"""
Compute extra metrics on the test set for the PyTorch model:
- PR-AUC
- Average precision
- Brier score
- Simple calibration bins (ECE-style summary)

Outputs: results/analysis/additional_metrics.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, brier_score_loss

from src.app.pytorch_classifier import PyTorchPatentClassifier


def calibration_bins(probs, labels, n_bins=10):
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_total = 0.0
    total_count = 0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            bins.append({"bin": i, "range": [lo, hi], "count": 0})
            continue
        p_hat = probs[mask].mean()
        p_true = labels[mask].mean()
        count = int(mask.sum())
        bins.append({"bin": i, "range": [lo, hi], "count": count, "pred_mean": p_hat, "true_mean": p_true})
        ece_total += count * abs(p_hat - p_true)
        total_count += count
    ece = ece_total / total_count if total_count else 0.0
    return bins, ece


def main():
    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(root / "data" / "features" / "test_features_v2.X.npy")
    y_test = np.load(root / "data" / "features" / "test_features_v2.y.npy")

    # Align to 10-feature model if needed
    if X_test.shape[1] == 13:
        idx_keep = [i for i in range(13) if i not in [0, 1, 6]]
        X_test = X_test[:, idx_keep]

    clf = PyTorchPatentClassifier()
    clf.load(root / "models" / "pytorch_nn")
    probs = clf.predict_proba(X_test)[:, 1]

    # PR-AUC (area under precision-recall curve)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_test, probs)

    # Brier score
    brier = brier_score_loss(y_test, probs)

    # Calibration bins
    bins_raw, ece = calibration_bins(probs, y_test, n_bins=10)
    # Ensure pure Python types
    bins = []
    for b in bins_raw:
        b_clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in b.items()}
        bins.append(b_clean)

    out = {
        "pr_auc": float(pr_auc),
        "average_precision": float(ap),
        "brier_score": float(brier),
        "ece": float(ece),
        "calibration_bins": bins,
    }

    with (results_dir / "additional_metrics.json").open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

