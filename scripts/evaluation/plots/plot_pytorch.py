"""Summarize PyTorch model evaluation metrics without writing plots."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.app.pytorch_classifier import PyTorchPatentClassifier

def main():
    model_dir = Path("models/pytorch_nn")
    features_dir = Path("data/features")

    model = PyTorchPatentClassifier()
    model.load(model_dir)

    X_test = np.load(features_dir/"test_features_v2.X.npy")
    y_test = np.load(features_dir/"test_features_v2.y.npy")
    if X_test.shape[1] == 13:
        keep = [i for i in range(13) if i not in (0, 1, 6)]
        X_test = X_test[:, keep]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = model.evaluate(X_test, y_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr) if len(fpr) else 0.0
    cm = confusion_matrix(y_test, y_pred)

    print("PyTorch evaluation summary (no plots written):")
    print(json.dumps(metrics, indent=2))
    print(f"ROC curve points: {len(fpr)}; AUC={roc_auc_val:.4f}")
    print(f"Confusion matrix:\n{cm}")

if __name__ == "__main__":
    main()


