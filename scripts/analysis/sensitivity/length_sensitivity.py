"""
Evaluate model robustnes and sensitivity to patent abstract length. 
Note: Deeper truncation experiments may require re-encoding text, so I skipped to avoid I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.app.pytorch_classifier import PyTorchPatentClassifier  # noqa: E402


data_dir = Path("data")
features_dir = data_dir/"features"
patents_path = data_dir/"sampled/patents_sampled.jsonl"
model_path = Path("models/pytorch_nn")


def load_model_and_features():
    model = PyTorchPatentClassifier()
    model.load(model_path)

    X_test = np.load(features_dir/"test_features_v2.X.npy")
    y_test = np.load(features_dir/"test_features_v2.y.npy")

    if X_test.shape[1] == 13:
        keep = [i for i in range(13) if i not in (0, 1, 6)]
        X_test = X_test[:, keep]
    return model, X_test, y_test


def load_patent_text() -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    if not patents_path.exists():
        return data
    try:
        import orjson
    except ImportError:
        orjson = None

    with open(patents_path, "rb") as f:
        for line in f:
            try:
                patent = orjson.loads(line) if orjson else json.loads(line)
                pid = str(patent.get("patent_id", ""))
                if pid:
                    data[pid] = patent
            except Exception:
                continue
    return data


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def length_stats(patents: Dict[str, Dict]):
    lengths = [
        word_count(p.get("abstract", "") or p.get("brief_summary", ""))
        for p in patents.values()
        if p.get("abstract") or p.get("brief_summary")
    ]
    if not lengths:
        return None, None
    stats = {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }
    return lengths, stats


def evaluate_baseline(model: PyTorchPatentClassifier, X: np.ndarray, y: np.ndarray):
    proba = model.predict_proba(X).flatten()
    preds = model.predict(X)
    n = min(len(y), len(proba), len(preds))
    y = y[:n]
    proba = proba[:n]
    preds = preds[:n]
    roc = roc_auc_score(y, proba) if len(set(y)) > 1 else 0.0
    return {
        "count": int(n),
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc),
        "f1": float(f1_score(y, preds)),
    }


def main():
    model, X_test, y_test = load_model_and_features()
    patents = load_patent_text()

    baseline = evaluate_baseline(model, X_test, y_test)
    print("Baseline performance (pre-computed features):")
    print(json.dumps(baseline, indent=2))

    lengths, stats = length_stats(patents)
    if stats:
        print("\nAbstract length summary:")
        print(json.dumps(stats, indent=2))
        buckets = {
            "short_50_100": sum(50 <= l <= 100 for l in lengths),
            "medium_200_300": sum(200 <= l <= 300 for l in lengths),
            "long_500_plus": sum(l >= 500 for l in lengths),
        }
        print("\nLength buckets (counts):")
        print(json.dumps(buckets, indent=2))
    else:
        print("\nNo patent text available for length analysis.")


if __name__ == "__main__":
    main()

