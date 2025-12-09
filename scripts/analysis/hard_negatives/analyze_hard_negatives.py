"""Analyze model performance on hard negative patent pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.app.pytorch_classifier import PyTorchPatentClassifier  # noqa: E402
from src.features.feature_extract import FeatureExtractor  # noqa: E402


data_dir = Path("data")
results_dir = Path("results/analysis")
hard_negatives_path = data_dir/"features/test_hard_negatives.json"
patents_path = data_dir/"sampled/patents_sampled.jsonl"
model_path = Path("models/pytorch_nn")


def load_hard_negatives(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Hard negatives not found at {path}. Run scripts/analysis/hard_negatives/generate_test_hard_negatives.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_patents(patent_ids: Iterable[str]) -> Dict[str, Dict]:
    patents: Dict[str, Dict] = {}
    ids = set(patent_ids)
    with open(patents_path, "rb") as f:
        for line in tqdm(f, desc="Loading patents"):
            try:
                patent = json.loads(line)
                pid = str(patent.get("patent_id", ""))
                if pid in ids:
                    patents[pid] = patent
            except Exception:
                continue
    return patents


def extract_features_for_pairs(
    hard_negatives: List[Dict],
    patents: Dict[str, Dict],
    feature_extractor: FeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for pair in tqdm(hard_negatives, desc="Extracting features"):
        pid1 = str(pair["patent_id_1"])
        pid2 = str(pair["patent_id_2"])
        if pid1 not in patents or pid2 not in patents:
            continue
        try:
            feature_vector = feature_extractor.extract_features(
                patents[pid1], patents[pid2]
            )
            rows.append(
                feature_vector.to_array(FeatureExtractor.BASE_FEATURE_NAMES)
            )
            labels.append(0)
        except Exception as e:
            print(f"Feature extraction failed for {pid1}-{pid2}: {e}")
    return np.array(rows), np.array(labels)


def evaluate_pytorch(model: PyTorchPatentClassifier, X: np.ndarray, y: np.ndarray):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    stats = {
        "min": float(np.min(proba)),
        "max": float(np.max(proba)),
        "mean": float(np.mean(proba)),
        "median": float(np.median(proba)),
        "pairs_predicted_novel": int(np.sum(proba > 0.5)),
        "pairs_confident_not_novel": int(np.sum(proba < 0.3)),
        "pairs_uncertain": int(np.sum((proba >= 0.3) & (proba < 0.5))),
    }
    return metrics, stats


def summarize_confusion(cm: List[List[int]]) -> Dict[str, float]:
    arr = np.array(cm)
    if arr.shape != (2, 2):
        return {"tn": 0, "fp": 0}
    return {"tn": int(arr[0, 0]), "fp": int(arr[0, 1])}


def save_results(
    num_pairs: int,
    metrics: Dict,
    stats: Dict,
    summary: Dict,
    path: Path = results_dir/"hard_negatives_analysis.json",
):
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_pairs": num_pairs,
        "pytorch": metrics,
        "pytorch_probability_stats": stats,
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to: {path}")


def run():
    hard_negatives = load_hard_negatives(hard_negatives_path)
    patent_ids = {
        str(p["patent_id_1"]) for p in hard_negatives
    } | {str(p["patent_id_2"]) for p in hard_negatives}

    patents = load_patents(patent_ids)
    st_model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")
    feature_extractor = FeatureExtractor(st_model)

    X, y = extract_features_for_pairs(hard_negatives, patents, feature_extractor)
    print(f"Prepared {len(X)} pairs; feature shape: {X.shape}")

    model = PyTorchPatentClassifier()
    model.load(model_path)

    metrics, stats = evaluate_pytorch(model, X, y)
    cm_summary = summarize_confusion(metrics["confusion_matrix"])
    summary = {
        "pytorch_fp_rate": cm_summary["fp"] / len(y) if len(y) else 0,
        "pytorch_tn_rate": cm_summary["tn"] / len(y) if len(y) else 0,
        "pytorch_accuracy": metrics["accuracy"],
    }

    save_results(len(X), metrics, stats, summary)


if __name__ == "__main__":
    run()