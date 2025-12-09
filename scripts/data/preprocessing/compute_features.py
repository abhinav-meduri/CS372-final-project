"""Compute patent-pair features for train/val/test splits used by the PyTorch novelty model.

Loads precomputed embeddings when available, aligns them to patent IDs, extracts the 10-base
feature set from `FeatureExtractor`, and reports summary statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.features.feature_extract import FeatureExtractor


data_dir = Path("data")
emb_dir = data_dir/"embeddings"
patents_file = data_dir/"sampled/patents_sampled.jsonl"
splits = ["train", "val", "test"]


def load_patents(path: Path = patents_file) -> Dict[str, dict]:
    patents: Dict[str, dict] = {}
    with path.open("r") as f:
        for line in tqdm(f, desc="Loading patents"):
            patent = json.loads(line)
            patents[str(patent["patent_id"])] = patent
    return patents


def load_pairs(path: Path) -> List[dict]:
    pairs: List[dict] = []
    with path.open("r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_embeddings(path: Path = emb_dir/"patent_embeddings.npy") -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    if not path.exists():
        print("Embeddings not found; embedding features will be zero.")
        return None, None
    ids_json = emb_dir/"patent_ids.json"
    ids_npy = emb_dir/"patent_ids.npy"
    embeddings = np.load(path)
    patent_ids: Optional[List[str]] = None
    if ids_json.exists():
        with ids_json.open("r") as f:
            patent_ids = json.load(f)
    elif ids_npy.exists():
        patent_ids = np.load(ids_npy, allow_pickle=True).tolist()
    else:
        print("Embedding IDs not found; cannot align embeddings.")
        return None, None
    return embeddings, patent_ids


def compute_split_features(pairs: List[dict], patents: Dict[str, dict], extractor: FeatureExtractor):
    rows = []
    labels = []
    missing = 0
    for pair in tqdm(pairs, desc="Computing features"):
        pid1 = str(pair["patent_id_1"])
        pid2 = str(pair["patent_id_2"])
        label = pair.get("label", 0)
        p1, p2 = patents.get(pid1), patents.get(pid2)
        if p1 is None or p2 is None:
            missing += 1
            continue
        fv = extractor.extract_features(p1, p2, label=label)
        rows.append(fv.to_array(FeatureExtractor.BASE_FEATURE_NAMES))
        labels.append(label)
    if missing:
        print(f"Skipped {missing} pairs due to missing patents")
    X = np.array(rows)
    y = np.array(labels)
    return X, y


def summarize(name: str, X: np.ndarray, y: np.ndarray):
    if X.size == 0:
        print(f"{name}: no data")
        return
    print(f"{name}: {len(y)} samples, {X.shape[1]} features, positive rate {y.mean():.2%}")
    for i, fname in enumerate(FeatureExtractor.BASE_FEATURE_NAMES):
        col = X[:, i]
        print(f"  {fname}: [{col.min():.3f}, {col.max():.3f}] mean={col.mean():.3f}")


def main():
    patents = load_patents()
    embeddings, ids = load_embeddings()

    extractor = FeatureExtractor()
    if embeddings is not None and ids is not None:
        extractor.set_embeddings(embeddings, ids)
        print("Embeddings loaded into feature extractor")
    else:
        print("Running without embeddings")

    for split in splits:
        path = data_dir/"training"/f"{split}_pairs.jsonl"
        if not path.exists():
            print(f"{path} missing; skipping.")
            continue
        pairs = load_pairs(path)
        X, y = compute_split_features(pairs, patents, extractor)
        summarize(split.upper(), X, y)

    print("Feature computation complete (no files written).")

if __name__ == "__main__":
    main()

