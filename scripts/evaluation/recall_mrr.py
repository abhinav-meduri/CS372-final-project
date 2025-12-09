"""
Compute Recall@K and MRR using known citation pairs as positives.
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys

from src.app.pytorch_classifier import PyTorchPatentClassifier


def load_pairs_from_jsonl(path: Path):
    pairs = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            p1 = row.get("patent_id_1") or row.get("patent_id") or row.get("citing")
            p2 = row.get("patent_id_2") or row.get("cited")
            label = row.get("label", 1)
            if p1 and p2:
                pairs.append((p1, p2, label))
    return pairs


def build_positive_map(pairs):
    pos = defaultdict(set)
    for p1, p2, label in pairs:
        if label == 1:
            pos[p1].add(p2)
    return pos


def compute_recall_mrr(scores, positives, k_values):
    recall_at_k = {k: 0 for k in k_values}
    total_pos = 0
    mrr_total = 0
    mrr_count = 0

    for query, scored in scores.items():
        if query not in positives or len(positives[query]) == 0:
            continue
        pos_set = positives[query]
        total_pos += len(pos_set)

        ranked = sorted(scored, key=lambda x: -x[1])
        top_ids = [pid for pid, _ in ranked]

        # Recall@K
        for k in k_values:
            hits = sum(1 for pid in top_ids[:k] if pid in pos_set)
            recall_at_k[k] += hits

        # MRR
        rr = 0
        for rank, pid in enumerate(top_ids, 1):
            if pid in pos_set:
                rr = 1 / rank
                break
        if rr > 0:
            mrr_total += rr
            mrr_count += 1

    recall_at_k = {k: (v / total_pos if total_pos else 0) for k, v in recall_at_k.items()}
    mrr = mrr_total / mrr_count if mrr_count else 0
    return recall_at_k, mrr, total_pos, mrr_count


def main():
    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load positives (citations)
    pairs_path = root / "data" / "training" / "test_pairs.jsonl"
    fallback_citations = root / "data" / "citations" / "filtered_citations.jsonl"
    if pairs_path.exists():
        pairs = load_pairs_from_jsonl(pairs_path)
    elif fallback_citations.exists():
        pairs = load_pairs_from_jsonl(fallback_citations)
    else:
        print("No citation pairs file found.")
        sys.exit(1)

    positives = build_positive_map(pairs)

    X_test = np.load(root / "data" / "features" / "test_features_v2.X.npy")
    pairs_list = json.load(open(root / "data" / "features" / "test_features.pairs.json"))

    if X_test.shape[1] == 13:
        idx_keep = [i for i in range(13) if i not in [0, 1, 6]]
        X_test = X_test[:, idx_keep]

    clf = PyTorchPatentClassifier()
    clf.load(root / "models" / "pytorch_nn")

    probs = clf.predict_proba(X_test)[:, 1]

    scores = defaultdict(list)
    for (pid1, pid2), p in zip(pairs_list, probs):
        scores[pid1].append((pid2, p))

    k_values = [1, 5, 10, 20]
    recall_at_k, mrr, total_pos, mrr_count = compute_recall_mrr(scores, positives, k_values)

    out = {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "total_positive_edges": total_pos,
        "queries_with_positive": mrr_count,
    }

    with (results_dir / "recall_mrr.json").open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

