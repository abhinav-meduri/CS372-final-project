"""Summarize Recall@K and MRR metrics without writing plots."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent.parent
    metrics_path = root/"results"/"metrics"/"recall_mrr.json"
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with metrics_path.open() as f:
        data = json.load(f)

    recall_at_k = data.get("recall_at_k", {})
    mrr = data.get("mrr", 0)

    ks = sorted(int(k) for k in recall_at_k.keys())
    recalls = [recall_at_k[str(k)] for k in ks]

    print("Recall@K summary (no plots written):")
    for k, v in zip(ks, recalls):
        print(f"  K={k}: recall={v:.4f}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()

