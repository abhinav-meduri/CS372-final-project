"""Generate hard negative patent pairs for the test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

data_dir = Path("data")
features_dir = data_dir/"features"
patents_path = data_dir/"sampled/patents_sampled.jsonl"
output_path = features_dir/"test_hard_negatives.json"

try:
    import orjson

    def load_jsonl_line(line):
        return orjson.loads(line)

except ImportError:

    def load_jsonl_line(line):
        return json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)

def load_test_patent_ids() -> List[str]:
    y_test = np.load(features_dir/"test_features_v2.y.npy")
    pairs_file = features_dir/"test_features_v2.pairs.json"
    if not pairs_file.exists():
        pairs_file = features_dir/"test_features.pairs.json"
    with open(pairs_file, "r") as f:
        pairs = json.load(f)
    ids = set()
    for pair in pairs[: len(y_test)]:
        ids.add(str(pair[0]))
        ids.add(str(pair[1]))
    return list(ids)

def load_patents(patent_ids: Iterable[str]) -> Dict[str, Dict]:
    patents: Dict[str, Dict] = {}
    ids = set(patent_ids)
    with open(patents_path, "rb") as f:
        for line in tqdm(f, desc="Loading"):
            try:
                patent = load_jsonl_line(line)
                pid = str(patent.get("patent_id", ""))
                if pid in ids:
                    patents[pid] = patent
            except Exception:
                continue
    return patents

def compute_embeddings(patents: Dict[str, Dict], st_model: SentenceTransformer):
    embeddings: Dict[str, np.ndarray] = {}
    for pid, patent in tqdm(patents.items(), desc="Embedding"):
        text = f"{patent.get('title', '')} {patent.get('abstract', '')}".strip()
        if not text:
            continue
        embeddings[pid] = st_model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
    return embeddings

def find_hard_negatives(
    patents: Dict[str, Dict],
    embeddings: Dict[str, np.ndarray],
    threshold: float = 0.85,
    max_pairs: int = 500,
):
    results: List[Dict] = []
    patent_ids = list(patents.keys())
    for i in tqdm(range(len(patent_ids)), desc="Searching"):
        if len(results) >= max_pairs:
            break
        pid1 = patent_ids[i]
        emb1 = embeddings.get(pid1)
        if emb1 is None:
            continue
        for pid2 in patent_ids[i + 1 :]:
            emb2 = embeddings.get(pid2)
            if emb2 is None:
                continue
            similarity = float(np.dot(emb1, emb2))
            if similarity <= threshold:
                continue
            p1, p2 = patents[pid1], patents[pid2]
            cpc1 = set(p1.get("cpc", [])[:3]) if p1.get("cpc") else set()
            cpc2 = set(p2.get("cpc", [])[:3]) if p2.get("cpc") else set()
            year1, year2 = p1.get("year", 0), p2.get("year", 0)
            if len(cpc1 & cpc2) == 0 or abs(year1 - year2) > 5:
                results.append(
                    {
                        "patent_id_1": pid1,
                        "patent_id_2": pid2,
                        "similarity": similarity,
                        "cpc_overlap": len(cpc1 & cpc2),
                        "year_diff": abs(year1 - year2),
                    }
                )
                if len(results) >= max_pairs:
                    break
    return results

def main():
    test_ids = load_test_patent_ids()
    patents = load_patents(set(test_ids))

    st_model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")
    embeddings = compute_embeddings(patents, st_model)

    hard_negatives = find_hard_negatives(
        patents, embeddings, threshold=0.85, max_pairs=500
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hard_negatives, f, indent=2)

    if hard_negatives:
        sims = [hn["similarity"] for hn in hard_negatives]
        print(f"Count: {len(hard_negatives)}")
        print(f"Mean similarity: {np.mean(sims):.3f}")
        print(f"Min similarity: {np.min(sims):.3f}")
        print(f"Max similarity: {np.max(sims):.3f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

