"""Generate PatentSBERTa embeddings in-memory for the sampled patent set.

1) Load the sampled patents and extract short text (abstract/summary/first-claim).
2) Encode texts with PatentSBERTa in batches and report timing/shape."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings.patent_sberta import PatentEmbedder  # noqa: E402


sample_path = Path("data/sampled/patents_sampled.jsonl")


def load_sampled_patents(path: Path = sample_path) -> List[dict]:
    patents: List[dict] = []
    with path.open("r") as f:
        for line in tqdm(f, desc="Loading"):
            patents.append(json.loads(line))
    return patents


def get_patent_text(patent: dict) -> str:
    if patent.get("abstract"):
        return patent["abstract"][:500]
    if patent.get("summary"):
        return patent["summary"][:500]
    claims = patent.get("claims", [])
    if claims:
        first = claims[0]
        if isinstance(first, dict):
            return (first.get("text") or "")[:500]
        if isinstance(first, str):
            return first[:500]
    return ""


def prepare_texts(patents: List[dict]) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for patent in tqdm(patents, desc="Extracting text"):
        pid = str(patent["patent_id"])
        text = get_patent_text(patent)
        if text.strip():
            ids.append(pid)
            texts.append(text)
    return ids, texts


def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    encoder = PatentEmbedder(batch_size=batch_size)
    start = time.time()
    embeddings = encoder.encode(texts, show_progress=True)
    elapsed = time.time() - start
    rate = len(texts) / elapsed if elapsed else 0.0
    print("Embedding generation complete (dry run; nothing saved).")
    print(f"Samples: {len(texts)} | Shape: {embeddings.shape} | {elapsed:.1f}s ({rate:.1f}/s)")
    return embeddings


def main():
    patents = load_sampled_patents()
    patent_ids, texts = prepare_texts(patents)
    if not texts:
        print("No text available for embedding.")
        return
    _ = generate_embeddings(texts, batch_size=64)
    print("Done (no files written, no FAISS index).")


if __name__ == "__main__":
    main()

