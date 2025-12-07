"""
Generate Hard Negatives for Test Set

The edge case analysis found only 1 hard negative in the test set.
This script generates more hard negatives specifically for the test set
to enable proper edge case evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Try to use orjson for faster JSON parsing, fallback to standard json
try:
    import orjson
    def load_jsonl_line(line):
        return orjson.loads(line)
except ImportError:
    def load_jsonl_line(line):
        return json.loads(line.decode('utf-8') if isinstance(line, bytes) else line)

def load_test_patent_ids():
    """Load patent IDs in test set."""
    features_dir = Path("data/features")
    
    # Load test labels to get test set size
    y_test = np.load(features_dir / "test_features_v2.y.npy")
    
    # Load patent pairs and extract unique IDs
    # Try v2 first, then fallback to v1
    pairs_file = features_dir / "test_features_v2.pairs.json"
    if not pairs_file.exists():
        pairs_file = features_dir / "test_features.pairs.json"
    
    if pairs_file.exists():
        with open(pairs_file, "r") as f:
            pairs = json.load(f)
        
        # Extract unique patent IDs from pairs
        test_ids = set()
        for pair in pairs[:len(y_test)]:
            test_ids.add(str(pair[0]))
            test_ids.add(str(pair[1]))
        
        return list(test_ids)
    else:
        raise FileNotFoundError(f"Test pairs file not found. Tried: test_features_v2.pairs.json and test_features.pairs.json")

def load_patents(patent_ids):
    """Load patent data for given IDs."""
    patents_path = Path("data/sampled/patents_sampled.jsonl")
    patents = {}
    
    print(f"Loading patents for {len(patent_ids)} IDs...")
    with open(patents_path, 'rb') as f:
        for line in tqdm(f, desc="Loading"):
            try:
                patent = load_jsonl_line(line)
                pid = str(patent.get('patent_id', ''))
                if pid in patent_ids:
                    patents[pid] = patent
            except:
                continue
    
    return patents

def compute_embeddings(patents, st_model):
    """Compute embeddings for all patents."""
    print("Computing embeddings...")
    embeddings = {}
    
    for pid, patent in tqdm(patents.items(), desc="Embedding"):
        text = f"{patent.get('title', '')} {patent.get('abstract', '')}"
        if text.strip():
            emb = st_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            embeddings[pid] = emb
    
    return embeddings

def find_hard_negatives(patents, embeddings, threshold=0.85, max_pairs=1000):
    """Find hard negative pairs (high similarity but different patents)."""
    print(f"Finding hard negatives (similarity > {threshold})...")
    
    patent_ids = list(patents.keys())
    hard_negatives = []
    
    # Compute pairwise similarities
    for i in tqdm(range(len(patent_ids)), desc="Searching"):
        if len(hard_negatives) >= max_pairs:
            break
        
        pid1 = patent_ids[i]
        if pid1 not in embeddings:
            continue
        
        emb1 = embeddings[pid1]
        
        for j in range(i + 1, len(patent_ids)):
            pid2 = patent_ids[j]
            if pid2 not in embeddings:
                continue
            
            emb2 = embeddings[pid2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            
            if similarity > threshold:
                # Check if they're actually different (different CPC or different year)
                p1 = patents[pid1]
                p2 = patents[pid2]
                
                # Different if different primary CPC or different year
                cpc1 = set(p1.get('cpc', [])[:3]) if p1.get('cpc') else set()
                cpc2 = set(p2.get('cpc', [])[:3]) if p2.get('cpc') else set()
                year1 = p1.get('year', 0)
                year2 = p2.get('year', 0)
                
                # Consider hard negative if high similarity but different domain
                if len(cpc1 & cpc2) == 0 or abs(year1 - year2) > 5:
                    hard_negatives.append({
                        'patent_id_1': pid1,
                        'patent_id_2': pid2,
                        'similarity': float(similarity),
                        'cpc_overlap': len(cpc1 & cpc2),
                        'year_diff': abs(year1 - year2)
                    })
                    
                    if len(hard_negatives) >= max_pairs:
                        break
    
    return hard_negatives

def main():
    print("="*70)
    print("GENERATING HARD NEGATIVES FOR TEST SET")
    print("="*70)
    
    # Load test patent IDs
    test_ids = load_test_patent_ids()
    print(f"Test set size: {len(test_ids)} patents")
    
    # Load patents
    patents = load_patents(set(test_ids))
    print(f"Loaded {len(patents)} patents")
    
    # Load embeddings model
    print("\nLoading PatentSBERTa...")
    st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    # Compute embeddings
    embeddings = compute_embeddings(patents, st_model)
    
    # Find hard negatives
    hard_negatives = find_hard_negatives(patents, embeddings, threshold=0.85, max_pairs=500)
    
    print(f"\nFound {len(hard_negatives)} hard negative pairs")
    
    # Save results
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "test_hard_negatives.json", "w") as f:
        json.dump(hard_negatives, f, indent=2)
    
    print(f"\nSaved to: {output_dir / 'test_hard_negatives.json'}")
    
    # Summary
    if hard_negatives:
        similarities = [hn['similarity'] for hn in hard_negatives]
        print(f"\nHard Negative Statistics:")
        print(f"  Count: {len(hard_negatives)}")
        print(f"  Mean similarity: {np.mean(similarities):.3f}")
        print(f"  Min similarity: {np.min(similarities):.3f}")
        print(f"  Max similarity: {np.max(similarities):.3f}")
        print(f"\nThese can be used to evaluate model on edge cases.")

if __name__ == '__main__':
    main()

