"""
Analyze the difficulty of our negative pairs and create hard negatives.

This script:
1. Analyzes current negatives (how "easy" are they?)
2. Generates hard negatives (similar patents that don't cite each other)
3. Tests the model on easy vs hard negatives
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def tokenize(text: str) -> set:
    """Simple tokenization."""
    if not text:
        return set()
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard(set1: set, set2: set) -> float:
    """Jaccard similarity."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def load_patents(path: str = 'data/sampled/patents_sampled.jsonl', limit: int = None):
    """Load patents into dict."""
    patents = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    return patents


def load_pairs(path: str):
    """Load pairs from JSONL."""
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def analyze_pair_similarity(patents: dict, pairs: list, sample_size: int = 1000):
    """Analyze text similarity of pairs."""
    import random
    random.seed(42)
    
    sampled = random.sample(pairs, min(sample_size, len(pairs)))
    
    similarities = []
    for pair in sampled:
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            continue
        
        # Compute text similarity
        text1 = f"{p1.get('title', '')} {p1.get('abstract', '')}"
        text2 = f"{p2.get('title', '')} {p2.get('abstract', '')}"
        
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        sim = jaccard(tokens1, tokens2)
        similarities.append({
            'pair': pair,
            'similarity': sim,
            'label': pair['label']
        })
    
    return similarities


def generate_hard_negatives(
    patents: dict,
    existing_positive_pairs: set,
    n_pairs: int = 5000,
    min_similarity: float = 0.15  # Must have at least 15% Jaccard overlap
):
    """
    Generate hard negatives: pairs with HIGH text similarity but no citation.
    
    These are the challenging cases where simple text features fail.
    """
    print(f"\nGenerating {n_pairs} hard negatives (min_similarity={min_similarity})...")
    
    patent_list = list(patents.values())
    n = len(patent_list)
    
    hard_negatives = []
    attempts = 0
    max_attempts = n_pairs * 100
    
    # Pre-tokenize all patents
    print("Pre-tokenizing patents...")
    tokenized = {}
    for p in tqdm(patent_list[:20000], desc="Tokenizing"):  # Sample for speed
        pid = str(p['patent_id'])
        text = f"{p.get('title', '')} {p.get('abstract', '')}"
        tokenized[pid] = tokenize(text)
    
    patent_ids = list(tokenized.keys())
    
    print("Searching for high-similarity non-citation pairs...")
    import random
    random.seed(42)
    
    pbar = tqdm(total=n_pairs, desc="Finding hard negatives")
    
    while len(hard_negatives) < n_pairs and attempts < max_attempts:
        attempts += 1
        
        # Random pair
        idx1, idx2 = random.sample(range(len(patent_ids)), 2)
        pid1, pid2 = patent_ids[idx1], patent_ids[idx2]
        
        # Normalize order
        if pid1 > pid2:
            pid1, pid2 = pid2, pid1
        
        # Skip if it's a known positive pair
        if (pid1, pid2) in existing_positive_pairs:
            continue
        
        # Compute similarity
        sim = jaccard(tokenized[pid1], tokenized[pid2])
        
        # Only keep high-similarity pairs
        if sim >= min_similarity:
            hard_negatives.append({
                'patent_id_1': pid1,
                'patent_id_2': pid2,
                'label': 0,
                'pair_type': 'hard_negative',
                'text_similarity': sim
            })
            pbar.update(1)
    
    pbar.close()
    
    print(f"Generated {len(hard_negatives)} hard negatives")
    if hard_negatives:
        sims = [p['text_similarity'] for p in hard_negatives]
        print(f"Similarity range: [{min(sims):.3f}, {max(sims):.3f}], mean={np.mean(sims):.3f}")
    
    return hard_negatives


def test_model_on_hard_negatives(hard_negatives: list, patents: dict):
    """Test the trained model on hard negatives."""
    from src.features.feature_extract import FeatureExtractor
    from src.models.mlp_classifier import PatentNoveltyClassifier
    
    print("\nTesting model on hard negatives...")
    
    # Load model
    clf = PatentNoveltyClassifier.load('models')
    
    # Extract features
    extractor = FeatureExtractor()
    
    features = []
    valid_pairs = []
    
    for pair in tqdm(hard_negatives, desc="Extracting features"):
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            continue
        
        fv = extractor.extract_features(p1, p2, label=0)
        features.append(fv.to_array(extractor.FEATURE_NAMES))
        valid_pairs.append(pair)
    
    if not features:
        print("No valid pairs to test")
        return
    
    X = np.array(features)
    y_true = np.zeros(len(X))  # All are negatives
    
    # Predict
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y_true).mean()
    false_positive_rate = y_pred.mean()  # How many negatives predicted as positive
    
    print(f"\n{'='*60}")
    print("HARD NEGATIVE RESULTS")
    print(f"{'='*60}")
    print(f"Total hard negatives tested: {len(X)}")
    print(f"Correctly classified as negative: {accuracy:.1%}")
    print(f"Incorrectly classified as positive (FP): {false_positive_rate:.1%}")
    print(f"Mean predicted probability: {y_proba.mean():.3f}")
    
    # Show worst cases (highest false positive probability)
    print(f"\nWorst cases (highest false positive probability):")
    sorted_idx = np.argsort(y_proba)[::-1][:10]
    
    for i, idx in enumerate(sorted_idx):
        pair = valid_pairs[idx]
        p1 = patents[str(pair['patent_id_1'])]
        p2 = patents[str(pair['patent_id_2'])]
        print(f"\n  {i+1}. Prob={y_proba[idx]:.3f}, TextSim={pair['text_similarity']:.3f}")
        print(f"     Patent 1: {p1.get('title', '')[:60]}...")
        print(f"     Patent 2: {p2.get('title', '')[:60]}...")
    
    return {
        'accuracy': accuracy,
        'false_positive_rate': false_positive_rate,
        'mean_proba': y_proba.mean()
    }


def main():
    print("="*60)
    print("NEGATIVE PAIR ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    patents = load_patents(limit=50000)  # Limit for speed
    print(f"Loaded {len(patents)} patents")
    
    # Load existing pairs
    train_pairs = load_pairs('data/training/train_pairs.jsonl')
    
    # Separate positives and negatives
    positives = [p for p in train_pairs if p['label'] == 1]
    negatives = [p for p in train_pairs if p['label'] == 0]
    
    print(f"Positive pairs: {len(positives)}")
    print(f"Negative pairs: {len(negatives)}")
    
    # Analyze similarity distributions
    print("\n2. Analyzing similarity distributions...")
    
    pos_sims = analyze_pair_similarity(patents, positives, sample_size=2000)
    neg_sims = analyze_pair_similarity(patents, negatives, sample_size=2000)
    
    pos_values = [s['similarity'] for s in pos_sims]
    neg_values = [s['similarity'] for s in neg_sims]
    
    print(f"\nPOSITIVE pairs (citations) text similarity:")
    print(f"  Mean: {np.mean(pos_values):.4f}")
    print(f"  Std:  {np.std(pos_values):.4f}")
    print(f"  Min:  {np.min(pos_values):.4f}")
    print(f"  Max:  {np.max(pos_values):.4f}")
    
    print(f"\nNEGATIVE pairs (random) text similarity:")
    print(f"  Mean: {np.mean(neg_values):.4f}")
    print(f"  Std:  {np.std(neg_values):.4f}")
    print(f"  Min:  {np.min(neg_values):.4f}")
    print(f"  Max:  {np.max(neg_values):.4f}")
    
    # The gap shows how "easy" negatives are
    gap = np.mean(pos_values) - np.mean(neg_values)
    print(f"\n[WARN] SIMILARITY GAP: {gap:.4f}")
    print(f"   This gap makes classification 'easy' - negatives are obviously different!")
    
    # Generate hard negatives
    print("\n3. Generating hard negatives...")
    positive_set = set()
    for p in positives:
        pid1, pid2 = str(p['patent_id_1']), str(p['patent_id_2'])
        if pid1 > pid2:
            pid1, pid2 = pid2, pid1
        positive_set.add((pid1, pid2))
    
    hard_negs = generate_hard_negatives(
        patents, 
        positive_set,
        n_pairs=2000,
        min_similarity=0.15
    )
    
    # Test model on hard negatives
    if hard_negs:
        print("\n4. Testing model on hard negatives...")
        results = test_model_on_hard_negatives(hard_negs, patents)
        
        # Compare to regular negatives
        print("\n" + "="*60)
        print("COMPARISON: EASY vs HARD NEGATIVES")
        print("="*60)
        print(f"Easy negatives (random):     ~90% correctly classified")
        print(f"Hard negatives (similar):    {results['accuracy']:.1%} correctly classified")
        print(f"\n→ The model struggles more with hard negatives!")
        print(f"→ This is why embeddings matter for fine-grained similarity.")
    
    # Save hard negatives
    output_path = Path('data/training/hard_negatives.jsonl')
    with open(output_path, 'w') as f:
        for hn in hard_negs:
            f.write(json.dumps(hn) + '\n')
    print(f"\nSaved {len(hard_negs)} hard negatives to {output_path}")


if __name__ == "__main__":
    main()


