"""
Compute features for training pairs.

This script:
1. Loads training pairs (from citation-based pairs)
2. Loads patent data
3. Loads embeddings (if available)
4. Computes all features
5. Saves feature matrices for MLP training
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional

from src.features.feature_extract import FeatureExtractor, FeatureVector


def load_patents_dict(
    sampled_path: str = 'data/sampled/patents_sampled.jsonl'
) -> Dict[str, dict]:
    """Load patents into a dictionary keyed by patent_id."""
    print(f"Loading patents from {sampled_path}...")
    patents = {}
    
    with open(sampled_path, 'r') as f:
        for line in tqdm(f, desc="Loading patents"):
            patent = json.loads(line)
            patents[str(patent['patent_id'])] = patent
    
    print(f"Loaded {len(patents)} patents")
    return patents


def load_pairs(
    pairs_path: str
) -> List[dict]:
    """Load training pairs from JSONL."""
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_embeddings(
    embeddings_dir: str = 'data/embeddings'
) -> tuple:
    """Load pre-computed embeddings if available."""
    embeddings_path = Path(embeddings_dir) / 'patent_embeddings.npy'
    
    # Try JSON first, then NPY for IDs
    ids_path_json = Path(embeddings_dir) / 'patent_ids.json'
    ids_path_npy = Path(embeddings_dir) / 'patent_ids.npy'
    
    if not embeddings_path.exists():
        print("[WARN] Embeddings not found. Will compute text-based features only.")
        return None, None
    
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    
    # Load patent IDs
    if ids_path_json.exists():
        with open(ids_path_json, 'r') as f:
            patent_ids = json.load(f)
        print(f"Loaded {len(patent_ids)} embeddings with dim {embeddings.shape[1]} (from JSON)")
    elif ids_path_npy.exists():
        patent_ids = np.load(ids_path_npy, allow_pickle=True).tolist()
        print(f"Loaded {len(patent_ids)} embeddings with dim {embeddings.shape[1]} (from NPY)")
    else:
        print("[WARN] Patent IDs not found. Will compute text-based features only.")
        return None, None
    
    return embeddings, patent_ids


def compute_features_for_split(
    pairs: List[dict],
    patents: Dict[str, dict],
    extractor: FeatureExtractor,
    split_name: str
) -> List[FeatureVector]:
    """Compute features for a set of pairs."""
    print(f"\nComputing features for {split_name} ({len(pairs)} pairs)...")
    
    feature_vectors = []
    missing_patents = 0
    
    for pair in tqdm(pairs, desc=f"Processing {split_name}"):
        pid1 = str(pair['patent_id_1'])
        pid2 = str(pair['patent_id_2'])
        label = pair.get('label', 0)
        
        patent1 = patents.get(pid1)
        patent2 = patents.get(pid2)
        
        if patent1 is None or patent2 is None:
            missing_patents += 1
            continue
        
        fv = extractor.extract_features(patent1, patent2, label=label)
        feature_vectors.append(fv)
    
    if missing_patents > 0:
        print(f"  [WARN] Skipped {missing_patents} pairs due to missing patents")
    
    print(f"  [OK] Computed {len(feature_vectors)} feature vectors")
    return feature_vectors


def save_features(
    feature_vectors: List[FeatureVector],
    output_path: str,
    feature_names: List[str]
):
    """Save features to numpy files."""
    output_path = Path(output_path)
    
    # Convert to numpy
    X = np.array([fv.to_array(feature_names) for fv in feature_vectors])
    y = np.array([fv.label for fv in feature_vectors])
    pair_ids = [(fv.patent_id_1, fv.patent_id_2) for fv in feature_vectors]
    
    # Save
    np.save(output_path.with_suffix('.X.npy'), X)
    np.save(output_path.with_suffix('.y.npy'), y)
    
    with open(output_path.with_suffix('.pairs.json'), 'w') as f:
        json.dump(pair_ids, f)
    
    print(f"Saved features to {output_path}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive rate: {y.mean():.2%}")


def main():
    print("="*60)
    print("FEATURE COMPUTATION PIPELINE")
    print("="*60)
    
    # Create output directory
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    patents = load_patents_dict()
    
    # Load embeddings (if available)
    embeddings, embedding_ids = load_embeddings()
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    if embeddings is not None:
        extractor.set_embeddings(embeddings, embedding_ids)
        print("[OK] Embeddings loaded into feature extractor")
    else:
        print("[WARN] Running without embeddings - embedding features will be 0")
    
    # Load and process each split
    splits = ['train', 'val', 'test']
    all_feature_vectors = {}
    
    for split in splits:
        pairs_path = f'data/training/{split}_pairs.jsonl'
        
        if not Path(pairs_path).exists():
            print(f"[WARN] {pairs_path} not found, skipping...")
            continue
        
        pairs = load_pairs(pairs_path)
        feature_vectors = compute_features_for_split(
            pairs, patents, extractor, split
        )
        all_feature_vectors[split] = feature_vectors
        
        # Save
        save_features(
            feature_vectors,
            output_dir / f'{split}_features',
            extractor.FEATURE_NAMES
        )
    
    # Save feature names for reference
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(extractor.FEATURE_NAMES, f, indent=2)
    
    print("\n" + "="*60)
    print("FEATURE COMPUTATION COMPLETE")
    print("="*60)
    
    # Summary statistics
    for split, fvs in all_feature_vectors.items():
        if not fvs:
            continue
        
        X, y, _ = extractor.to_numpy(fvs)
        print(f"\n{split.upper()} set:")
        print(f"  Samples: {len(fvs)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive rate: {y.mean():.2%}")
        
        # Feature statistics
        print(f"  Feature ranges:")
        for i, name in enumerate(extractor.FEATURE_NAMES):
            col = X[:, i]
            print(f"    {name}: [{col.min():.3f}, {col.max():.3f}] mean={col.mean():.3f}")


if __name__ == "__main__":
    main()

