"""
OPTIMIZED: Add claim-level embedding features.

Strategy:
1. Pre-compute claim embeddings for all patents in training pairs (batch)
2. Store in dict for O(1) lookup
3. Compute similarities quickly
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.models.mlp_classifier import PatentNoveltyClassifier


def load_patents(path: str = 'data/sampled/patents_sampled.jsonl'):
    """Load patents."""
    patents = {}
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading patents"):
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    return patents


def get_unique_patent_ids(splits=['train', 'val', 'test']):
    """Get all unique patent IDs from training pairs."""
    unique_ids = set()
    for split in splits:
        path = f'data/training/{split}_pairs.jsonl'
        with open(path, 'r') as f:
            for line in f:
                pair = json.loads(line)
                unique_ids.add(str(pair['patent_id_1']))
                unique_ids.add(str(pair['patent_id_2']))
    return unique_ids


def extract_first_claim(patent: dict) -> str:
    """Extract first/main claim text."""
    # Try independent claims first
    ind_claims = patent.get('independent_claims', [])
    if ind_claims:
        c = ind_claims[0]
        if isinstance(c, dict):
            return c.get('text', '')[:500]
        return str(c)[:500]
    
    # Fall back to first claim
    claims = patent.get('claims', [])
    if claims:
        c = claims[0]
        if isinstance(c, dict):
            return c.get('text', '')[:500]
        return str(c)[:500]
    
    return ""


def precompute_claim_embeddings(patents: dict, patent_ids: set, model):
    """Pre-compute claim embeddings for all needed patents."""
    print(f"\nPre-computing claim embeddings for {len(patent_ids)} patents...")
    
    # Collect claim texts
    texts = []
    ids = []
    
    for pid in tqdm(patent_ids, desc="Extracting claims"):
        patent = patents.get(pid)
        if patent:
            claim_text = extract_first_claim(patent)
            if claim_text:
                texts.append(claim_text)
                ids.append(pid)
    
    print(f"Encoding {len(texts)} claim texts...")
    
    # Batch encode
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Create lookup dict
    claim_emb_dict = {pid: emb for pid, emb in zip(ids, embeddings)}
    
    print(f"Created embeddings for {len(claim_emb_dict)} patents")
    return claim_emb_dict


def compute_claim_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def main():
    print("="*60)
    print("FAST CLAIM FEATURE COMPUTATION")
    print("="*60)
    
    # Load patents
    patents = load_patents()
    
    # Get unique patent IDs
    unique_ids = get_unique_patent_ids()
    print(f"Unique patents in pairs: {len(unique_ids)}")
    
    # Also add hard negatives
    hard_negs = []
    with open('data/training/hard_negatives.jsonl', 'r') as f:
        for line in f:
            pair = json.loads(line)
            hard_negs.append(pair)
            unique_ids.add(str(pair['patent_id_1']))
            unique_ids.add(str(pair['patent_id_2']))
    print(f"After adding hard negatives: {len(unique_ids)}")
    
    # Load model
    print("\nLoading PatentSBERTa model...")
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    # Pre-compute claim embeddings
    claim_emb_dict = precompute_claim_embeddings(patents, unique_ids, model)
    
    # Now process each split
    features_dir = Path('data/features')
    
    with open(features_dir / 'feature_names.json', 'r') as f:
        old_names = json.load(f)
    
    new_feature_name = 'claim_similarity'  # Single claim feature for speed
    all_names = old_names + [new_feature_name]
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*40}")
        print(f"Processing {split}...")
        
        # Load existing features
        X_old = np.load(features_dir / f'{split}_features.X.npy')
        y = np.load(features_dir / f'{split}_features.y.npy')
        
        # Load pairs
        pairs = []
        with open(f'data/training/{split}_pairs.jsonl', 'r') as f:
            for line in f:
                pairs.append(json.loads(line))
        
        # Compute claim similarities
        claim_sims = []
        for pair in tqdm(pairs, desc="Computing similarities"):
            pid1 = str(pair['patent_id_1'])
            pid2 = str(pair['patent_id_2'])
            
            emb1 = claim_emb_dict.get(pid1)
            emb2 = claim_emb_dict.get(pid2)
            
            sim = compute_claim_similarity(emb1, emb2)
            claim_sims.append(sim)
        
        claim_features = np.array(claim_sims).reshape(-1, 1)
        
        # Combine
        X_new = np.hstack([X_old, claim_features])
        
        print(f"Old shape: {X_old.shape}, New shape: {X_new.shape}")
        print(f"Claim sim: mean={np.mean(claim_sims):.3f}, std={np.std(claim_sims):.3f}")
        
        # Save
        np.save(features_dir / f'{split}_features_v2.X.npy', X_new)
        np.save(features_dir / f'{split}_features_v2.y.npy', y)
    
    # Save feature names
    with open(features_dir / 'feature_names_v2.json', 'w') as f:
        json.dump(all_names, f, indent=2)
    
    # Process hard negatives
    print(f"\n{'='*40}")
    print("Processing hard negatives...")
    
    # Load doc embeddings for other features
    embeddings = np.load('data/embeddings/patent_embeddings.npy')
    with open('data/embeddings/patent_ids.json', 'r') as f:
        patent_ids = json.load(f)
    
    pid_to_idx = {pid: i for i, pid in enumerate(patent_ids)}
    
    from src.features.feature_extract import FeatureExtractor
    extractor = FeatureExtractor()
    extractor.set_embeddings(embeddings, patent_ids)
    
    hard_features = []
    for pair in tqdm(hard_negs, desc="Hard negative features"):
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            hard_features.append(np.zeros(len(all_names)))
            continue
        
        # Old features
        fv = extractor.extract_features(p1, p2, label=0)
        old_feats = fv.to_array(old_names)
        
        # Claim similarity
        pid1, pid2 = str(pair['patent_id_1']), str(pair['patent_id_2'])
        emb1 = claim_emb_dict.get(pid1)
        emb2 = claim_emb_dict.get(pid2)
        claim_sim = compute_claim_similarity(emb1, emb2)
        
        hard_features.append(np.concatenate([old_feats, [claim_sim]]))
    
    X_hard = np.array(hard_features)
    
    # Train improved model
    print(f"\n{'='*60}")
    print("TRAINING IMPROVED MODEL")
    print("="*60)
    
    X_train = np.load(features_dir / 'train_features_v2.X.npy')
    y_train = np.load(features_dir / 'train_features_v2.y.npy')
    X_val = np.load(features_dir / 'val_features_v2.X.npy')
    y_val = np.load(features_dir / 'val_features_v2.y.npy')
    X_test = np.load(features_dir / 'test_features_v2.X.npy')
    y_test = np.load(features_dir / 'test_features_v2.y.npy')
    
    print(f"Features: {X_train.shape[1]} (was 12, now +1 claim)")
    
    # Add hard negatives to training
    n_hard = len(X_hard)
    train_hard_idx = int(n_hard * 0.7)
    
    X_train_aug = np.vstack([X_train, X_hard[:train_hard_idx]])
    y_train_aug = np.hstack([y_train, np.zeros(train_hard_idx)])
    
    print(f"Training set: {len(X_train_aug)} (including {train_hard_idx} hard negatives)")
    
    clf = PatentNoveltyClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20
    )
    
    clf.fit(X_train_aug, y_train_aug, X_val, y_val, all_names)
    
    # Evaluate
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    test_metrics = clf.evaluate(X_test, y_test)
    print(f"\nStandard Test Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    
    # Hard negatives
    y_hard_pred = clf.predict(X_hard)
    hard_acc = (y_hard_pred == 0).mean()
    
    print(f"\nHard Negatives:")
    print(f"  Correctly classified: {hard_acc:.1%}")
    
    # Feature importance
    print("\nTop Features:")
    importance = clf.get_feature_importance()
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(imp * 40)
        print(f"  {name:30s} {imp:.3f} {bar}")
    
    # Save
    clf.save('models')
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()


