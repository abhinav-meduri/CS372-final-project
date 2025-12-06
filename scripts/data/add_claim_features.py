"""
Add claim-level embedding features and retrain the model.

This adds 3 new features:
1. max_claim_similarity - highest similarity between any two claims
2. mean_claim_similarity - average claim pair similarity  
3. independent_claim_similarity - similarity of main claims
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.claim_embeddings import ClaimEmbedder
from src.models.mlp_classifier import PatentNoveltyClassifier


def load_patents(path: str = 'data/sampled/patents_sampled.jsonl'):
    """Load all patents."""
    patents = {}
    print(f"Loading patents from {path}...")
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading"):
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    print(f"Loaded {len(patents)} patents")
    return patents


def load_pairs(split: str):
    """Load pairs for a split."""
    path = f'data/training/{split}_pairs.jsonl'
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def compute_claim_features_for_pairs(
    pairs: list,
    patents: dict,
    embedder: ClaimEmbedder,
    batch_size: int = 100
) -> np.ndarray:
    """Compute claim features for a list of pairs."""
    features = []
    
    for pair in tqdm(pairs, desc="Computing claim features"):
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            # Missing patent - use zeros
            features.append([0.0, 0.0, 0.0])
            continue
        
        try:
            sims = embedder.compute_claim_similarity(p1, p2)
            features.append([
                sims['max_claim_similarity'],
                sims['mean_claim_similarity'],
                sims['independent_claim_similarity']
            ])
        except Exception as e:
            # On error, use zeros
            features.append([0.0, 0.0, 0.0])
    
    return np.array(features)


def main():
    print("="*60)
    print("ADDING CLAIM-LEVEL FEATURES")
    print("="*60)
    
    # Load data
    patents = load_patents()
    
    # Initialize claim embedder
    print("\nInitializing claim embedder...")
    embedder = ClaimEmbedder(batch_size=32)
    
    # Load existing features
    features_dir = Path('data/features')
    
    new_feature_names = [
        'max_claim_similarity',
        'mean_claim_similarity', 
        'independent_claim_similarity'
    ]
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*40}")
        print(f"Processing {split} split...")
        print(f"{'='*40}")
        
        # Load existing features
        X_old = np.load(features_dir / f'{split}_features.X.npy')
        y = np.load(features_dir / f'{split}_features.y.npy')
        
        print(f"Existing features shape: {X_old.shape}")
        
        # Load pairs
        pairs = load_pairs(split)
        print(f"Pairs: {len(pairs)}")
        
        # Compute claim features (sample for speed if too large)
        if len(pairs) > 10000:
            print(f"Large split - computing claim features for sample first...")
            # For training, compute all. For val/test, can sample if needed
        
        claim_features = compute_claim_features_for_pairs(pairs, patents, embedder)
        print(f"Claim features shape: {claim_features.shape}")
        
        # Combine features
        X_new = np.hstack([X_old, claim_features])
        print(f"Combined features shape: {X_new.shape}")
        
        # Save
        np.save(features_dir / f'{split}_features_v2.X.npy', X_new)
        np.save(features_dir / f'{split}_features_v2.y.npy', y)
        
        # Show claim feature statistics
        print(f"\nClaim feature statistics for {split}:")
        for i, name in enumerate(new_feature_names):
            col = claim_features[:, i]
            print(f"  {name}: mean={col.mean():.3f}, std={col.std():.3f}, range=[{col.min():.3f}, {col.max():.3f}]")
    
    # Update feature names
    with open(features_dir / 'feature_names.json', 'r') as f:
        old_names = json.load(f)
    
    all_names = old_names + new_feature_names
    with open(features_dir / 'feature_names_v2.json', 'w') as f:
        json.dump(all_names, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CLAIM FEATURES COMPUTED")
    print(f"{'='*60}")
    print(f"Total features: {len(all_names)}")
    print(f"New features: {new_feature_names}")
    
    # Train new model
    print(f"\n{'='*60}")
    print("TRAINING MODEL WITH CLAIM FEATURES")
    print(f"{'='*60}")
    
    X_train = np.load(features_dir / 'train_features_v2.X.npy')
    y_train = np.load(features_dir / 'train_features_v2.y.npy')
    X_val = np.load(features_dir / 'val_features_v2.X.npy')
    y_val = np.load(features_dir / 'val_features_v2.y.npy')
    X_test = np.load(features_dir / 'test_features_v2.X.npy')
    y_test = np.load(features_dir / 'test_features_v2.y.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train
    clf = PatentNoveltyClassifier(
        hidden_layer_sizes=(128, 64),  # Larger network for more features
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20
    )
    
    clf.fit(X_train, y_train, X_val, y_val, all_names)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    test_metrics = clf.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = clf.get_feature_importance()
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(imp * 50)
        print(f"  {name:35s} {imp:.4f} {bar}")
    
    # Test on hard negatives
    print("\n" + "="*60)
    print("HARD NEGATIVE EVALUATION")
    print("="*60)
    
    # Load hard negatives and compute their features
    hard_negs = []
    with open('data/training/hard_negatives.jsonl', 'r') as f:
        for line in f:
            hard_negs.append(json.loads(line))
    
    print(f"Computing features for {len(hard_negs)} hard negatives...")
    
    # Load old features for hard negs and add claim features
    from src.features.feature_extractor import FeatureExtractor
    
    # Load embeddings
    embeddings = np.load('data/embeddings/patent_embeddings.npy')
    with open('data/embeddings/patent_ids.json', 'r') as f:
        patent_ids = json.load(f)
    
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
        
        # Claim features
        try:
            sims = embedder.compute_claim_similarity(p1, p2)
            claim_feats = [
                sims['max_claim_similarity'],
                sims['mean_claim_similarity'],
                sims['independent_claim_similarity']
            ]
        except:
            claim_feats = [0.0, 0.0, 0.0]
        
        hard_features.append(np.concatenate([old_feats, claim_feats]))
    
    X_hard = np.array(hard_features)
    y_hard = np.zeros(len(X_hard))
    
    # Evaluate on hard negatives
    y_hard_pred = clf.predict(X_hard)
    hard_accuracy = (y_hard_pred == 0).mean()
    
    print(f"\nHard Negatives:")
    print(f"  Correctly classified: {hard_accuracy:.1%}")
    print(f"  False positive rate:  {1-hard_accuracy:.1%}")
    
    # Save model
    clf.save('models')
    print("\nModel saved to models/")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Features: {len(old_names)} → {len(all_names)} (+3 claim features)")
    print(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Hard Negative Accuracy: {hard_accuracy:.1%}")


if __name__ == "__main__":
    main()


