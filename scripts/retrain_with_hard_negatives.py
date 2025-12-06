"""
Retrain MLP with hard negatives included in training data.

This teaches the model that high similarity doesn't always mean "related".
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_extractor import FeatureExtractor
from src.models.mlp_classifier import PatentNoveltyClassifier
from tqdm import tqdm


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


def load_hard_negatives(path: str = 'data/training/hard_negatives.jsonl'):
    """Load hard negative pairs."""
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    print("="*60)
    print("RETRAINING WITH HARD NEGATIVES")
    print("="*60)
    
    # Load existing features
    print("\n1. Loading existing features...")
    features_dir = Path('data/features')
    
    X_train = np.load(features_dir / 'train_features.X.npy')
    y_train = np.load(features_dir / 'train_features.y.npy')
    X_val = np.load(features_dir / 'val_features.X.npy')
    y_val = np.load(features_dir / 'val_features.y.npy')
    X_test = np.load(features_dir / 'test_features.X.npy')
    y_test = np.load(features_dir / 'test_features.y.npy')
    
    with open(features_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    print(f"Original training set: {len(X_train)} samples")
    print(f"  Positives: {y_train.sum():.0f} ({y_train.mean():.1%})")
    print(f"  Negatives: {(1-y_train).sum():.0f} ({(1-y_train.mean()):.1%})")
    
    # Load hard negatives
    print("\n2. Loading hard negatives...")
    hard_negs = load_hard_negatives()
    print(f"Hard negatives available: {len(hard_negs)}")
    
    # Load patents and embeddings for feature extraction
    print("\n3. Loading patents and embeddings...")
    patents = load_patents()
    
    embeddings = np.load('data/embeddings/patent_embeddings.npy')
    with open('data/embeddings/patent_ids.json', 'r') as f:
        patent_ids = json.load(f)
    
    # Setup feature extractor
    extractor = FeatureExtractor()
    extractor.set_embeddings(embeddings, patent_ids)
    
    # Extract features for hard negatives
    print("\n4. Extracting features for hard negatives...")
    hard_neg_features = []
    
    for pair in tqdm(hard_negs, desc="Processing hard negatives"):
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            continue
        
        fv = extractor.extract_features(p1, p2, label=0)
        hard_neg_features.append(fv.to_array(feature_names))
    
    X_hard = np.array(hard_neg_features)
    y_hard = np.zeros(len(X_hard))
    
    print(f"Hard negative features: {X_hard.shape}")
    
    # Check similarity distribution
    cos_idx = feature_names.index('cosine_doc_similarity')
    print(f"\nHard negative cosine similarities:")
    print(f"  Mean: {X_hard[:, cos_idx].mean():.3f}")
    print(f"  Min:  {X_hard[:, cos_idx].min():.3f}")
    print(f"  Max:  {X_hard[:, cos_idx].max():.3f}")
    
    # Split hard negatives: 70% train, 15% val, 15% test
    n = len(X_hard)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Augment training data
    print("\n5. Augmenting training data with hard negatives...")
    X_train_aug = np.vstack([X_train, X_hard[train_idx]])
    y_train_aug = np.hstack([y_train, y_hard[train_idx]])
    
    X_val_aug = np.vstack([X_val, X_hard[val_idx]])
    y_val_aug = np.hstack([y_val, y_hard[val_idx]])
    
    X_test_aug = np.vstack([X_test, X_hard[test_idx]])
    y_test_aug = np.hstack([y_test, y_hard[test_idx]])
    
    print(f"Augmented training set: {len(X_train_aug)} samples")
    print(f"  Positives: {y_train_aug.sum():.0f} ({y_train_aug.mean():.1%})")
    print(f"  Negatives: {(1-y_train_aug).sum():.0f} ({1-y_train_aug.mean():.1%})")
    
    # Train new model
    print("\n6. Training model with augmented data...")
    clf = PatentNoveltyClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20
    )
    
    clf.fit(X_train_aug, y_train_aug, X_val_aug, y_val_aug, feature_names)
    
    # Evaluate
    print("\n7. Evaluation...")
    
    # Original test set
    test_metrics = clf.evaluate(X_test, y_test)
    print(f"\nOriginal Test Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    
    # Hard negatives only
    y_hard_pred = clf.predict(X_hard)
    hard_accuracy = (y_hard_pred == 0).mean()  # Should all be 0
    print(f"\nHard Negatives Only:")
    print(f"  Correctly classified: {hard_accuracy:.1%}")
    print(f"  False positive rate:  {1-hard_accuracy:.1%}")
    
    # Augmented test set
    aug_metrics = clf.evaluate(X_test_aug, y_test_aug)
    print(f"\nAugmented Test Set:")
    print(f"  Accuracy:  {aug_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:   {aug_metrics['roc_auc']:.4f}")
    print(f"  F1:        {aug_metrics['f1']:.4f}")
    
    # Save model
    print("\n8. Saving improved model...")
    clf.save('models')
    
    # Save augmented features
    np.save(features_dir / 'train_features_aug.X.npy', X_train_aug)
    np.save(features_dir / 'train_features_aug.y.npy', y_train_aug)
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    
    print("\nCOMPARISON:")
    print(f"{'Metric':<30} {'Before':<15} {'After':<15}")
    print("-"*60)
    print(f"{'Hard negative accuracy':<30} {'0.0%':<15} {hard_accuracy:.1%}")
    print(f"{'Original test accuracy':<30} {'87.4%':<15} {test_metrics['accuracy']:.1%}")


if __name__ == "__main__":
    np.random.seed(42)
    main()


