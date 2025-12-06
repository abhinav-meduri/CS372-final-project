"""
Train MLP classifier on patent pair features.

This script:
1. Loads computed features
2. Trains an MLP classifier
3. Evaluates on validation and test sets
4. Saves model, metrics, and plots
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mlp_classifier import PatentNoveltyClassifier


def load_features(features_dir: str = 'data/features'):
    """Load pre-computed features."""
    features_dir = Path(features_dir)
    
    data = {}
    for split in ['train', 'val', 'test']:
        X = np.load(features_dir / f'{split}_features.X.npy')
        y = np.load(features_dir / f'{split}_features.y.npy')
        data[split] = {'X': X, 'y': y}
        print(f"Loaded {split}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Load feature names
    with open(features_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return data, feature_names


def train_and_evaluate():
    """Main training pipeline."""
    print("="*60)
    print("MLP CLASSIFIER TRAINING")
    print("="*60)
    
    # Create output directories
    models_dir = Path('models')
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    metrics_dir = results_dir / 'metrics'
    
    for d in [models_dir, plots_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print("\n1. Loading features...")
    data, feature_names = load_features()
    
    X_train, y_train = data['train']['X'], data['train']['y']
    X_val, y_val = data['val']['X'], data['val']['y']
    X_test, y_test = data['test']['X'], data['test']['y']
    
    print(f"\nFeature names: {feature_names}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Check for zero-variance features (all zeros)
    print("\n2. Feature analysis...")
    for i, name in enumerate(feature_names):
        train_col = X_train[:, i]
        if train_col.std() == 0:
            print(f"  ⚠️ {name}: zero variance (all {train_col[0]:.2f})")
        else:
            print(f"  ✓ {name}: mean={train_col.mean():.3f}, std={train_col.std():.3f}")
    
    # Train model
    print("\n3. Training MLP classifier...")
    clf = PatentNoveltyClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20
    )
    
    clf.fit(X_train, y_train, X_val, y_val, feature_names)
    
    # Evaluate
    print("\n4. Evaluation...")
    
    # Training metrics
    train_metrics = clf.evaluate(X_train, y_train)
    print(f"\nTraining Metrics:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1:        {train_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
    
    # Validation metrics
    val_metrics = clf.evaluate(X_val, y_val)
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
    
    # Test metrics
    test_metrics = clf.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Brier:     {test_metrics['brier_score']:.4f}")
    
    # Feature importance
    print("\n5. Feature Importance:")
    importance = clf.get_feature_importance()
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:30s} {imp:.4f} {bar}")
    
    # Save model
    print("\n6. Saving model and results...")
    clf.save(models_dir)
    
    # Save plots
    clf.plot_training_curve(plots_dir / 'training_curve.png')
    clf.plot_roc_curve(X_test, y_test, plots_dir / 'roc_curve.png')
    clf.plot_confusion_matrix(X_test, y_test, plots_dir / 'confusion_matrix.png')
    
    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'feature_importance': importance
    }
    
    with open(metrics_dir / 'all_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel saved to: {models_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Metrics saved to: {metrics_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    
    # Note about missing features
    zero_var_features = [name for i, name in enumerate(feature_names) 
                         if X_train[:, i].std() == 0]
    if zero_var_features:
        print(f"\n⚠️ Note: {len(zero_var_features)} features have zero variance.")
        print("   These will be populated once embeddings are ready.")
        print("   Re-run this script after embedding generation completes.")


if __name__ == "__main__":
    train_and_evaluate()


