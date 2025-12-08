"""
Ablation Study for MLP and PyTorch Models

Runs ablation study on both models to measure impact of removing feature groups.
Uses current 10 features (BM25 and CPC already removed).
"""

import numpy as np
import json
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mlp_classifier import PatentNoveltyClassifier
from src.app.pytorch_classifier import PyTorchPatentClassifier


def load_features(features_dir: str = 'data/features'):
    """Load pre-computed features."""
    features_dir = Path(features_dir)
    
    X_train = np.load(features_dir / 'train_features_v2.X.npy')
    y_train = np.load(features_dir / 'train_features_v2.y.npy')
    X_val = np.load(features_dir / 'val_features_v2.X.npy')
    y_val = np.load(features_dir / 'val_features_v2.y.npy')
    X_test = np.load(features_dir / 'test_features_v2.X.npy')
    y_test = np.load(features_dir / 'test_features_v2.y.npy')
    
    with open(features_dir / 'feature_names_v2.json', 'r') as f:
        feature_names = json.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def evaluate_model(model, X_test, y_test, is_pytorch=False):
    """Evaluate model and return metrics."""
    if is_pytorch:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    }


def run_ablation_for_model(model_name, model_class, model_kwargs, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Run ablation study for a specific model."""
    print(f"{model_name.upper()} MODEL ABLATION STUDY")
    
    print(f"\nLoaded {len(feature_names)} features: {feature_names}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Define feature groups based on current 10 features
    feature_groups = {
        'embedding': ['cosine_doc_similarity', 'cosine_max_claim_similarity', 
                      'embedding_diff_mean', 'embedding_diff_std'],
        'text_similarity': ['title_jaccard', 'shared_rare_terms_ratio'],
        'metadata': ['year_diff', 'claim_count_ratio', 'abstract_length_ratio'],
        'claim': ['claim_similarity']
    }
    
    group_indices = {}
    for group_name, features in feature_groups.items():
        indices = [feature_names.index(f) for f in features if f in feature_names]
        if indices:
            group_indices[group_name] = indices
            print(f"  {group_name}: {len(indices)} features at indices {indices}")
    
    results = {}
    is_pytorch = 'PyTorch' in model_name
    
    # Train full model
    print(f"\n1. Training full {model_name} model (all {len(feature_names)} features)...")
    full_model = model_class(**model_kwargs)
    full_model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    full_metrics = evaluate_model(full_model, X_test, y_test, is_pytorch=is_pytorch)
    results['All Features'] = full_metrics
    print(f"   Accuracy: {full_metrics['accuracy']:.4f}, ROC-AUC: {full_metrics['roc_auc']:.4f}, F1: {full_metrics['f1']:.4f}")
    
    # Ablation: remove each feature group
    for group_name, indices in group_indices.items():
        if not indices:
            continue
        
        print(f"\n2. Training without {group_name} features ({len(indices)} features removed)...")
        
        keep_indices = [i for i in range(len(feature_names)) if i not in indices]
        ablated_names = [feature_names[i] for i in keep_indices]
        
        X_train_ablated = X_train[:, keep_indices]
        X_val_ablated = X_val[:, keep_indices]
        X_test_ablated = X_test[:, keep_indices]
        
        ablated_model = model_class(**model_kwargs)
        ablated_model.fit(X_train_ablated, y_train, X_val_ablated, y_val, feature_names=ablated_names)
        ablated_metrics = evaluate_model(ablated_model, X_test_ablated, y_test, is_pytorch=is_pytorch)
        results[f'Without {group_name}'] = ablated_metrics
        print(f"   Accuracy: {ablated_metrics['accuracy']:.4f}, ROC-AUC: {ablated_metrics['roc_auc']:.4f}, F1: {ablated_metrics['f1']:.4f}")
    
    # Print summary table
    print(f"\n{'-'*70}")
    print(f"{'Configuration':<30} {'Accuracy':<12} {'ROC-AUC':<12} {'F1':<12} {'Δ AUC':<12}")
    print("-"*70)
    
    full_auc = results['All Features']['roc_auc']
    
    for config, metrics in results.items():
        acc = f"{metrics['accuracy']:.4f}"
        auc = f"{metrics['roc_auc']:.4f}"
        f1 = f"{metrics['f1']:.4f}"
        
        if config == 'All Features':
            delta = "--"
        else:
            delta_val = metrics['roc_auc'] - full_auc
            delta = f"{delta_val:+.4f}"
        
        print(f"{config:<30} {acc:<12} {auc:<12} {f1:<12} {delta:<12}")
    
    print("-"*70)
    
    # Find most impactful feature group
    impacts = {}
    for config, metrics in results.items():
        if config != 'All Features':
            impacts[config] = full_auc - metrics['roc_auc']
    
    if impacts:
        most_impactful = max(impacts, key=impacts.get)
        least_impactful = min(impacts, key=impacts.get)
        print(f"\nMost impactful feature group: {most_impactful}")
        print(f"  Removing it decreases ROC-AUC by {impacts[most_impactful]:.4f}")
        print(f"\nLeast impactful feature group: {least_impactful}")
        print(f"  Removing it decreases ROC-AUC by {impacts[least_impactful]:.4f}")
    
    return results


def main():
    print("ABLATION STUDY FOR MLP AND PYTORCH MODELS")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_features()
    
    # MLP model kwargs (using best hyperparameters from tuning)
    mlp_kwargs = {
        'hidden_layer_sizes': (64,),
        'alpha': 1e-5,
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'early_stopping': True,
        'n_iter_no_change': 20,
        'random_state': 42
    }
    
    # PyTorch model kwargs
    pytorch_kwargs = {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'max_epochs': 100,
        'patience': 15,
        'use_residual': True,
        'bn_momentum': 0.1
    }
    
    # Run ablation for MLP
    mlp_results = run_ablation_for_model(
        'MLP',
        PatentNoveltyClassifier,
        mlp_kwargs,
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Run ablation for PyTorch
    pytorch_results = run_ablation_for_model(
        'PyTorch',
        PyTorchPatentClassifier,
        pytorch_kwargs,
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Save results
    output_dir = Path('results/ablation_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'mlp': mlp_results,
        'pytorch': pytorch_results,
        'feature_names': feature_names
    }
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("ABLATION STUDY COMPLETE")
    print(f"\nResults saved to: {output_dir / 'ablation_results.json'}")
    
    # Print comparison
    print("COMPARISON: MLP vs PyTorch")
    print(f"\n{'Model':<20} {'Full Accuracy':<15} {'Full ROC-AUC':<15} {'Full F1':<15}")
    print("-"*70)
    print(f"{'MLP':<20} {mlp_results['All Features']['accuracy']:.4f}          {mlp_results['All Features']['roc_auc']:.4f}          {mlp_results['All Features']['f1']:.4f}")
    print(f"{'PyTorch':<20} {pytorch_results['All Features']['accuracy']:.4f}          {pytorch_results['All Features']['roc_auc']:.4f}          {pytorch_results['All Features']['f1']:.4f}")
    print("-"*70)


if __name__ == "__main__":
    main()

