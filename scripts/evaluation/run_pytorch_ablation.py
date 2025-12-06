"""
Ablation Study for PyTorch Neural Network (13 features)

Re-runs ablation study on the trained PyTorch model to measure
impact of removing feature groups.
"""

import numpy as np
import json
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pytorch_classifier import PyTorchPatentClassifier


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


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }


def run_ablation_study():
    """Run ablation study on PyTorch model."""
    print("="*70)
    print("PYTORCH MODEL ABLATION STUDY (13 Features)")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_features()
    
    print(f"\nLoaded {len(feature_names)} features: {feature_names}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    feature_groups = {
        'embedding': ['cosine_doc_similarity', 'cosine_max_claim_similarity', 
                      'embedding_diff_mean', 'embedding_diff_std'],
        'text_similarity': ['title_jaccard', 'shared_rare_terms_ratio'],
        'metadata': ['year_diff', 'claim_count_ratio', 'abstract_length_ratio'],
        'bm25': ['bm25_doc_score', 'bm25_best_claim_score'],
        'cpc': ['cpc_jaccard'],
        'claim': ['claim_similarity']
    }
    
    group_indices = {}
    for group_name, features in feature_groups.items():
        indices = [feature_names.index(f) for f in features if f in feature_names]
        group_indices[group_name] = indices
        print(f"  {group_name}: {len(indices)} features {indices}")
    
    results = {}
    
    print("\n1. Training full model (all 13 features)...")
    full_model = PyTorchPatentClassifier(
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        max_epochs=50,
        patience=10,
        batch_size=256
    )
    full_model.fit(X_train, y_train, X_val, y_val, feature_names, use_mixup=True)
    full_metrics = evaluate_model(full_model, X_test, y_test)
    results['All Features'] = full_metrics
    print(f"   Accuracy: {full_metrics['accuracy']:.4f}, ROC-AUC: {full_metrics['roc_auc']:.4f}")
    
    for group_name, indices in group_indices.items():
        if not indices:
            continue
        
        print(f"\n2. Training without {group_name} features ({len(indices)} features)...")
        
        keep_indices = [i for i in range(len(feature_names)) if i not in indices]
        ablated_names = [feature_names[i] for i in keep_indices]
        
        X_train_ablated = X_train[:, keep_indices]
        X_val_ablated = X_val[:, keep_indices]
        X_test_ablated = X_test[:, keep_indices]
        
        ablated_model = PyTorchPatentClassifier(
            hidden_dims=[128, 64, 32],
            dropout=0.3,
            learning_rate=0.001,
            max_epochs=50,
            patience=10,
            batch_size=256
        )
        ablated_model.fit(X_train_ablated, y_train, X_val_ablated, y_val, ablated_names, use_mixup=True)
        ablated_metrics = evaluate_model(ablated_model, X_test_ablated, y_test)
        results[f'Without {group_name}'] = ablated_metrics
        print(f"   Accuracy: {ablated_metrics['accuracy']:.4f}, ROC-AUC: {ablated_metrics['roc_auc']:.4f}")
    
    print("\n" + "-"*70)
    print(f"{'Configuration':<30} {'Accuracy':<12} {'ROC-AUC':<12} {'Δ AUC':<12}")
    print("-"*70)
    
    full_auc = results['All Features']['roc_auc']
    
    for config, metrics in results.items():
        acc = f"{metrics['accuracy']:.1%}"
        auc = f"{metrics['roc_auc']:.4f}"
        
        if config == 'All Features':
            delta = "--"
        else:
            delta_val = metrics['roc_auc'] - full_auc
            delta = f"{delta_val:+.4f}"
        
        print(f"{config:<30} {acc:<12} {auc:<12} {delta:<12}")
    
    print("-"*70)
    
    impacts = {}
    for config, metrics in results.items():
        if config != 'All Features':
            impacts[config] = full_auc - metrics['roc_auc']
    
    if impacts:
        most_impactful = max(impacts, key=impacts.get)
        print(f"\nMost impactful feature group: {most_impactful}")
        print(f"  Removing it decreases ROC-AUC by {impacts[most_impactful]:.4f}")
    
    output_path = Path('results/comprehensive_evaluation/pytorch_ablation_study.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_ablation_study()

