"""
Edge Case Analysis for Patent Novelty Assessment

Analyzes model performance on:
1. Hard negatives (high similarity but different semantics)
2. Easy cases (low similarity, clear labels)
3. Boundary cases (similarity ~0.5)

Generates visualizations and examples for rubric documentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from collections import defaultdict
import pandas as pd

from src.models.pytorch_classifier import PyTorchPatentClassifier
from src.features.feature_extract import FeatureExtractor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_model_and_data():
    """Load trained model and test data."""
    print("Loading model and data...")
    
    # Load model
    model = PyTorchPatentClassifier()
    model.load('models/pytorch_nn')
    
    # Load test data
    features_dir = Path("data/features")
    X_test = np.load(features_dir / "test_features_v2.X.npy")
    y_test = np.load(features_dir / "test_features_v2.y.npy")
    
    # Load feature names and remove BM25/CPC features
    with open(features_dir / "feature_names_v2.json", "r") as f:
        feature_names = json.load(f)
    
    # If we have 13 features, remove BM25 and CPC (indices 0, 1, 6)
    if X_test.shape[1] == 13:
        old_names = [
            'bm25_doc_score', 'bm25_best_claim_score', 'cosine_doc_similarity',
            'cosine_max_claim_similarity', 'embedding_diff_mean', 'embedding_diff_std',
            'cpc_jaccard', 'year_diff', 'title_jaccard', 'abstract_length_ratio',
            'claim_count_ratio', 'shared_rare_terms_ratio', 'claim_similarity'
        ]
        indices_to_remove = [0, 1, 6]
        indices_to_keep = [i for i in range(13) if i not in indices_to_remove]
        X_test = X_test[:, indices_to_keep]
    
    return model, X_test, y_test, feature_names

def extract_similarity_scores(X_test, feature_names):
    """Extract cosine similarity scores from features."""
    # Find cosine similarity feature index
    sim_idx = None
    for i, name in enumerate(feature_names):
        if 'cosine' in name.lower() and 'similarity' in name.lower():
            sim_idx = i
            break
    
    if sim_idx is None:
        # Fallback: use first feature (usually similarity)
        sim_idx = 0
    
    similarities = X_test[:, sim_idx]
    return similarities

def categorize_cases(similarities, y_true, y_pred, y_proba):
    """Categorize test cases into easy, hard, and boundary."""
    categories = {
        'easy_positive': [],  # High similarity, label=1
        'easy_negative': [],  # Low similarity, label=0
        'hard_negative': [],  # High similarity, label=0 (hard negative)
        'boundary': []        # Similarity ~0.5
    }
    
    for i in range(len(similarities)):
        sim = similarities[i]
        label = y_true[i]
        pred = y_pred[i]
        proba = y_proba[i]
        
        case = {
            'index': i,
            'similarity': float(sim),
            'true_label': int(label),
            'pred_label': int(pred),
            'probability': float(proba),
            'correct': int(label == pred)
        }
        
        if 0.4 <= sim <= 0.6:
            categories['boundary'].append(case)
        elif sim > 0.85:
            if label == 0:
                categories['hard_negative'].append(case)
            else:
                categories['easy_positive'].append(case)
        elif sim < 0.3:
            if label == 0:
                categories['easy_negative'].append(case)
    
    return categories

def compute_metrics_by_category(categories):
    """Compute performance metrics for each category."""
    metrics = {}
    
    for cat_name, cases in categories.items():
        if not cases:
            metrics[cat_name] = None
            continue
        
        y_true = [c['true_label'] for c in cases]
        y_pred = [c['pred_label'] for c in cases]
        y_proba = [c['probability'] for c in cases]
        
        metrics[cat_name] = {
            'count': len(cases),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.0,
            'mean_similarity': np.mean([c['similarity'] for c in cases]),
            'error_rate': 1 - accuracy_score(y_true, y_pred)
        }
    
    return metrics

def plot_similarity_distribution(similarities, y_true, output_dir):
    """Plot similarity score distribution by label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(similarities[y_true == 0], bins=50, alpha=0.6, label='Not Novel (0)', color='red')
    axes[0].hist(similarities[y_true == 1], bins=50, alpha=0.6, label='Novel (1)', color='blue')
    axes[0].axvline(0.85, color='orange', linestyle='--', label='Hard Negative Threshold')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Similarity Score Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data = [similarities[y_true == 0], similarities[y_true == 1]]
    axes[1].boxplot(data, labels=['Not Novel (0)', 'Novel (1)'])
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Score by Label')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'similarity_distribution.png'}")

def plot_category_performance(metrics, output_dir):
    """Plot performance metrics by category."""
    categories = ['easy_positive', 'easy_negative', 'hard_negative', 'boundary']
    cat_labels = ['Easy Positive', 'Easy Negative', 'Hard Negative', 'Boundary']
    
    # Filter out None categories
    valid_metrics = {k: v for k, v in metrics.items() if v is not None}
    valid_cats = [c for c in categories if c in valid_metrics]
    valid_labels = [cat_labels[categories.index(c)] for c in valid_cats]
    
    if not valid_metrics:
        print("  Warning: No valid categories for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    accs = [valid_metrics[c]['accuracy'] for c in valid_cats]
    axes[0, 0].bar(valid_labels, accs, color=['green', 'green', 'orange', 'blue'])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy by Case Category')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(accs):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Error Rate
    errors = [valid_metrics[c]['error_rate'] for c in valid_cats]
    axes[0, 1].bar(valid_labels, errors, color=['green', 'green', 'orange', 'blue'])
    axes[0, 1].set_ylabel('Error Rate')
    axes[0, 1].set_title('Error Rate by Case Category')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(errors):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # F1 Score
    f1s = [valid_metrics[c]['f1'] for c in valid_cats]
    axes[1, 0].bar(valid_labels, f1s, color=['green', 'green', 'orange', 'blue'])
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score by Case Category')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(f1s):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Count
    counts = [valid_metrics[c]['count'] for c in valid_cats]
    axes[1, 1].bar(valid_labels, counts, color=['green', 'green', 'orange', 'blue'])
    axes[1, 1].set_ylabel('Number of Cases')
    axes[1, 1].set_title('Case Count by Category')
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(counts):
        axes[1, 1].text(i, v + max(counts)*0.02, f'{v}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_performance.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'category_performance.png'}")

def plot_roc_by_category(categories, output_dir):
    """Plot ROC curves for different categories."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {
        'easy_positive': 'green',
        'easy_negative': 'blue',
        'hard_negative': 'orange',
        'boundary': 'purple'
    }
    
    for cat_name, cases in categories.items():
        if not cases or len(cases) < 10:
            continue
        
        y_true = np.array([c['true_label'] for c in cases])
        y_proba = np.array([c['probability'] for c in cases])
        
        if len(set(y_true)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        label = cat_name.replace('_', ' ').title()
        ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.3f})', 
                color=colors.get(cat_name, 'gray'))
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Case Category')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_by_category.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'roc_by_category.png'}")

def plot_confusion_matrices(categories, output_dir):
    """Plot confusion matrices for each category."""
    n_cats = len([c for c in categories.values() if c])
    if n_cats == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    cat_names = ['easy_positive', 'easy_negative', 'hard_negative', 'boundary']
    cat_labels = ['Easy Positive', 'Easy Negative', 'Hard Negative', 'Boundary']
    
    for idx, (cat_name, cat_label) in enumerate(zip(cat_names, cat_labels)):
        if cat_name not in categories or not categories[cat_name]:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(cat_label)
            continue
        
        cases = categories[cat_name]
        y_true = [c['true_label'] for c in cases]
        y_pred = [c['pred_label'] for c in cases]
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Not Novel', 'Novel'],
                   yticklabels=['Not Novel', 'Novel'])
        axes[idx].set_title(f'{cat_label}\n(n={len(cases)})')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_by_category.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'confusion_matrices_by_category.png'}")

def save_examples(categories, output_dir, max_examples=5):
    """Save example cases for manual inspection."""
    examples = {}
    
    for cat_name, cases in categories.items():
        if not cases:
            continue
        
        # Sort by error (wrong predictions first)
        sorted_cases = sorted(cases, key=lambda x: (not x['correct'], abs(x['probability'] - 0.5)))
        
        examples[cat_name] = []
        for case in sorted_cases[:max_examples]:
            examples[cat_name].append({
                'index': case['index'],
                'similarity': case['similarity'],
                'true_label': case['true_label'],
                'predicted_label': case['pred_label'],
                'probability': case['probability'],
                'correct': case['correct']
            })
    
    with open(output_dir / 'edge_case_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"  Saved: {output_dir / 'edge_case_examples.json'}")

def main():
    print("="*70)
    print("EDGE CASE ANALYSIS")
    print("="*70)
    
    # Setup
    output_dir = Path("results/analysis/edge_cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data()
    
    # Get predictions
    print("\nGenerating predictions...")
    y_proba = model.predict_proba(X_test).flatten()
    y_pred = model.predict(X_test)
    
    # Extract similarities
    similarities = extract_similarity_scores(X_test, feature_names)
    
    # Categorize cases
    print("\nCategorizing test cases...")
    categories = categorize_cases(similarities, y_test, y_pred, y_proba)
    
    print(f"\nCase Distribution:")
    for cat_name, cases in categories.items():
        print(f"  {cat_name}: {len(cases)} cases")
    
    # Compute metrics
    print("\nComputing metrics by category...")
    metrics = compute_metrics_by_category(categories)
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE BY CATEGORY")
    print("="*70)
    
    for cat_name, cat_metrics in metrics.items():
        if cat_metrics is None:
            continue
        print(f"\n{cat_name.upper().replace('_', ' ')}:")
        print(f"  Count: {cat_metrics['count']}")
        print(f"  Accuracy: {cat_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {cat_metrics['f1']:.4f}")
        print(f"  Error Rate: {cat_metrics['error_rate']:.4f}")
        print(f"  Mean Similarity: {cat_metrics['mean_similarity']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1. Similarity distribution...")
    plot_similarity_distribution(similarities, y_test, output_dir)
    
    print("\n2. Category performance...")
    plot_category_performance(metrics, output_dir)
    
    print("\n3. ROC curves by category...")
    plot_roc_by_category(categories, output_dir)
    
    print("\n4. Confusion matrices...")
    plot_confusion_matrices(categories, output_dir)
    
    # Save examples
    print("\n5. Saving examples...")
    save_examples(categories, output_dir)
    
    # Save metrics
    print("\n6. Saving metrics...")
    metrics_serializable = {k: v for k, v in metrics.items() if v is not None}
    with open(output_dir / 'edge_case_metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\n  Saved: {output_dir / 'edge_case_metrics.json'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey Findings:")
    
    if 'hard_negative' in metrics and metrics['hard_negative']:
        hn_metrics = metrics['hard_negative']
        print(f"\n  Hard Negatives:")
        print(f"    - {hn_metrics['count']} cases identified")
        print(f"    - Accuracy: {hn_metrics['accuracy']:.2%}")
        print(f"    - Error Rate: {hn_metrics['error_rate']:.2%}")
        print(f"    - Mean Similarity: {hn_metrics['mean_similarity']:.3f}")
        print(f"    - This shows model's ability to distinguish semantically")
        print(f"      different patents despite high surface similarity")

if __name__ == '__main__':
    main()

