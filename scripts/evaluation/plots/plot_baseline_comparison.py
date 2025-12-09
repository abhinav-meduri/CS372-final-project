"""Compare baselines vs MLP/PyTorch metrics and print summaries."""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("BASELINE COMPARISON (dry run)")

# Load test data
print("\n1. Loading test data...")
X_test = np.load(project_root/'data'/'features'/'test_features_v2.X.npy')
y_test = np.load(project_root/'data'/'features'/'test_features_v2.y.npy')

# Remove BM25 and CPC features (indices 0, 1, 6) to match 10-feature models
indices_to_remove = [0, 1, 6]
indices_to_keep = [i for i in range(13) if i not in indices_to_remove]
X_test_10feat = X_test[:, indices_to_keep]

print(f"   Test set size: {len(y_test)} samples")
print(f"   Features: {X_test_10feat.shape[1]} (10 features after removal)")

# Load model metrics
print("\n2. Loading model metrics...")
mlp_metrics = json.load(open(project_root/'models'/'mlp'/'mlp_metrics.json'))
pytorch_metrics = json.load(open(project_root/'results'/'pytorch_nn'/'pytorch_metrics.json'))

# Baseline 1: Random Guessing (50/50)
print("\n3. Evaluating baseline methods...")
print("   - Random Guessing (50/50)...")
np.random.seed(42)
random_pred = np.random.randint(0, 2, size=len(y_test))
random_proba = np.random.rand(len(y_test))
random_acc = accuracy_score(y_test, random_pred)
random_roc = roc_auc_score(y_test, random_proba)
random_f1 = f1_score(y_test, random_pred)
random_precision = precision_score(y_test, random_pred, zero_division=0)
random_recall = recall_score(y_test, random_pred, zero_division=0)

# Baseline 2: Majority Class
print("   - Majority Class Classifier...")
dummy_majority = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_majority.fit(X_test_10feat, y_test)
majority_pred = dummy_majority.predict(X_test_10feat)
majority_proba = dummy_majority.predict_proba(X_test_10feat)[:, 1]
majority_acc = accuracy_score(y_test, majority_pred)
majority_roc = roc_auc_score(y_test, majority_proba)
majority_f1 = f1_score(y_test, majority_pred)
majority_precision = precision_score(y_test, majority_pred, zero_division=0)
majority_recall = recall_score(y_test, majority_pred, zero_division=0)

# Baseline 3: Logistic Regression
print("   - Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_test_10feat, y_test)
lr_pred = lr.predict(X_test_10feat)
lr_proba = lr.predict_proba(X_test_10feat)[:, 1]
lr_acc = accuracy_score(y_test, lr_pred)
lr_roc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)

# Baseline 4: Cosine Similarity Threshold (heuristic)
# Use the first feature which is cosine_doc_similarity
print("   - Cosine Similarity Threshold (heuristic)...")
cosine_sim = X_test_10feat[:, 0]  # cosine_doc_similarity is first feature
# Find optimal threshold
thresholds = np.linspace(0.3, 0.9, 100)
best_threshold = 0.5
best_acc = 0
for thresh in thresholds:
    pred = (cosine_sim >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

cosine_pred = (cosine_sim >= best_threshold).astype(int)
cosine_proba = cosine_sim  # Use similarity as probability proxy
cosine_acc = accuracy_score(y_test, cosine_pred)
cosine_roc = roc_auc_score(y_test, cosine_proba)
cosine_f1 = f1_score(y_test, cosine_pred)
cosine_precision = precision_score(y_test, cosine_pred, zero_division=0)
cosine_recall = recall_score(y_test, cosine_pred, zero_division=0)

# Collect all results
results = {
    'Random Guessing': {
        'accuracy': random_acc,
        'precision': random_precision,
        'recall': random_recall,
        'roc_auc': random_roc,
        'f1': random_f1
    },
    'Majority Class': {
        'accuracy': majority_acc,
        'precision': majority_precision,
        'recall': majority_recall,
        'roc_auc': majority_roc,
        'f1': majority_f1
    },
    'Logistic Regression': {
        'accuracy': lr_acc,
        'precision': lr_precision,
        'recall': lr_recall,
        'roc_auc': lr_roc,
        'f1': lr_f1
    },
    'Cosine Similarity (heuristic)': {
        'accuracy': cosine_acc,
        'precision': cosine_precision,
        'recall': cosine_recall,
        'roc_auc': cosine_roc,
        'f1': cosine_f1,
        'threshold': best_threshold
    },
    'MLP Classifier': {
        'accuracy': mlp_metrics['test']['accuracy'],
        'roc_auc': mlp_metrics['test']['roc_auc'],
        'f1': mlp_metrics['test']['f1']
    },
    'PyTorch Neural Net': {
        'accuracy': pytorch_metrics['test']['accuracy'],
        'roc_auc': pytorch_metrics['test']['roc_auc'],
        'f1': pytorch_metrics['test']['f1']
    }
}

# Print results
print("RESULTS SUMMARY")
print(f"{'Method':<30} {'Accuracy':<12} {'ROC-AUC':<12} {'F1 Score':<12}")
print("-"*70)
for method, metrics in results.items():
    print(f"{method:<30} {metrics['accuracy']:<12.4f} {metrics['roc_auc']:<12.4f} {metrics['f1']:<12.4f}")

print("\nSummary (no plots saved):")
for method, metrics in results.items():
    print(
        f"{method:<28} acc={metrics['accuracy']:.4f} "
        f"roc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f}"
    )

print("\nBaseline comparison complete (dry run).")

