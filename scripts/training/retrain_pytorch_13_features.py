"""
Retrain PyTorch Neural Network on 13 Features (with real BM25)

This script:
1. Loads training data (13 features)
2. Retrains PyTorch model on 13 features
3. Saves new model checkpoint
4. Evaluates performance

Estimated time: 10-20 min (GPU) or 30-60 min (CPU)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import time
from datetime import datetime
from src.models.pytorch_classifier import PyTorchPatentClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

print("="*70)
print("RETRAINING PYTORCH MODEL ON 13 FEATURES (WITH REAL BM25)")
print("="*70)

# Load features
features_dir = Path("data/features")
print("\n1. Loading training data...")

X_train = np.load(features_dir / "train_features_v2.X.npy")
y_train = np.load(features_dir / "train_features_v2.y.npy")
X_val = np.load(features_dir / "val_features_v2.X.npy")
y_val = np.load(features_dir / "val_features_v2.y.npy")
X_test = np.load(features_dir / "test_features_v2.X.npy")
y_test = np.load(features_dir / "test_features_v2.y.npy")

with open(features_dir / "feature_names_v2.json", "r") as f:
    feature_names = json.load(f)

print(f"   Training set: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"   Validation set: {len(X_val)} samples")
print(f"   Test set: {len(X_test)} samples")
print(f"   Features: {feature_names}")

# Verify we have 13 features
assert X_train.shape[1] == 13, f"Expected 13 features, got {X_train.shape[1]}"

# Initialize model
print("\n2. Initializing PyTorch model...")
model = PyTorchPatentClassifier(
    hidden_dims=[128, 64, 32],
    dropout=0.3,
    learning_rate=0.001,
    max_epochs=100,
    patience=15,
    batch_size=256
)

# Train
print("\n3. Training model...")
print("   This will take 10-20 minutes on GPU, 30-60 minutes on CPU")
start_time = time.time()

model.fit(
    X_train, y_train,
    X_val, y_val,
    feature_names=feature_names,
    use_mixup=True
)

training_time = time.time() - start_time
print(f"\n   Training completed in {training_time/60:.1f} minutes")

# Evaluate
print("\n4. Evaluating model...")
train_metrics = model.evaluate(X_train, y_train)
val_metrics = model.evaluate(X_val, y_val)
test_metrics = model.evaluate(X_test, y_test)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nTraining Set:")
print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
print(f"  F1:        {train_metrics['f1']:.4f}")

print(f"\nValidation Set:")
print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
print(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
print(f"  F1:        {val_metrics['f1']:.4f}")

print(f"\nTest Set:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"  F1:        {test_metrics['f1']:.4f}")

# Save model
print("\n5. Saving model...")
model_path = Path("models/pytorch_nn")
model_path.mkdir(parents=True, exist_ok=True)
model.save(model_path)

# Save results
results = {
    "training_time_minutes": training_time / 60,
    "features": feature_names,
    "num_features": 13,
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "timestamp": datetime.now().isoformat()
}

results_path = model_path / "training_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"   Model saved to: {model_path}")
print(f"   Results saved to: {results_path}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"\nTo use this model, update patent_analyzer.py:")
print(f"  self.pytorch_model.load('models/pytorch_nn')")

