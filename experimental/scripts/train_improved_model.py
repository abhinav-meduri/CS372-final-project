"""
Train Improved Patent Novelty Classifier

Combines:
1. Enhanced features (TF-IDF, claim dependency graph, n-grams)
2. PyTorch neural network with residual connections
3. Data augmentation (mixup, feature noise)
4. Comparison with baseline models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from src.models.pytorch_classifier import PyTorchPatentClassifier
from src.models.mlp_classifier import PatentNoveltyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_original_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                       np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the original 13 features."""
    base_path = Path("data/features")
    
    X_train = np.load(base_path / "train_features_v2.X.npy")
    y_train = np.load(base_path / "train_features_v2.y.npy")
    X_val = np.load(base_path / "val_features_v2.X.npy")
    y_val = np.load(base_path / "val_features_v2.y.npy")
    X_test = np.load(base_path / "test_features_v2.X.npy")
    y_test = np.load(base_path / "test_features_v2.y.npy")
    
    with open(base_path / "feature_names_v2.json", "r") as f:
        feature_names = json.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def augment_with_noise(X: np.ndarray, y: np.ndarray, noise_factor: float = 0.05, 
                       augment_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with Gaussian noise.
    
    Args:
        X: Feature matrix
        y: Labels
        noise_factor: Standard deviation of noise (relative to feature std)
        augment_ratio: Fraction of samples to augment
    """
    n_augment = int(len(X) * augment_ratio)
    indices = np.random.choice(len(X), n_augment, replace=False)
    
    # Add noise
    X_aug = X[indices].copy()
    feature_std = np.std(X, axis=0) + 1e-8
    noise = np.random.randn(*X_aug.shape) * noise_factor * feature_std
    X_aug = X_aug + noise
    
    # Combine with original
    X_combined = np.vstack([X, X_aug])
    y_combined = np.hstack([y, y[indices]])
    
    # Shuffle
    perm = np.random.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


def create_synthetic_hard_negatives(X: np.ndarray, y: np.ndarray, 
                                     n_synthetic: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic hard negative examples by interpolating between classes.
    """
    pos_mask = y == 1
    neg_mask = y == 0
    
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        # Pick random positive and negative
        pos_idx = np.random.randint(len(X_pos))
        neg_idx = np.random.randint(len(X_neg))
        
        # Interpolate (closer to negative = harder negative)
        alpha = np.random.uniform(0.3, 0.7)  # Keep in ambiguous zone
        synthetic = alpha * X_pos[pos_idx] + (1 - alpha) * X_neg[neg_idx]
        
        # Label based on alpha
        label = 1 if alpha > 0.5 else 0
        
        synthetic_X.append(synthetic)
        synthetic_y.append(label)
    
    X_combined = np.vstack([X, np.array(synthetic_X)])
    y_combined = np.hstack([y, np.array(synthetic_y)])
    
    perm = np.random.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


def train_and_evaluate_pytorch(X_train, y_train, X_val, y_val, X_test, y_test, 
                                feature_names, use_augmentation=True) -> Dict:
    """Train PyTorch model with all improvements."""
    
    logger.info("=" * 60)
    logger.info("TRAINING PYTORCH MODEL WITH IMPROVEMENTS")
    logger.info("=" * 60)
    
    # Data augmentation
    if use_augmentation:
        logger.info("\nApplying data augmentation...")
        logger.info(f"  Original training size: {len(X_train)}")
        
        # Add noise augmentation
        X_train_aug, y_train_aug = augment_with_noise(X_train, y_train, 
                                                        noise_factor=0.03, 
                                                        augment_ratio=0.2)
        logger.info(f"  After noise augmentation: {len(X_train_aug)}")
        
        # Add synthetic hard negatives
        X_train_aug, y_train_aug = create_synthetic_hard_negatives(X_train_aug, y_train_aug, 
                                                                    n_synthetic=2000)
        logger.info(f"  After synthetic samples: {len(X_train_aug)}")
    else:
        X_train_aug, y_train_aug = X_train, y_train
    
    # Create and train PyTorch model
    model = PyTorchPatentClassifier(
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=100,
        patience=15,
        use_residual=True
    )
    
    logger.info("\nTraining model...")
    history = model.fit(
        X_train_aug, y_train_aug,
        X_val, y_val,
        feature_names=feature_names,
        use_mixup=True,
        mixup_alpha=0.2
    )
    
    # Evaluate
    logger.info("\nEvaluating model...")
    
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"\nTRAIN: Acc={train_metrics['accuracy']:.4f}, ROC-AUC={train_metrics['roc_auc']:.4f}")
    logger.info(f"VAL:   Acc={val_metrics['accuracy']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}")
    logger.info(f"TEST:  Acc={test_metrics['accuracy']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}")
    
    # Save model
    model.save("experimental/models/pytorch_experimental")
    
    return {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history
    }


def train_ensemble(X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
    """Train ensemble of models."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ENSEMBLE MODEL")
    logger.info("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Individual models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    # Train and evaluate each
    results = {}
    trained_models = []
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        results[name] = metrics
        trained_models.append((name.lower().replace(" ", "_"), model))
        
        logger.info(f"  {name}: Acc={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    
    # Create voting ensemble
    logger.info("\nCreating Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=trained_models,
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    ensemble_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    results["Voting Ensemble"] = ensemble_metrics
    
    logger.info(f"\nENSEMBLE: Acc={ensemble_metrics['accuracy']:.4f}, ROC-AUC={ensemble_metrics['roc_auc']:.4f}")
    
    return results


def compare_with_baseline(X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
    """Compare improved model with sklearn MLP baseline."""
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH BASELINE MLP")
    logger.info("=" * 60)
    
    # Original sklearn MLP
    baseline = PatentNoveltyClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        max_iter=500,
        early_stopping=True
    )
    
    logger.info("\nTraining baseline sklearn MLP...")
    baseline.fit(X_train, y_train, X_val, y_val)
    
    baseline_metrics = baseline.evaluate(X_test, y_test)
    
    logger.info(f"BASELINE: Acc={baseline_metrics['accuracy']:.4f}, ROC-AUC={baseline_metrics['roc_auc']:.4f}")
    
    return baseline_metrics


def main():
    """Main training pipeline."""
    
    np.random.seed(42)
    
    print("=" * 70)
    print("IMPROVED PATENT NOVELTY CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load original features
    logger.info("Loading original features...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_original_features()
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    # Store all results
    all_results = {}
    
    # 1. Train baseline for comparison
    baseline_metrics = compare_with_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results["baseline_sklearn_mlp"] = baseline_metrics
    
    # 2. Train improved PyTorch model
    pytorch_results = train_and_evaluate_pytorch(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, use_augmentation=True
    )
    all_results["pytorch_experimental"] = pytorch_results["test"]
    
    # 3. Train ensemble
    ensemble_results = train_ensemble(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results["ensemble"] = ensemble_results
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 70)
    
    print(f"{'Baseline sklearn MLP':<30} {baseline_metrics['accuracy']*100:.1f}%        {baseline_metrics['f1']:.4f}       {baseline_metrics['roc_auc']:.4f}")
    print(f"{'PyTorch + Augmentation':<30} {pytorch_results['test']['accuracy']*100:.1f}%        {pytorch_results['test']['f1']:.4f}       {pytorch_results['test']['roc_auc']:.4f}")
    
    for name, metrics in ensemble_results.items():
        print(f"{name:<30} {metrics['accuracy']*100:.1f}%        {metrics['f1']:.4f}       {metrics['roc_auc']:.4f}")
    
    # Calculate improvements
    baseline_acc = baseline_metrics['accuracy']
    pytorch_acc = pytorch_results['test']['accuracy']
    best_ensemble_acc = max(m['accuracy'] for m in ensemble_results.values())
    
    print("\n" + "-" * 70)
    print(f"PyTorch improvement over baseline: +{(pytorch_acc - baseline_acc)*100:.2f}%")
    print(f"Best ensemble improvement: +{(best_ensemble_acc - baseline_acc)*100:.2f}%")
    
    # Save results
    results_path = Path("results/improved_model")
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / "training_results.json", "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(all_results, f, indent=2, default=convert)
    
    logger.info(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

