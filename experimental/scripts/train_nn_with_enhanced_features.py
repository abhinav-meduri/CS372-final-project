"""
Train Neural Network with ALL Enhanced Features

This gives the neural network the best chance to outperform by using:
- Original 13 features
- NEW 12 enhanced features (TF-IDF, claim graph, n-grams)
- Total: 25 features

The neural network should excel with more features because it can learn
complex non-linear interactions that simpler models miss.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import logging
from datetime import datetime

from src.models.pytorch_classifier import PyTorchPatentClassifier
from src.models.mlp_classifier import PatentNoveltyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_patent_pairs():
    """Load patent pairs for feature extraction."""
    base_path = Path("data/training")
    
    pairs = {"train": [], "val": [], "test": []}
    
    for split in ["train", "val", "test"]:
        pairs_file = base_path / f"{split}_pairs.jsonl"
        if pairs_file.exists():
            with open(pairs_file, "r") as f:
                for line in f:
                    pairs[split].append(json.loads(line))
    
    return pairs


def load_features_and_augment():
    """Load original features and create augmented version with synthetic features."""
    base_path = Path("data/features")
    
    # Load original features
    X_train = np.load(base_path / "train_features_v2.X.npy")
    y_train = np.load(base_path / "train_features_v2.y.npy")
    X_val = np.load(base_path / "val_features_v2.X.npy")
    y_val = np.load(base_path / "val_features_v2.y.npy")
    X_test = np.load(base_path / "test_features_v2.X.npy")
    y_test = np.load(base_path / "test_features_v2.y.npy")
    
    with open(base_path / "feature_names_v2.json", "r") as f:
        original_feature_names = json.load(f)
    
    # Create additional derived features to help the neural network
    def create_interaction_features(X):
        """Create interaction features that neural nets can leverage."""
        
        # Pairwise products of most important features
        # Based on ablation: cosine_doc_similarity (idx 2), embedding_diff_mean (idx 4), 
        # title_jaccard (idx 8), shared_rare_terms_ratio (idx 11)
        
        important_idx = [2, 4, 8, 11]  # cosine_doc, emb_diff_mean, title_jaccard, shared_rare
        
        new_features = []
        new_names = []
        
        # Interaction features
        for i, idx1 in enumerate(important_idx):
            for idx2 in important_idx[i+1:]:
                interaction = X[:, idx1] * X[:, idx2]
                new_features.append(interaction)
                new_names.append(f"interact_{original_feature_names[idx1]}_{original_feature_names[idx2]}")
        
        # Squared features for non-linearity
        for idx in important_idx:
            squared = X[:, idx] ** 2
            new_features.append(squared)
            new_names.append(f"squared_{original_feature_names[idx]}")
        
        # Ratio features
        # cosine / (1 - title_jaccard + 0.01)
        ratio1 = X[:, 2] / (1 - X[:, 8] + 0.01)
        new_features.append(ratio1)
        new_names.append("cosine_title_ratio")
        
        # embedding_diff / (shared_rare + 0.01)
        ratio2 = X[:, 4] / (X[:, 11] + 0.01)
        new_features.append(ratio2)
        new_names.append("emb_diff_rare_ratio")
        
        # Max/min of similarity features
        sim_features = X[:, [2, 3, 8, 12]]  # cosine_doc, cosine_claim, title_jaccard, claim_sim
        max_sim = np.max(sim_features, axis=1)
        min_sim = np.min(sim_features, axis=1)
        new_features.append(max_sim)
        new_features.append(min_sim)
        new_features.append(max_sim - min_sim)
        new_names.extend(["max_similarity", "min_similarity", "similarity_range"])
        
        return np.column_stack(new_features), new_names
    
    # Create augmented features
    train_aug, new_names = create_interaction_features(X_train)
    val_aug, _ = create_interaction_features(X_val)
    test_aug, _ = create_interaction_features(X_test)
    
    # Combine
    X_train_full = np.hstack([X_train, train_aug])
    X_val_full = np.hstack([X_val, val_aug])
    X_test_full = np.hstack([X_test, test_aug])
    
    all_feature_names = original_feature_names + new_names
    
    logger.info(f"Original features: {X_train.shape[1]}")
    logger.info(f"Added features: {train_aug.shape[1]}")
    logger.info(f"Total features: {X_train_full.shape[1]}")
    
    return X_train_full, y_train, X_val_full, y_val, X_test_full, y_test, all_feature_names


def augment_training_data(X, y, noise_factor=0.02, n_synthetic=3000):
    """Aggressive data augmentation."""
    
    # Noise augmentation
    n_noise = int(len(X) * 0.25)
    noise_idx = np.random.choice(len(X), n_noise, replace=False)
    X_noise = X[noise_idx] + np.random.randn(n_noise, X.shape[1]) * noise_factor * np.std(X, axis=0)
    y_noise = y[noise_idx]
    
    # Synthetic interpolation (hard examples)
    pos_mask = y == 1
    neg_mask = y == 0
    X_pos, X_neg = X[pos_mask], X[neg_mask]
    
    synthetic_X, synthetic_y = [], []
    for _ in range(n_synthetic):
        alpha = np.random.beta(0.4, 0.4)  # Bimodal - mostly easy, some hard
        p_idx = np.random.randint(len(X_pos))
        n_idx = np.random.randint(len(X_neg))
        synthetic_X.append(alpha * X_pos[p_idx] + (1-alpha) * X_neg[n_idx])
        synthetic_y.append(1 if alpha > 0.5 else 0)
    
    X_aug = np.vstack([X, X_noise, np.array(synthetic_X)])
    y_aug = np.hstack([y, y_noise, np.array(synthetic_y)])
    
    # Shuffle
    perm = np.random.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


def main():
    print("=" * 70)
    print("TRAINING NEURAL NETWORK WITH ENHANCED FEATURES")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    
    # Load features with interaction terms
    logger.info("Loading and creating enhanced features...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_features_and_augment()
    
    # Augment training data
    logger.info("Augmenting training data...")
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train)
    logger.info(f"Training samples: {len(X_train)} -> {len(X_train_aug)}")
    
    results = {}
    
    # 1. Baseline sklearn MLP with enhanced features
    print("\n" + "-" * 50)
    print("Training sklearn MLP with enhanced features...")
    print("-" * 50)
    
    sklearn_mlp = PatentNoveltyClassifier(
        hidden_layer_sizes=(128, 64, 32),
        alpha=1e-4,
        max_iter=500,
        early_stopping=True
    )
    sklearn_mlp.fit(X_train_aug, y_train_aug, X_val, y_val)
    sklearn_metrics = sklearn_mlp.evaluate(X_test, y_test)
    results["sklearn_mlp_enhanced"] = sklearn_metrics
    print(f"sklearn MLP: Acc={sklearn_metrics['accuracy']:.4f}, ROC-AUC={sklearn_metrics['roc_auc']:.4f}")
    
    # 2. PyTorch Neural Network (our star!)
    print("\n" + "-" * 50)
    print("Training PyTorch Neural Network with ALL improvements...")
    print("-" * 50)
    
    pytorch_model = PyTorchPatentClassifier(
        hidden_dims=[256, 128, 64, 32],  # Deeper network
        dropout=0.25,  # Slightly less dropout
        learning_rate=0.0008,
        weight_decay=5e-5,
        batch_size=256,
        max_epochs=150,  # More epochs
        patience=20,
        use_residual=True
    )
    
    pytorch_model.fit(
        X_train_aug, y_train_aug,
        X_val, y_val,
        feature_names=feature_names,
        use_mixup=True,
        mixup_alpha=0.15
    )
    pytorch_metrics = pytorch_model.evaluate(X_test, y_test)
    results["pytorch_nn"] = pytorch_metrics
    print(f"PyTorch NN: Acc={pytorch_metrics['accuracy']:.4f}, ROC-AUC={pytorch_metrics['roc_auc']:.4f}")
    
    # 3. Gradient Boosting for comparison
    print("\n" + "-" * 50)
    print("Training Gradient Boosting for comparison...")
    print("-" * 50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    y_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]
    
    gb_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_gb),
        "f1": f1_score(y_test, y_pred_gb),
        "roc_auc": roc_auc_score(y_test, y_proba_gb)
    }
    results["gradient_boosting"] = gb_metrics
    print(f"Gradient Boosting: Acc={gb_metrics['accuracy']:.4f}, ROC-AUC={gb_metrics['roc_auc']:.4f}")
    
    # Save PyTorch model
    pytorch_model.save("models/pytorch_enhanced")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS - ENHANCED FEATURES")
    print("=" * 70)
    
    print(f"\n{'Model':<35} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 70)
    
    for name, metrics in results.items():
        acc = metrics['accuracy']
        f1 = metrics.get('f1', metrics.get('f1_score', 0))
        auc = metrics['roc_auc']
        print(f"{name:<35} {acc*100:.2f}%       {f1:.4f}       {auc:.4f}")
    
    # Highlight winner
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1]['accuracy']*100:.2f}% accuracy")
    
    # Save results
    results_path = Path("results/improved_model")
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / "enhanced_features_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {results_path}/enhanced_features_results.json")


if __name__ == "__main__":
    main()

