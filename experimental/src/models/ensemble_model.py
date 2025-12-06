"""
Stacking Ensemble Model for Patent Novelty Classification

Combines multiple models for better performance:
1. PyTorch Neural Network (captures non-linear interactions)
2. Gradient Boosting (captures feature importance)
3. Meta-learner combines predictions

Also includes:
- Probability calibration (Platt scaling)
- Confidence estimation
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibratedEnsemble:
    """
    Stacking ensemble with probability calibration.
    
    Combines multiple models:
    - PyTorch Neural Network
    - Gradient Boosting
    - Logistic Regression
    
    Uses a meta-learner (Logistic Regression) to combine predictions,
    followed by probability calibration (Platt scaling).
    """
    
    def __init__(
        self,
        use_calibration: bool = True,
        calibration_method: str = 'sigmoid',  # 'sigmoid' (Platt) or 'isotonic'
        n_estimators_gb: int = 100,
        max_depth_gb: int = 5
    ):
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.n_estimators_gb = n_estimators_gb
        self.max_depth_gb = max_depth_gb
        
        self.scaler = StandardScaler()
        self.ensemble = None
        self.calibrated_model = None
        self.feature_names = None
        self.feature_importance = None
        
    def _create_ensemble(self):
        """Create stacking ensemble."""
        
        # Base models
        base_models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=self.n_estimators_gb,
                max_depth=self.max_depth_gb,
                random_state=42
            )),
            ('lr', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=500, random_state=42)
        
        # Stacking ensemble
        self.ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            passthrough=True  # Include original features
        )
        
        return self.ensemble
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None
    ):
        """
        Train the ensemble model.
        """
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train ensemble
        logger.info("Training stacking ensemble...")
        self._create_ensemble()
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities if requested
        if self.use_calibration and X_val is not None:
            logger.info(f"Calibrating probabilities ({self.calibration_method})...")
            X_val_scaled = self.scaler.transform(X_val)
            
            self.calibrated_model = CalibratedClassifierCV(
                self.ensemble,
                method=self.calibration_method,
                cv='prefit'
            )
            self.calibrated_model.fit(X_val_scaled, y_val)
        
        # Compute feature importance from Gradient Boosting
        gb_model = self.ensemble.named_estimators_['gb']
        self.feature_importance = dict(zip(
            feature_names or [f'f{i}' for i in range(X_train.shape[1])],
            gb_model.feature_importances_
        ))
        
        logger.info("Ensemble training complete!")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probability predictions."""
        X_scaled = self.scaler.transform(X)
        
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)
        else:
            return self.ensemble.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Returns:
            predictions: Class predictions (0 or 1)
            probabilities: Probability of class 1
            confidence: Confidence level (distance from 0.5)
        """
        probs = self.predict_proba(X)[:, 1]
        predictions = (probs > 0.5).astype(int)
        confidence = np.abs(probs - 0.5) * 2  # Scale to 0-1
        
        return predictions, probs, confidence
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Comprehensive evaluation with calibration metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
            "brier_score": brier_score_loss(y, y_proba),
            "log_loss": log_loss(y, y_proba)
        }
        
        # Calibration quality (Expected Calibration Error approximation)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_acc = y[mask].mean()
                bin_conf = y_proba[mask].mean()
                ece += mask.sum() * np.abs(bin_acc - bin_conf)
        
        metrics["expected_calibration_error"] = ece / len(y)
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Gradient Boosting component."""
        return self.feature_importance
    
    def save(self, path: str):
        """Save the ensemble model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "ensemble_model.pkl", "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "ensemble": self.ensemble,
                "calibrated_model": self.calibrated_model,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance
            }, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load the ensemble model."""
        path = Path(path)
        
        with open(path / "ensemble_model.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.scaler = data["scaler"]
        self.ensemble = data["ensemble"]
        self.calibrated_model = data["calibrated_model"]
        self.feature_names = data["feature_names"]
        self.feature_importance = data["feature_importance"]
        
        logger.info(f"Ensemble loaded from {path}")


def train_and_evaluate_ensemble():
    """Train ensemble and compare with individual models."""
    
    # Load data
    base_path = Path("data/features")
    
    X_train = np.load(base_path / "train_features_v2.X.npy")
    y_train = np.load(base_path / "train_features_v2.y.npy")
    X_val = np.load(base_path / "val_features_v2.X.npy")
    y_val = np.load(base_path / "val_features_v2.y.npy")
    X_test = np.load(base_path / "test_features_v2.X.npy")
    y_test = np.load(base_path / "test_features_v2.y.npy")
    
    with open(base_path / "feature_names_v2.json", "r") as f:
        feature_names = json.load(f)
    
    print("=" * 60)
    print("CALIBRATED ENSEMBLE MODEL")
    print("=" * 60)
    print(f"\nTraining data: {len(X_train)} samples")
    print(f"Features: {len(feature_names)}")
    
    # Train ensemble
    ensemble = CalibratedEnsemble(
        use_calibration=True,
        calibration_method='sigmoid',
        n_estimators_gb=150,
        max_depth_gb=6
    )
    
    ensemble.fit(X_train, y_train, X_val, y_val, feature_names)
    
    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION RESULTS")
    print("-" * 60)
    
    test_metrics = ensemble.evaluate(X_test, y_test)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Precision:    {test_metrics['precision']:.4f}")
    print(f"  Recall:       {test_metrics['recall']:.4f}")
    print(f"  F1 Score:     {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"  Brier Score:  {test_metrics['brier_score']:.4f}")
    print(f"  Log Loss:     {test_metrics['log_loss']:.4f}")
    print(f"  ECE:          {test_metrics['expected_calibration_error']:.4f}")
    
    # Feature importance
    print("\n" + "-" * 60)
    print("FEATURE IMPORTANCE (from Gradient Boosting)")
    print("-" * 60)
    
    importance = ensemble.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    for name, imp in sorted_importance[:10]:
        bar = "█" * int(imp * 50)
        print(f"  {name:<30} {imp:.4f} {bar}")
    
    # Save model
    ensemble.save("models/ensemble")
    
    # Save results
    results_path = Path("results/ensemble")
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path / "ensemble_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    with open(results_path / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return ensemble, test_metrics


if __name__ == "__main__":
    ensemble, metrics = train_and_evaluate_ensemble()

