"""
MLP Classifier for Patent Novelty Scoring

Architecture:
- (64, 32) hidden units
- L2 regularization with alpha=1e-4
- Early stopping with validation
- StandardScaler preprocessing
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, brier_score_loss
)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentNoveltyClassifier:
    """MLP classifier for patent pair similarity scoring."""
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        alpha: float = 1e-4,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            hidden_layer_sizes: Architecture of hidden layers
            alpha: L2 regularization strength
            learning_rate_init: Initial learning rate
            max_iter: Maximum training iterations
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction for validation during training
            n_iter_no_change: Patience for early stopping
            random_state: Random seed
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
    def _create_model(self) -> MLPClassifier:
        """Create a new MLP model."""
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=self.alpha,
            learning_rate='adaptive',
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=True
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'PatentNoveltyClassifier':
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (for evaluation only)
            y_val: Optional validation labels
            feature_names: Names of features
            
        Returns:
            self
        """
        logger.info(f"Training MLP with {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Architecture: {self.hidden_layer_sizes}, alpha={self.alpha}")
        
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Record training history
        self.training_history = {
            'loss_curve': self.model.loss_curve_,
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_ if hasattr(self.model, 'best_loss_') else None
        }
        
        # Training evaluation
        train_pred = self.model.predict(X_train_scaled)
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        
        logger.info(f"Training completed after {self.model.n_iter_} iterations")
        logger.info(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
        logger.info(f"Training ROC-AUC: {roc_auc_score(y_train, train_proba):.4f}")
        
        # Validation evaluation if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Returns:
            Dictionary with all metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'brier_score': brier_score_loss(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance from model weights.
        
        Uses the absolute sum of weights from input to first hidden layer.
        """
        if self.model is None or self.feature_names is None:
            return {}
        
        # Get weights from input to first hidden layer
        first_layer_weights = np.abs(self.model.coefs_[0])
        importance = first_layer_weights.sum(axis=1)
        
        # Normalize
        importance = importance / importance.sum()
        
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
    
    def save(self, output_dir: str):
        """Save model and scaler."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(output_dir / 'mlp_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save config
        config = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'feature_names': self.feature_names,
            'training_history': {
                k: v if not isinstance(v, list) else v 
                for k, v in self.training_history.items()
            }
        }
        with open(output_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str) -> 'PatentNoveltyClassifier':
        """Load a saved model."""
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'model_config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            hidden_layer_sizes=tuple(config['hidden_layer_sizes']),
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init']
        )
        
        # Load model
        with open(model_dir / 'mlp_model.pkl', 'rb') as f:
            instance.model = pickle.load(f)
        
        # Load scaler
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            instance.scaler = pickle.load(f)
        
        instance.feature_names = config.get('feature_names')
        instance.training_history = config.get('training_history', {})
        
        logger.info(f"Model loaded from {model_dir}")
        return instance
    
    def plot_training_curve(self, save_path: Optional[str] = None):
        """Plot training loss curve."""
        if not self.training_history.get('loss_curve'):
            logger.warning("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss_curve'], 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('MLP Training Loss Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curve saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve."""
        y_proba = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'MLP (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=14)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()


def demo_classifier():
    """Demo the classifier with random data."""
    print("=== MLP Classifier Demo ===\n")
    
    # Generate random data
    np.random.seed(42)
    n_samples = 1000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule for demo
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    feature_names = [f'feature_{i}' for i in range(n_features)]
    clf = PatentNoveltyClassifier()
    clf.fit(X_train, y_train, X_test, y_test, feature_names)
    
    # Evaluate
    metrics = clf.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'classification_report']:
            print(f"  {key}: {value:.4f}")
    
    # Feature importance
    importance = clf.get_feature_importance()
    print("\nFeature Importance:")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    demo_classifier()


