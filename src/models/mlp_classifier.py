"""
MLP Classifier for Patent Novelty Assessment

A wrapper around sklearn's MLPClassifier with additional functionality:
- Training history tracking
- Feature importance analysis
- Model saving/loading
- Plotting utilities
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class PatentNoveltyClassifier:
    """
    Multi-layer Perceptron classifier for patent novelty assessment.
    
    Wraps sklearn's MLPClassifier with additional features:
    - Training history tracking
    - Feature importance via permutation importance
    - Model persistence
    - Visualization utilities
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64,),
        alpha: float = 1e-5,
        learning_rate_init: float = 0.005,
        max_iter: int = 500,
        early_stopping: bool = True,
        n_iter_no_change: int = 20,
        random_state: int = 42
    ):
        """
        Initialize the MLP classifier.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            alpha: L2 regularization strength
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            early_stopping: Whether to use early stopping
            n_iter_no_change: Number of iterations with no improvement before stopping
            random_state: Random seed
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            verbose=False
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Train the MLP classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features (optional)
        """
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit model
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(X_train_scaled, y_train)
            
            # Track training history
            if hasattr(self.model, 'loss_curve_'):
                self.training_history = {
                    'loss': self.model.loss_curve_,
                    'n_iter': self.model.n_iter_
                }
        else:
            self.model.fit(X_train_scaled, y_train)
            if hasattr(self.model, 'loss_curve_'):
                self.training_history = {
                    'loss': self.model.loss_curve_,
                    'n_iter': self.model.n_iter_
                }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the model and return metrics.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0
        }
        
        # Brier score
        metrics['brier_score'] = np.mean((y_proba - y) ** 2)
        
        return metrics
    
    def get_feature_importance(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Estimate feature importance using permutation importance.
        
        Args:
            n_samples: Number of samples to use for importance estimation
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_names is None:
            return {}
        
        # This is a simplified version - full permutation importance would be better
        # but computationally expensive. We use a simple heuristic based on
        # the model's weight magnitudes.
        importance = {}
        
        if hasattr(self.model, 'coefs_'):
            # Use first layer weights as a proxy for importance
            first_layer_weights = np.abs(self.model.coefs_[0])
            # Average across all neurons in first hidden layer
            feature_importance = np.mean(first_layer_weights, axis=1)
            
            # Normalize
            total = np.sum(feature_importance)
            if total > 0:
                feature_importance = feature_importance / total
            
            for i, name in enumerate(self.feature_names):
                if i < len(feature_importance):
                    importance[name] = float(feature_importance[i])
                else:
                    importance[name] = 0.0
        
        return importance
    
    def plot_training_curve(self, output_path: Optional[str] = None):
        """Plot training loss curve."""
        if self.training_history is None or 'loss' not in self.training_history:
            print("No training history available")
            return
        
        # Create parameter string for title
        arch_str = str(self.hidden_layer_sizes).replace(' ', '')
        param_str = f"Architecture: {arch_str}, α={self.alpha:.0e}, lr={self.learning_rate_init:.4f}"
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], label='Training Loss', linewidth=2)
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title(f'MLP Training Loss Curve\n{param_str}', fontsize=13, fontweight='bold', pad=15)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray, output_path: Optional[str] = None):
        """Plot ROC curve with comparison to random classifier."""
        y_proba = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        # Create parameter string for title
        arch_str = str(self.hidden_layer_sizes).replace(' ', '')
        param_str = f"Architecture: {arch_str}, α={self.alpha:.0e}, lr={self.learning_rate_init:.4f}"
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f'MLP Classifier (AUC = {auc:.4f})', linewidth=2.5, color='#2C5F7D')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', linewidth=2, alpha=0.7)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'MLP ROC Curve\n{param_str}', fontsize=13, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray, output_path: Optional[str] = None):
        """Plot confusion matrix."""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        # Calculate accuracy for title
        accuracy = accuracy_score(y, y_pred)
        
        # Create parameter string for title
        arch_str = str(self.hidden_layer_sizes).replace(' ', '')
        param_str = f"Architecture: {arch_str}, α={self.alpha:.0e}, lr={self.learning_rate_init:.4f}"
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                   linewidths=1, linecolor='gray', square=True)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(f'MLP Confusion Matrix (Accuracy = {accuracy:.4f})\n{param_str}', 
                 fontsize=13, fontweight='bold', pad=15)
        
        # Add class labels
        plt.xticks([0.5, 1.5], ['Novel', 'Not Novel'], fontsize=11)
        plt.yticks([0.5, 1.5], ['Novel', 'Not Novel'], fontsize=11, rotation=0)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save(self, models_dir: str):
        """
        Save the model to disk.
        
        Args:
            models_dir: Directory to save the model
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        mlp_dir = models_dir / 'mlp'
        mlp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(mlp_dir / 'mlp_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(mlp_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'n_iter_no_change': self.n_iter_no_change,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(mlp_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    @classmethod
    def load(cls, models_dir: str):
        """
        Load a saved model from disk.
        
        Args:
            models_dir: Directory containing the saved model
            
        Returns:
            Loaded PatentNoveltyClassifier instance
        """
        models_dir = Path(models_dir)
        mlp_dir = models_dir / 'mlp'
        
        if not mlp_dir.exists():
            raise FileNotFoundError(f"MLP model directory not found: {mlp_dir}")
        
        # Load metadata
        with open(mlp_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            hidden_layer_sizes=tuple(metadata['hidden_layer_sizes']),
            alpha=metadata['alpha'],
            learning_rate_init=metadata['learning_rate_init'],
            max_iter=metadata['max_iter'],
            early_stopping=metadata['early_stopping'],
            n_iter_no_change=metadata['n_iter_no_change'],
            random_state=metadata['random_state']
        )
        
        # Load model
        with open(mlp_dir / 'mlp_model.pkl', 'rb') as f:
            instance.model = pickle.load(f)
        
        # Load scaler
        with open(mlp_dir / 'scaler.pkl', 'rb') as f:
            instance.scaler = pickle.load(f)
        
        # Load metadata
        instance.feature_names = metadata.get('feature_names')
        instance.training_history = metadata.get('training_history')
        
        return instance

