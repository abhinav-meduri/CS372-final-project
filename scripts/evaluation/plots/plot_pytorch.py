"""
Update PyTorch Model Plots

Generates confusion matrix, ROC curve, and training curve plots
for the optimized PyTorch model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

from src.models.pytorch_classifier import PyTorchPatentClassifier

def plot_confusion_matrix(y_true, y_pred, save_path, accuracy):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Novel', 'Not Novel'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title(f'PyTorch Classifier - Confusion Matrix\nArchitecture: [256, 128], Dropout: 0.3, LR: 0.002, Accuracy: {accuracy:.2f}%', 
                 fontsize=12, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true, y_proba, save_path, roc_auc):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, 
            label=f'PyTorch Classifier (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
            label='Random Classifier (AUC = 0.5000)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('PyTorch Classifier - ROC Curve\nArchitecture: [256, 128], Dropout: 0.3, LR: 0.002', 
                 fontsize=12, pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved ROC curve to {save_path}")

def plot_training_curve(training_history, save_path):
    """Plot training curve."""
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, training_history['train_loss'], 'o-', color='#2E86AB', 
             label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, training_history['val_loss'], 's-', color='#A23B72', 
             label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss\nArchitecture: [256, 128], Dropout: 0.3, LR: 0.002', 
                  fontsize=12, pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, training_history['val_acc'], 'o-', color='#06A77D', 
             label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Validation Accuracy\nArchitecture: [256, 128], Dropout: 0.3, LR: 0.002', 
                  fontsize=12, pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved training curve to {save_path}")

def main():
    print("UPDATING PYTORCH MODEL PLOTS")
    
    features_dir = Path("data/features")
    model_dir = Path("models/pytorch_nn")
    plots_dir = Path("results/plots/pytorch")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading model and data...")
    model = PyTorchPatentClassifier()
    model.load(model_dir)
    
    X_test = np.load(features_dir / "test_features_v2.X.npy")
    y_test = np.load(features_dir / "test_features_v2.y.npy")
    
    # Remove BM25 and CPC features (indices 0, 1, 6) to match 10-feature model
    if X_test.shape[1] == 13:
        indices_to_remove = [0, 1, 6]
        indices_to_keep = [i for i in range(13) if i not in indices_to_remove]
        X_test = X_test[:, indices_to_keep]
        print(f"   Filtered features from 13 to 10")
    
    print(f"   Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    
    print("\n2. Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    test_metrics = model.evaluate(X_test, y_test)
    accuracy = test_metrics['accuracy']
    roc_auc = test_metrics['roc_auc']
    
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Test ROC-AUC: {roc_auc:.6f}")
    
    print("\n3. Loading training history...")
    with open(model_dir / "training_history_pytorch.json", "r") as f:
        training_history = json.load(f)
    
    print("\n4. Generating plots...")
    plot_confusion_matrix(y_test, y_pred, plots_dir / "confusion_matrix.png", accuracy * 100)
    plot_roc_curve(y_test, y_proba, plots_dir / "roc_curve.png", roc_auc)
    plot_training_curve(training_history, plots_dir / "training_curve.png")
    
    print("PLOTS UPDATED SUCCESSFULLY")
    print(f"\nAll plots saved to: {plots_dir}")

if __name__ == '__main__':
    main()


