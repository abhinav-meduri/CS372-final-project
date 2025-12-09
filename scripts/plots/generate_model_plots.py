import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

root = Path(__file__).parent.parent.parent

def plot_confusion_matrix(y_true, y_pred, title, subtitle, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={'size': 14}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.text(0.5, 1.05, subtitle, transform=ax.transAxes, 
            ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_proba, title, subtitle, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2.5, color='#2ecc71', label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.text(0.5, 1.05, subtitle, transform=ax.transAxes, 
            ha='center', fontsize=11, style='italic')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curve(history, model_type, title, subtitle, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], linewidth=2, color='#3498db', 
             marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], linewidth=2, color='#e74c3c', 
             marker='s', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    if model_type == 'pytorch':
        ax2.plot(epochs, history['val_acc'], linewidth=2, color='#2ecc71', 
                 marker='^', markersize=4, label='Val Accuracy')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    else:
        ax2.plot(epochs, history['loss'], linewidth=2, color='#9b59b6', 
                 marker='d', markersize=4, label='Train Loss')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Training Loss', fontsize=13, fontweight='bold')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, 0.96, subtitle, ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_pytorch_plots():
    print("Generating PyTorch plots...")
    from src.app.pytorch_classifier import PyTorchPatentClassifier
    
    with open(root / "results/metrics/pytorch_metrics.json") as f:
        metrics = json.load(f)
    
    X_test = np.load(root / "data/features/test_features_v2.X.npy")
    y_test = np.load(root / "data/features/test_features_v2.y.npy")
    
    clf = PyTorchPatentClassifier(input_dim=10, hidden_dims=[256, 128], 
                                   dropout_rate=0.3, learning_rate=0.002)
    clf.model.load_state_dict(torch.load(root / "models/pytorch_nn/pytorch_model.pt", 
                                          map_location='cpu'))
    clf.model.eval()
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = metrics['accuracy']
    subtitle = f"Architecture: [256, 128], Dropout: 0.3, Accuracy: {acc:.2%}"
    output_dir = root / "results/plots/pytorch"
    
    plot_confusion_matrix(y_test, y_pred, "PyTorch Classifier: Confusion Matrix", 
                         subtitle, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, "PyTorch Classifier: ROC Curve", 
                   subtitle, output_dir / "roc_curve.png")
    
    with open(root / "models/pytorch_nn/training_history_pytorch.json") as f:
        history = json.load(f)
    
    if history and len(history.get('train_loss', [])) > 0:
        plot_training_curve(history, 'pytorch', "PyTorch Classifier: Training History", 
                           subtitle, output_dir / "training_curve.png")
    
    print(f"Saved: {output_dir}")

def generate_mlp_plots():
    print("Generating MLP plots...")
    import pickle
    
    with open(root / "results/metrics/mlp_metrics.json") as f:
        metrics = json.load(f)
    
    X_test = np.load(root / "data/features/test_features.X.npy")
    y_test = np.load(root / "data/features/test_features.y.npy")
    
    with open(root / "models/mlp/mlp_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(root / "models/mlp/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = metrics['accuracy']
    subtitle = f"Architecture: (64,), Alpha: 1e-5, LR: 0.005, Accuracy: {acc:.2%}"
    output_dir = root / "results/plots/mlp"
    
    plot_confusion_matrix(y_test, y_pred, "MLP Classifier: Confusion Matrix", 
                         subtitle, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, "MLP Classifier: ROC Curve", 
                   subtitle, output_dir / "roc_curve.png")
    
    with open(root / "models/mlp/metadata.json") as f:
        metadata = json.load(f)
    
    if 'training_history' in metadata and metadata['training_history']:
        history = metadata['training_history']
        plot_training_curve(history, 'mlp', "MLP Classifier: Training History", 
                           subtitle, output_dir / "training_curve.png")
    
    print(f"Saved: {output_dir}")

if __name__ == "__main__":
    generate_pytorch_plots()
    generate_mlp_plots()
    print("\nAll model plots generated!")
