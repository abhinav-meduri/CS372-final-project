"""
Generate side-by-side ablation study plot comparing MLP and PyTorch models.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Set style - neutral and professional
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Output directory
output_dir = project_root / 'results' / 'plots' / 'ablation'
output_dir.mkdir(parents=True, exist_ok=True)


def get_model_architecture():
    """Load model architecture information."""
    mlp_arch = "MLP: [64]"
    pytorch_arch = "PyTorch: [128, 64, 32]"
    
    # Try to load from metadata
    mlp_metadata_path = project_root / 'models' / 'mlp' / 'metadata.json'
    if mlp_metadata_path.exists():
        with open(mlp_metadata_path, 'r') as f:
            mlp_meta = json.load(f)
            hidden = mlp_meta.get('hidden_layer_sizes', [64])
            mlp_arch = f"MLP: {hidden}"
    
    pytorch_results_path = project_root / 'models' / 'pytorch_nn' / 'training_results.json'
    if pytorch_results_path.exists():
        with open(pytorch_results_path, 'r') as f:
            pytorch_data = json.load(f)
            # PyTorch architecture is typically [128, 64, 32] based on training
            pytorch_arch = "PyTorch: [128, 64, 32]"
    
    return mlp_arch, pytorch_arch


def regenerate_ablation_plot():
    """Generate side-by-side ablation study plot comparing MLP and PyTorch."""
    
    # Load ablation results
    ablation_path = project_root / 'results' / 'ablation_study' / 'ablation_results.json'
    
    if not ablation_path.exists():
        print(f"Error: Ablation results not found at {ablation_path}")
        return
    
    with open(ablation_path, 'r') as f:
        ablation_data = json.load(f)
    
    mlp_data = ablation_data.get('mlp', {})
    pytorch_data = ablation_data.get('pytorch', {})
    
    if not mlp_data or not pytorch_data:
        print("Error: Missing MLP or PyTorch ablation data")
        return
    
    # Get architecture info
    mlp_arch, pytorch_arch = get_model_architecture()
    
    # Extract configurations (same for both models)
    configs = list(mlp_data.keys())
    if 'All Features' in configs:
        configs.remove('All Features')
        configs = ['All Features'] + sorted(configs)
    
    # Extract metrics for both models
    mlp_accuracies = [mlp_data[c].get('accuracy', 0) for c in configs]
    mlp_roc_aucs = [mlp_data[c].get('roc_auc', 0) for c in configs]
    mlp_f1_scores = [mlp_data[c].get('f1', 0) for c in configs]
    
    pytorch_accuracies = [pytorch_data[c].get('accuracy', 0) for c in configs]
    pytorch_roc_aucs = [pytorch_data[c].get('roc_auc', 0) for c in configs]
    pytorch_f1_scores = [pytorch_data[c].get('f1', 0) for c in configs]
    
    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Define neutral color scheme
    base_color = '#4A90A4'  # Neutral teal-blue
    highlight_color = '#2C5F7D'  # Darker blue for "All Features"
    light_color = '#7FB3C3'  # Lighter blue for others
    
    # Create 3 rows x 2 columns (MLP left, PyTorch right)
    # Row 1: Accuracy
    # Row 2: ROC-AUC
    # Row 3: F1 Score
    
    metrics_data = [
        ('Accuracy', mlp_accuracies, pytorch_accuracies, (0.84, 0.92)),
        ('ROC-AUC', mlp_roc_aucs, pytorch_roc_aucs, (0.92, 0.975)),
        ('F1 Score', mlp_f1_scores, pytorch_f1_scores, (0.81, 0.92))
    ]
    
    for row_idx, (metric_name, mlp_vals, pytorch_vals, ylim) in enumerate(metrics_data):
        # MLP plot (left column)
        ax_mlp = plt.subplot(3, 2, row_idx * 2 + 1)
        mlp_colors = [highlight_color if c == 'All Features' else light_color for c in configs]
        bars_mlp = ax_mlp.barh(range(len(configs)), mlp_vals, color=mlp_colors, 
                               alpha=0.8, edgecolor='#1a1a1a', linewidth=0.8)
        ax_mlp.set_yticks(range(len(configs)))
        ax_mlp.set_yticklabels(configs, fontsize=9)
        ax_mlp.set_xlabel(metric_name, fontsize=11, fontweight='bold')
        ax_mlp.set_title(f'{mlp_arch}', fontsize=12, fontweight='bold', pad=10)
        ax_mlp.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax_mlp.set_xlim(ylim)
        ax_mlp.spines['top'].set_visible(False)
        ax_mlp.spines['right'].set_visible(False)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars_mlp, mlp_vals)):
            ax_mlp.text(val + (ylim[1] - ylim[0]) * 0.005, i, f'{val:.4f}', 
                      va='center', fontsize=8, fontweight='bold', color='#1a1a1a')
        
        # PyTorch plot (right column)
        ax_pytorch = plt.subplot(3, 2, row_idx * 2 + 2)
        pytorch_colors = [highlight_color if c == 'All Features' else light_color for c in configs]
        bars_pytorch = ax_pytorch.barh(range(len(configs)), pytorch_vals, color=pytorch_colors,
                                      alpha=0.8, edgecolor='#1a1a1a', linewidth=0.8)
        ax_pytorch.set_yticks(range(len(configs)))
        ax_pytorch.set_yticklabels(configs, fontsize=9)
        ax_pytorch.set_xlabel(metric_name, fontsize=11, fontweight='bold')
        ax_pytorch.set_title(f'{pytorch_arch}', fontsize=12, fontweight='bold', pad=10)
        ax_pytorch.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax_pytorch.set_xlim(ylim)
        ax_pytorch.spines['top'].set_visible(False)
        ax_pytorch.spines['right'].set_visible(False)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars_pytorch, pytorch_vals)):
            ax_pytorch.text(val + (ylim[1] - ylim[0]) * 0.005, i, f'{val:.4f}',
                           va='center', fontsize=8, fontweight='bold', color='#1a1a1a')
    
    # Main title
    plt.suptitle('Feature Ablation Study: MLP vs PyTorch Neural Network', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    output_path = output_dir / 'ablation_study_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved ablation study comparison plot to: {output_path}")
    print(f"  MLP configurations: {len(configs)}")
    print(f"  PyTorch configurations: {len(configs)}")
    print(f"  MLP full model: Accuracy={mlp_accuracies[0]:.4f}, ROC-AUC={mlp_roc_aucs[0]:.4f}, F1={mlp_f1_scores[0]:.4f}")
    print(f"  PyTorch full model: Accuracy={pytorch_accuracies[0]:.4f}, ROC-AUC={pytorch_roc_aucs[0]:.4f}, F1={pytorch_f1_scores[0]:.4f}")


if __name__ == '__main__':
    print("="*70)
    print("GENERATING ABLATION STUDY COMPARISON PLOT")
    print("="*70)
    regenerate_ablation_plot()
    print("="*70)
    print("COMPLETE")
    print("="*70)

