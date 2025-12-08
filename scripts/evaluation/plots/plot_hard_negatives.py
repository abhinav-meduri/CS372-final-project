"""
Generate plots for hard negative analysis results.
The model predicts citation relationships based on PatentsView ground truth.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11

def plot_hard_negatives_comparison(results_file, save_dir):
    """Plot comparison of model performance on hard negatives."""
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    pytorch_metrics = results['pytorch']
    prob_stats = results.get('pytorch_probability_stats', {})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Hard Negatives Analysis: PyTorch Model Performance on Challenging Cases', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    ax1 = axes[0]
    pytorch_cm = np.array(pytorch_metrics['confusion_matrix'])
    
    if pytorch_cm.shape == (1, 1):
        cm_display = np.zeros((2, 2))
        cm_display[0, 0] = pytorch_cm[0, 0]  # All true negatives
    else:
        cm_display = pytorch_cm
    
    labels = ['No Citation\n(Label 0)', 'Has Citation\n(Label 1)']
    im = ax1.imshow(cm_display, cmap='Blues', aspect='auto', vmin=0, vmax=cm_display.max())
    ax1.set_title('PyTorch Model: Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    
    thresh = cm_display.max() / 2. if cm_display.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            val = int(cm_display[i, j])
            if val > 0:
                text_color = "white" if cm_display[i, j] > thresh else "black"
                ax1.text(j, i, f'{val}', ha="center", va="center",
                       color=text_color, fontsize=18, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = axes[1]
    
    metrics = ['Accuracy', 'Mean Prob', 'Max Prob']
    values = [
        pytorch_metrics['accuracy'],
        prob_stats.get('mean', 0),
        prob_stats.get('max', 0)
    ]
    
    bars = ax2.bar(metrics, values, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold (0.5)')
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('PyTorch Model: Performance Metrics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3 = axes[2]
    
    stats_names = ['Min', 'Mean', 'Median', 'Max']
    stats_values = [
        prob_stats.get('min', 0),
        prob_stats.get('mean', 0),
        prob_stats.get('median', 0),
        prob_stats.get('max', 0)
    ]
    
    bars = ax3.bar(stats_names, stats_values, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold (0.5)')
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title('PyTorch Model: Prediction Probabilities', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, max(max(stats_values) * 1.3, 0.1)])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, stats_values):
        ax3.text(bar.get_x() + bar.get_width()/2., val + 0.002,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'hard_negatives_analysis.png', dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved plot to {save_dir / 'hard_negatives_analysis.png'}")

def main():
    results_file = Path("results/analysis/hard_negatives_analysis.json")
    plots_dir = Path("results/plots/hard_negatives")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run scripts/analysis/analyze_hard_negatives.py first.")
        return
    
    print(f"Loading results from {results_file}...")
    plot_hard_negatives_comparison(results_file, plots_dir)
    
    print("Hard negative plots generated.")

if __name__ == '__main__':
    main()
