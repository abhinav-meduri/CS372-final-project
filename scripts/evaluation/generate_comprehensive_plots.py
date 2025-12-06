"""
Generate comprehensive plots for all models.

Creates synthesized plots that compare:
- All model architectures
- Ablation study results
- Baseline comparisons
- Model metrics (accuracy, ROC-AUC, F1, etc.)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

# Output directory
output_dir = project_root / 'results' / 'plots'
output_dir.mkdir(parents=True, exist_ok=True)


def load_evaluation_results() -> Dict:
    """Load all evaluation results."""
    results = {}
    
    # Comprehensive evaluation
    comp_eval_path = project_root / 'results' / 'comprehensive_evaluation' / 'comprehensive_evaluation.json'
    if comp_eval_path.exists():
        with open(comp_eval_path, 'r') as f:
            results['comprehensive'] = json.load(f)
    
    # Improved model results
    improved_path = project_root / 'results' / 'improved_model' / 'enhanced_features_results.json'
    if improved_path.exists():
        with open(improved_path, 'r') as f:
            results['improved'] = json.load(f)
    
    # Ensemble results
    ensemble_path = project_root / 'results' / 'ensemble' / 'ensemble_metrics.json'
    if ensemble_path.exists():
        with open(ensemble_path, 'r') as f:
            results['ensemble'] = json.load(f)
    
    return results


def plot_model_comparison(results: Dict):
    """Plot comparison of all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract model metrics
    models_data = {}
    
    # From comprehensive evaluation
    if 'comprehensive' in results:
        comp = results['comprehensive']
        
        # Baseline comparison
        if 'baseline_comparison' in comp:
            for name, metrics in comp['baseline_comparison'].items():
                models_data[name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'f1': metrics.get('f1', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0)
                }
        
        # Model comparison
        if 'model_comparison' in comp:
            for name, metrics in comp['model_comparison'].items():
                models_data[name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'f1': metrics.get('f1', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0)
                }
    
    # From improved model results
    if 'improved' in results:
        improved = results['improved']
        for name, metrics in improved.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                models_data[name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'f1': metrics.get('f1', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0)
                }
    
    # From ensemble
    if 'ensemble' in results:
        ensemble = results['ensemble']
        if isinstance(ensemble, dict) and 'accuracy' in ensemble:
            models_data['Ensemble'] = {
                'accuracy': ensemble.get('accuracy', 0),
                'roc_auc': ensemble.get('roc_auc', 0),
                'f1': ensemble.get('f1', 0),
                'precision': ensemble.get('precision', 0),
                'recall': ensemble.get('recall', 0)
            }
    
    if not models_data:
        print("No model data found")
        return
    
    # Prepare data for plotting
    model_names = list(models_data.keys())
    accuracies = [models_data[m].get('accuracy', 0) or 0 for m in model_names]
    roc_aucs = [models_data[m].get('roc_auc', 0) or 0 for m in model_names]
    f1_scores = [models_data[m].get('f1', 0) or 0 for m in model_names]
    precisions = [models_data[m].get('precision', 0) or 0 for m in model_names]
    recalls = [models_data[m].get('recall', 0) or 0 for m in model_names]
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    colors = ['steelblue' if 'PyTorch' in name or 'pytorch' in name.lower() else 
              'green' if 'Gradient' in name or 'gradient' in name.lower() else
              'orange' if 'MLP' in name or 'mlp' in name.lower() else
              'red' if 'Random' in name or 'Majority' in name else 'gray'
              for name in model_names]
    bars = ax.barh(range(len(model_names)), accuracies, color=colors, alpha=0.7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.4, 1.0])
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=8)
    
    # Plot 2: ROC-AUC comparison
    ax = axes[0, 1]
    bars = ax.barh(range(len(model_names)), roc_aucs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel('ROC-AUC', fontsize=11)
    ax.set_title('Model ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.4, 1.0])
    
    for i, (bar, auc) in enumerate(zip(bars, roc_aucs)):
        if auc > 0:
            ax.text(auc + 0.005, i, f'{auc:.3f}', va='center', fontsize=8)
    
    # Plot 3: F1 Score comparison
    ax = axes[1, 0]
    bars = ax.barh(range(len(model_names)), f1_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title('Model F1 Score Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.4, 1.0])
    
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        if f1 > 0:
            ax.text(f1 + 0.005, i, f'{f1:.3f}', va='center', fontsize=8)
    
    # Plot 4: Precision-Recall tradeoff
    ax = axes[1, 1]
    ax.scatter(recalls, precisions, s=100, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    for i, name in enumerate(model_names):
        ax.annotate(name, (recalls[i], precisions[i]), fontsize=7, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    plt.suptitle('Comprehensive Model Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison_comprehensive.png'}")


def plot_ablation_study(results: Dict):
    """Plot ablation study results."""
    if 'comprehensive' not in results:
        print("No comprehensive evaluation results found for ablation study")
        return
    
    comp = results['comprehensive']
    if 'ablation_study' not in comp:
        print("No ablation study results found")
        return
    
    ablation = comp['ablation_study']
    
    # Extract metrics
    configs = list(ablation.keys())
    accuracies = [ablation[c].get('accuracy', 0) for c in configs]
    roc_aucs = [ablation[c].get('roc_auc', 0) for c in configs]
    f1_scores = [ablation[c].get('f1', 0) for c in configs]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy
    ax = axes[0]
    bars = ax.barh(range(len(configs)), accuracies, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Ablation Study: Accuracy', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.88, 0.92])
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.0005, i, f'{acc:.4f}', va='center', fontsize=8)
    
    # Plot 2: ROC-AUC
    ax = axes[1]
    bars = ax.barh(range(len(configs)), roc_aucs, color='green', alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel('ROC-AUC', fontsize=11)
    ax.set_title('Ablation Study: ROC-AUC', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.96, 0.975])
    
    for i, (bar, auc) in enumerate(zip(bars, roc_aucs)):
        ax.text(auc + 0.0005, i, f'{auc:.4f}', va='center', fontsize=8)
    
    # Plot 3: F1 Score
    ax = axes[2]
    bars = ax.barh(range(len(configs)), f1_scores, color='orange', alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title('Ablation Study: F1 Score', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.88, 0.92])
    
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax.text(f1 + 0.0005, i, f'{f1:.4f}', va='center', fontsize=8)
    
    plt.suptitle('Feature Ablation Study Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'ablation_study.png'}")


def plot_metrics_synthesis(results: Dict):
    """Create a single comprehensive metrics plot."""
    # Collect all model metrics
    all_models = {}
    
    if 'comprehensive' in results:
        comp = results['comprehensive']
        if 'baseline_comparison' in comp:
            all_models.update(comp['baseline_comparison'])
        if 'model_comparison' in comp:
            all_models.update(comp['model_comparison'])
    
    if 'improved' in results:
        improved = results['improved']
        for name, metrics in improved.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                all_models[name] = metrics
    
    if 'ensemble' in results:
        all_models['Ensemble'] = results['ensemble']
    
    if not all_models:
        print("No model data found for synthesis")
        return
    
    # Filter to only models with valid metrics
    valid_models = {k: v for k, v in all_models.items() 
                   if isinstance(v, dict) and 'accuracy' in v and v.get('accuracy', 0) > 0}
    
    model_names = list(valid_models.keys())
    metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
    
    # Create matrix (ensure all values are numeric)
    data = []
    for m in model_names:
        row = []
        for metric in metrics:
            val = valid_models[m].get(metric, 0)
            if val is None:
                val = 0
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)
        data.append(row)
    data = np.array(data, dtype=float)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(model_names) * 0.5)))
    
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0.4, vmax=1.0)
    
    # Set ticks
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            value = data[i, j]
            text_color = 'white' if value > 0.7 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Metric Value', fontsize=10)
    
    ax.set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_heatmap.png'}")


def main():
    print("="*60)
    print("GENERATING COMPREHENSIVE PLOTS")
    print("="*60)
    
    # Load results
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found!")
        return
    
    # Generate plots
    print("\n1. Generating model comparison plot...")
    plot_model_comparison(results)
    
    print("\n2. Generating ablation study plot...")
    plot_ablation_study(results)
    
    print("\n3. Generating metrics synthesis plot...")
    plot_metrics_synthesis(results)
    
    print("\n" + "="*60)
    print("PLOT GENERATION COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

