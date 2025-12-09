import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent.parent.parent
data_dir = root / "results/analysis/input_length"
output_dir = root / "results/plots/input_length"
output_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "sensitivity_results.json") as f:
    sensitivity = json.load(f)

with open(data_dir / "length_statistics.json") as f:
    stats = json.load(f)

with open(data_dir / "embedding_stability.json") as f:
    stability = json.load(f)

categories = list(stats['length_distribution'].keys())
counts = list(stats['length_distribution'].values())

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, counts, color='#3498db', alpha=0.7, edgecolor='black')
ax.set_xlabel('Abstract Length Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Patents', fontsize=12, fontweight='bold')
ax.set_title('Patent Abstract Length Distribution', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "length_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

category_keys = list(sensitivity['by_length_category'].keys())
accuracies = [sensitivity['by_length_category'][k]['accuracy'] for k in category_keys]
roc_aucs = [sensitivity['by_length_category'][k]['roc_auc'] for k in category_keys]

x = np.arange(len(category_keys))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
               color='#2ecc71', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', 
               color='#e74c3c', alpha=0.7, edgecolor='black')

ax.set_xlabel('Abstract Length Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance by Abstract Length', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(category_keys)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "length_categories.png", dpi=300, bbox_inches='tight')
plt.close()

lengths = stability['sample_lengths']
mean_dists = stability['mean_distances']
std_dists = stability['std_distances']

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(lengths, mean_dists, yerr=std_dists, marker='o', linewidth=2,
            markersize=8, color='#9b59b6', capsize=5, capthick=2,
            label='Mean ± Std Distance')
ax.set_xlabel('Abstract Length (characters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cosine Distance from Full Abstract', fontsize=12, fontweight='bold')
ax.set_title('Embedding Stability vs Abstract Length', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / "embedding_stability.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_dir}/length_distribution.png")
print(f"Saved: {output_dir}/length_categories.png")
print(f"Saved: {output_dir}/embedding_stability.png")
