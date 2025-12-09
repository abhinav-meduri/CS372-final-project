import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path(__file__).parent.parent.parent

with open(root / "results/metrics/baseline_comparison.json") as f:
    data = json.load(f)

models = list(data.keys())
accuracies = [data[m]['accuracy'] for m in models]
roc_aucs = [data[m]['roc_auc'] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Baseline Comparison: Accuracy and ROC-AUC', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_file = root / "results/plots/baseline/baseline_comparison.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_file}")
