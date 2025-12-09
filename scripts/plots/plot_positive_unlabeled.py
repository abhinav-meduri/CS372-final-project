import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent.parent.parent
output_dir = root / "results/plots/positive-unlabeled"
output_dir.mkdir(parents=True, exist_ok=True)

with open(root / "results/metrics/recall_mrr.json") as f:
    recall_data = json.load(f)

with open(root / "results/metrics/additional_metrics.json") as f:
    additional = json.load(f)

with open(root / "results/metrics/pytorch_metrics.json") as f:
    pytorch = json.load(f)

k_values = [int(k) for k in recall_data['recall_at_k'].keys()]
recall_values = list(recall_data['recall_at_k'].values())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, recall_values, marker='o', linewidth=2.5, 
        markersize=8, color='#3498db', label='Recall@K')
ax.set_xlabel('K (Number of Retrieved Patents)', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax.set_title('Recall@K: Citation-Based Evaluation', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)

for k, r in zip(k_values, recall_values):
    ax.text(k, r + 0.02, f'{r:.3f}', ha='center', fontsize=10)

mrr_text = f"MRR: {recall_data['mrr']:.4f}\nQueries: {recall_data['queries_with_positive']}"
ax.text(0.98, 0.02, mrr_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "recall_mrr.png", dpi=300, bbox_inches='tight')
plt.close()

precision_curve = np.array(additional['pr_curve']['precision'])
recall_curve = np.array(additional['pr_curve']['recall'])
pr_auc = additional['pr_auc']

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(recall_curve, precision_curve, linewidth=2.5, color='#2ecc71', 
        label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(output_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
plt.close()

prob_true = np.array(additional['calibration']['prob_true'])
prob_pred = np.array(additional['calibration']['prob_pred'])
ece = additional['ece']
brier = pytorch['brier_score']

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(prob_pred, prob_true, marker='o', linewidth=2.5, markersize=8, 
        color='#9b59b6', label='Model Calibration')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('True Probability', fontsize=12, fontweight='bold')
ax.set_title('Calibration Curve', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

calib_text = f"ECE: {ece:.4f}\nBrier Score: {brier:.4f}"
ax.text(0.02, 0.98, calib_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "calibration.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_dir}/recall_mrr.png")
print(f"Saved: {output_dir}/pr_curve.png")
print(f"Saved: {output_dir}/calibration.png")
