import json
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent.parent.parent

with open(root / "results/metrics/hard_negatives_analysis.json") as f:
    data = json.load(f)

prob_stats = data['pytorch_probability_stats']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

stats_labels = ['Min', 'Median', 'Mean', 'Max']
stats_values = [prob_stats['min'], prob_stats['median'], prob_stats['mean'], prob_stats['max']]
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars = ax1.bar(stats_labels, stats_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
ax1.set_title('Hard Negatives: Probability Statistics', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars, stats_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)

categories = ['Predicted Novel', 'Confident Not Novel', 'Uncertain']
counts = [prob_stats['pairs_predicted_novel'], 
          prob_stats['pairs_confident_not_novel'], 
          prob_stats['pairs_uncertain']]
pie_colors = ['#e74c3c', '#2ecc71', '#f39c12']

wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=pie_colors,
                                     autopct='%1.1f%%', startangle=90, 
                                     textprops={'fontsize': 11})
ax2.set_title(f'Hard Negatives Classification (n={data["num_pairs"]})', 
              fontsize=13, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
output_file = root / "results/plots/hard_negatives/hard_negatives_analysis.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_file}")
