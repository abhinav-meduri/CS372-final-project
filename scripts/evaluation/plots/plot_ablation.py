"""Summarize ablation results for MLP and PyTorch models."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def get_model_architecture():
    mlp_arch = "MLP: [64]"
    pytorch_arch = "PyTorch: [256, 128]"
    
    # Try to load from metadata
    mlp_metadata_path = project_root/'models'/'mlp'/'metadata.json'
    if mlp_metadata_path.exists():
        with open(mlp_metadata_path, 'r') as f:
            mlp_meta = json.load(f)
            hidden = mlp_meta.get('hidden_layer_sizes', [64])
            mlp_arch = f"MLP: {hidden}"
    
    # Check pytorch metrics for current architecture
    pytorch_metrics_path = project_root/'results'/'pytorch_nn'/'pytorch_metrics.json'
    if pytorch_metrics_path.exists():
        with open(pytorch_metrics_path, 'r') as f:
            pytorch_data = json.load(f)
            hidden_dims = pytorch_data.get('hyperparameters', {}).get('hidden_dims', [256, 128])
            pytorch_arch = f"PyTorch: {hidden_dims}"
    
    return mlp_arch, pytorch_arch


def summarize_ablation():
    ablation_path = project_root/"results"/"analysis"/"ablation_study"/"ablation_results.json"
    if not ablation_path.exists():
        print(f"Ablation results not found: {ablation_path}")
        return

    with ablation_path.open("r") as f:
        ablation_data = json.load(f)

    mlp_data = ablation_data.get("mlp", {})
    pytorch_data = ablation_data.get("pytorch", {})
    if not mlp_data or not pytorch_data:
        print("Missing MLP or PyTorch ablation data")
        return

    configs = list(mlp_data.keys())
    if "All Features" in configs:
        configs.remove("All Features")
        configs = ["All Features"] + sorted(configs)

    def metrics_for(model_data, name):
        acc = [model_data[c].get("accuracy", 0) for c in configs]
        roc = [model_data[c].get("roc_auc", 0) for c in configs]
        f1 = [model_data[c].get("f1", 0) for c in configs]
        print(f"\n{name} ({len(configs)} configs)")
        for c, a, r, f in zip(configs, acc, roc, f1):
            print(f"  {c}: acc={a:.4f} roc={r:.4f} f1={f:.4f}")

    mlp_arch, pyt_arch = get_model_architecture()
    print(f"Architectures -> {mlp_arch}, {pyt_arch}")
    metrics_for(mlp_data, "MLP")
    metrics_for(pytorch_data, "PyTorch")


if __name__ == "__main__":
    summarize_ablation()

