# Scripts Organization

Scripts are organized into subfolders by purpose:

## data/
- `generate_embeddings.py` - Generate PatentSBERTa embeddings
- `compute_features.py` - Extract features for training pairs
- `add_citation_features.py` - Add citation-based features
- `add_claim_features.py` - Add claim-level features (optimized with batch processing)

## training/
- `extract_citation_pairs.py` - Extract citation pairs from PatentsView data
- `generate_training_pairs.py` - Generate positive/negative training pairs
- `sample_diverse_patents.py` - Sample diverse patents for dataset
- `train_phi3_lora.py` - Fine-tune Phi-3 with LoRA (experimental)

## evaluation/
- `run_pytorch_ablation.py` - Ablation study for PyTorch model
- `mlp_tuning.py` - Grid search for MLP hyperparameters
- `nn_tuning.py` - Grid search for PyTorch neural network hyperparameters
- `run_ablation_study.py` - Ablation study for both MLP and PyTorch models
- `plots/plot_ablation.py` - Generate side-by-side MLP vs PyTorch ablation comparison plot

## analysis/
- `analyze_negatives.py` - Analyze negative pairs and generate hard negatives

## Usage

Run scripts from project root:
```bash
python scripts/data/compute_features.py
python scripts/training/generate_training_pairs.py
python scripts/evaluation/run_ablation_study.py
```

