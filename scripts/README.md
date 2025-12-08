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

## evaluation/
- `tuning/mlp_tuning.py` - Grid search for MLP hyperparameters
- `tuning/nn_tuning.py` - Grid search for PyTorch neural network hyperparameters
- `run_ablation_study.py` - Ablation study for both MLP and PyTorch models
- `plots/plot_ablation.py` - Generate side-by-side MLP vs PyTorch ablation comparison plot
- `plots/plot_baseline_comparison.py` - Generate baseline comparison plot
- `plots/plot_hard_negatives.py` - Generate hard negatives analysis plots
- `plots/plot_pytorch.py` - Generate PyTorch model plots

## analysis/
- `hard_negatives/analyze_hard_negatives.py` - Test model on hard negative pairs
- `hard_negatives/generate_test_hard_negatives.py` - Generate hard negatives for test set
- `hard_negatives/analyze_negatives.py` - Analyze negative pairs and generate hard negatives
- `sensitivity/length_sensitivity.py` - Analyze model sensitivity to input text length

## Usage

Run scripts from project root:
```bash
python scripts/data/compute_features.py
python scripts/training/generate_training_pairs.py
python scripts/evaluation/run_ablation_study.py
```

