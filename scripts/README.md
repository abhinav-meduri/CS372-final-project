# Scripts Organization

Scripts are organized into subfolders by purpose:

## data/
- `build_bm25_index.py` - Build BM25 lexical search index
- `generate_embeddings.py` - Generate PatentSBERTa embeddings
- `compute_features.py` - Extract features for training pairs
- `add_citation_features.py` - Add citation-based features
- `add_claim_features.py` - Add claim-level features
- `add_claim_features_fast.py` - Fast version of claim features

## training/
- `extract_citation_pairs.py` - Extract citation pairs from PatentsView data
- `generate_training_pairs.py` - Generate positive/negative training pairs
- `sample_diverse_patents.py` - Sample diverse patents for dataset
- `train_phi3_lora.py` - Fine-tune Phi-3 with LoRA (experimental)

## evaluation/
- `comprehensive_evaluation.py` - Baseline comparisons, ablation studies, model comparisons
- `hyperparameter_tuning.py` - Grid search for MLP hyperparameters
- `generate_comprehensive_plots.py` - Generate all evaluation plots

## analysis/
- `analyze_negatives.py` - Analyze negative pairs and generate hard negatives

## Usage

Run scripts from project root:
```bash
python scripts/data/compute_features.py
python scripts/training/generate_training_pairs.py
python scripts/evaluation/comprehensive_evaluation.py
```

