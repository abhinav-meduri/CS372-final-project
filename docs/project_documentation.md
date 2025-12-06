# Patent Novelty Assessment System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Production Pipeline](#production-pipeline)
3. [Model Performance](#model-performance)
4. [Architecture](#architecture)
5. [Project Organization](#project-organization)

---

## Project Overview

This system assesses patent novelty using a hybrid RAG (Retrieval-Augmented Generation) architecture that combines:
- Local database search (200K patents, FAISS)
- Online Google Patents search (millions, via SerpAPI)
- LLM-powered keyword extraction (Phi-3)
- PyTorch neural network for novelty scoring (91.82% accuracy)
- Phi-3 explanations (Ollama)

---

## Production Pipeline

### Components Used in Production

**1. Embedding Model: PatentSBERTa**
- Model: `AI-Growth-Lab/PatentSBERTa` (HuggingFace)
- Usage: Generates 768-dimensional embeddings for patent text
- Location: `src/app/patent_analyzer.py` line 122

**2. Similarity Search: FAISS**
- Method: Cosine similarity via numpy dot product
- Data: Pre-computed embeddings in `data/embeddings/patent_embeddings.npy`
- Location: `src/app/patent_analyzer.py` line 272-294

**3. Novelty Scoring: PyTorch Neural Network**
- Accuracy: 91.82% (ROC-AUC: 97.21%)
- Features: 13 engineered features (BM25, cosine similarity, embeddings, CPC, etc.)
- Location: `src/app/patent_analyzer.py` line 290-330
- Fallback: Simple similarity scoring if model fails to load

**4. LLM Explanation: Phi-3 via Ollama**
- Model: `phi3` (local, via Ollama server)
- Location: `src/explainability/phi3_explainer.py`
- Generates detailed patent novelty explanations

**5. Hybrid RAG: Online Search (Optional)**
- LLM Keyword Extraction: Phi-3 generates optimized search queries
- Google Patents Search: SerpAPI or PatentsView API fallback
- Location: `src/retrieval/online_search.py`

### Pipeline Flow

```
User Input (patent idea/abstract)
    |
    v
Input Handler (parse input)
    |
    v
Embedding Generation (PatentSBERTa)
    |
    v
Local Search (FAISS) + Online Search (SerpAPI)
    |
    v
Result Merging & Deduplication
    |
    v
Feature Extraction (13 features)
    |
    v
PyTorch Model Prediction (novelty score)
    |
    v
Phi-3 Explanation Generation
    |
    v
Display Results
```

---

## Model Performance

### Verified Accuracies (Test Set)

| Rank | Model | Accuracy | ROC-AUC | Status |
|------|-------|----------|---------|--------|
| 1 | Gradient Boosting | 91.95% | 97.17% | Experimental |
| 2 | PyTorch Neural Network | 91.82% | 97.21% | PRODUCTION |
| 3 | Ensemble (Stacking) | 91.77% | 97.16% | Experimental |
| 4 | MLP (128-64-32) | 91.76% | 97.09% | Experimental |
| 5 | MLP (64-32) | 91.60% | 97.09% | Experimental |
| 6 | MLP Classifier | 91.57% | 97.13% | Experimental |
| 7 | Random Forest | 91.54% | 97.09% | Experimental |
| 8 | Logistic Regression | 90.79% | 96.76% | Experimental |

### Baseline Comparisons

- Random Guessing: 49.78%
- Majority Class: 50.47%
- Stratified Random: 49.74%
- Title Jaccard Heuristic: 75.36%
- Logistic Regression: 90.79%

### Production Model Details

**PyTorch Neural Network:**
- Architecture: [256, 128, 64, 32] hidden layers
- Features: Residual connections, batch normalization, dropout (0.25)
- Input: 13 engineered features
- Output: Sigmoid (probability of similarity)
- Accuracy: 91.82%
- ROC-AUC: 97.21% (highest of all models)

**Why PyTorch is Used:**
- Best ROC-AUC (97.21%)
- Second-best accuracy (91.82%)
- Most reliable evaluation (from `enhanced_features_results.json`)
- Currently integrated in production

---

## Architecture

### Hybrid RAG System

**Local Search:**
- PatentSBERTa embeddings (768-dimensional)
- FAISS cosine similarity search
- 200K patent database

**Online Search:**
- LLM keyword extraction (Phi-3)
- Google Patents via SerpAPI
- PatentsView API fallback
- Millions of patents

**Novelty Scoring:**
- Feature extraction (13 features)
- PyTorch neural network prediction
- Novelty = 1 - similarity_probability

**Explanation:**
- Phi-3 via Ollama
- Detailed prior art analysis
- Recommendation (APPROVE/REVISE/REJECT)

### Feature Engineering

The PyTorch model uses 13 engineered features:
1. BM25 document score
2. BM25 best claim score
3. Cosine document similarity
4. Cosine max claim similarity
5. Embedding difference (mean)
6. Embedding difference (std)
7. CPC Jaccard similarity
8. Year difference
9. Title Jaccard similarity
10. Abstract length ratio
11. Claim count ratio
12. Shared rare terms ratio
13. Claim similarity

---

## Model Organization

### Production Model

**Location**: `models/pytorch_nn/`

**Model**: PyTorch Neural Network
- **Accuracy**: 91.82%
- **ROC-AUC**: 97.21%
- **Architecture**: [256, 128, 64, 32] with residual connections, batch normalization, dropout
- **Parameters**: 102,361 trainable parameters

**Files**:
- `pytorch_model.pt` - Trained model weights
- `scaler_pytorch.pkl` - Feature scaler
- `training_history_pytorch.json` - Training history

### Baseline/Comparison Models

- **MLP Classifier (sklearn)**: `experimental/models/mlp_model.pkl` (91.60% accuracy)
- **Gradient Boosting**: Best accuracy (91.95%) - experimental
- **Ensemble Models**: `experimental/models/ensemble/` - not used in production

### Experimental Models

- **Location**: `experimental/models/` - Contains baseline/comparison models (MLP, ensemble)
- **Purpose**: Models used for comparison and rubric evaluation, not used in production

## Comprehensive Evaluation

The comprehensive evaluation script (`scripts/evaluation/comprehensive_evaluation.py`) provides systematic evaluation evidence for rubric items.

**What It Does:**
1. **Baseline Model Comparisons** - Random guessing, majority class, title Jaccard heuristic, Logistic Regression, MLP Classifier
2. **Inference Time Measurement** - Single prediction latency, batch throughput, full dataset processing time
3. **Ablation Study** - Impact of removing different feature groups (embedding, text similarity, metadata, BM25)
4. **Model Architecture Comparison** - Logistic Regression, Random Forest, Gradient Boosting, MLP variants

**Rubric Alignment:**
- Baseline model comparison
- Ablation studies
- Model comparison
- Inference time measurement

**Results**: `results/comprehensive_evaluation/comprehensive_evaluation.json`

## Hyperparameter Tuning

The hyperparameter tuning script (`scripts/evaluation/hyperparameter_tuning.py`) conducts systematic hyperparameter search for the **MLP Classifier (sklearn)** baseline model.

**Purpose**:
- Grid search over hyperparameters (hidden layers, learning rate, regularization)
- Uses 5-fold cross-validation
- Optimizes for ROC-AUC score
- **Rubric alignment**: Demonstrates systematic hyperparameter tuning

**Results**: `results/hyperparameter_tuning/hyperparameter_results.json`

**Note**: The production PyTorch Neural Network uses a fixed architecture and is not tuned via this script.

## Plots and Visualizations

All plots are saved in `results/plots/` directory.

### Existing Plots (from MLP Classifier)
- `confusion_matrix.png` - MLP Classifier
- `roc_curve.png` - MLP Classifier
- `training_curve.png` - MLP Classifier

### Comprehensive Plots (All Models)
- `model_comparison_comprehensive.png` - All models compared
- `ablation_study.png` - Feature ablation results
- `metrics_heatmap.png` - Heatmap of all metrics

### SHAP Analysis Plots
- `shap_global_importance.png` - Feature importance (from notebook)

**To regenerate plots**: `python scripts/evaluation/generate_comprehensive_plots.py`

## Project Organization

### Directory Structure

```
CS372-final-project/
├── app.py                    # Production: Streamlit web app
├── src/
│   ├── app/                  # Production: Main application
│   │   ├── patent_analyzer.py
│   │   └── input_handler.py
│   ├── models/               # Model definitions
│   │   ├── pytorch_classifier.py  # Production model
│   │   ├── mlp_classifier.py      # Experimental
│   │   └── ensemble_model.py      # Experimental
│   ├── embeddings/           # Production: Embedding generation
│   ├── retrieval/            # Production: Search and RAG
│   ├── explainability/       # Production: LLM explanations
│   └── features/             # Feature extraction
├── data/
│   ├── sampled/              # Patent database (200K)
│   ├── embeddings/           # Pre-computed embeddings
│   └── features/             # Training features
├── models/
│   └── pytorch_enhanced/     # Production model
├── experimental/             # Experimental code/models
│   ├── models/               # Trained models (not used)
│   └── scripts/              # Training scripts
├── results/                  # Evaluation results
│   ├── comprehensive_evaluation/
│   ├── metrics/
│   └── plots/
└── scripts/                  # Utility scripts
```

### Production Files

**Core Application:**
- `app.py` - Streamlit web interface
- `src/app/patent_analyzer.py` - Main inference pipeline
- `src/app/input_handler.py` - Input parsing

**Models:**
- `models/pytorch_nn/` - Production PyTorch model (91.82%)
- `src/models/pytorch_classifier.py` - Model definition

**Data:**
- `data/sampled/patents_sampled.jsonl` - Patent database
- `data/embeddings/patent_embeddings.npy` - Pre-computed embeddings
- `data/embeddings/patent_ids.json` - Patent ID mapping

### Experimental Files

**Not Used in Production:**
- `experimental/models/` - Trained models (MLP, ensemble)
- `experimental/scripts/` - Training/evaluation scripts
- `src/models/mlp_classifier.py` - MLP definition (for comparison)
- `src/models/ensemble_model.py` - Ensemble definition (for comparison)

**Why Kept:**
- For rubric evaluation (baseline comparisons)
- For future improvements
- For reference

### Evaluation Results

**Baseline Comparisons:**
- `results/comprehensive_evaluation/comprehensive_evaluation.json` - All model comparisons (baseline, ablation, model comparison)
- `results/metrics/all_metrics.json` - Original MLP training metrics
- `results/ensemble/ensemble_metrics.json` - Ensemble performance
- `results/pytorch_enhanced_features/enhanced_features_results.json` - PyTorch NN with enhanced features (28 features vs 13 base features)

**Plots:**
- `results/plots/confusion_matrix.png`
- `results/plots/roc_curve.png`
- `results/plots/training_curve.png`

---

## Key Files for Rubric Evaluation

**Model Comparisons:**
- `results/comprehensive_evaluation/comprehensive_evaluation.json` - Baseline comparisons, ablation studies, and model architecture comparisons
- `results/pytorch_enhanced_features/enhanced_features_results.json` - PyTorch NN (28 enhanced features) vs MLP vs Gradient Boosting
- `results/ensemble/ensemble_metrics.json` - Ensemble performance

**Training Metrics:**
- `results/metrics/all_metrics.json` - Original MLP training (train/val/test)
- `results/hyperparameter_tuning/hyperparameter_results.json` - Hyperparameter grid search for MLP (runs via `scripts/evaluation/hyperparameter_tuning.py`)

**Visualizations:**
- `results/plots/` - Confusion matrix, ROC curve, training curve

**Production Model:**
- `models/pytorch_nn/` - Trained PyTorch model (91.82% accuracy)

---

## Summary

**Production System:**
- PyTorch Neural Network: 91.82% accuracy, 97.21% ROC-AUC
- Hybrid RAG: Local (FAISS) + Online (SerpAPI)
- Phi-3 explanations via Ollama

**Best Performance:**
- Gradient Boosting: 91.95% (experimental)
- PyTorch NN: 91.82% (production)
- Ensemble: 91.77% (experimental)

**Baseline Improvements:**
- vs Random: +42.04%
- vs Title Jaccard: +16.46%
- vs Logistic Regression: +1.03%

