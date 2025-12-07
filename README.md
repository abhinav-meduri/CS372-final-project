# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines PatentSBERTa embeddings, FAISS indexing, PyTorch neural network classification, and LLM-based explainability to assess patent novelty.

## What it Does

This system helps researchers and inventors quickly assess the novelty of patent applications by comparing them against a corpus of 200,000 USPTO patents (2021-2025). Given a query patent (title, abstract, and claims), the system performs hybrid retrieval combining local FAISS similarity search with online Google Patents search (via SerpAPI). LLM-powered keyword extraction (Phi-3) generates optimized search terms. A trained PyTorch neural network scores each candidate based on 10 engineered features (reduced from 13 via ablation study) including embedding similarity, text overlap metrics, and metadata features. Finally, Phi-3 LLM generates human-readable explanations citing specific evidence from the prior art, helping users understand why certain patents may pose novelty concerns.

## Quick Start

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd CS372-final-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required data files (see Installation section)
# Required: patents_sampled.jsonl, embeddings, and trained models

# 4. Install and start Ollama (for LLM explanations)
brew install ollama
brew services start ollama
ollama pull phi3

# 5. Run the application
streamlit run app.py
```

## Installation

### Prerequisites
- Python 3.9 or higher
- 16GB RAM recommended
- ~50GB disk space for data and models

### Setup Steps

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama (for LLM explanations)**
   ```bash
   # macOS
   brew install ollama
   brew services start ollama
   ollama pull phi3
   
   # Linux/Windows: See https://ollama.ai
   ```

4. **Download Required Data Files**
   
   The application requires pre-processed data files that are too large for GitHub. You need to download:
   
   **Required files:**
   - `data/sampled/patents_sampled.jsonl` (~3.8GB) - 200K patent database
   - `data/embeddings/patent_embeddings.npy` - Pre-computed embeddings
   - `data/embeddings/patent_ids.json` - Patent ID mapping
   - `data/features/feature_names_v2.json` - Feature names (10 features, included in repo)
   - `models/pytorch_nn/pytorch_model.pt` - Trained PyTorch model (10 features)
   - `models/pytorch_nn/scaler_pytorch.pkl` - Feature scaler
   - `models/pytorch_nn/training_history_pytorch.json` - Training metadata
   
   **Options:**
   - Download from a shared location (Google Drive, Dropbox, etc.)
   - Generate from raw PatentsView data using scripts in `scripts/data/`
   - Contact repository maintainer for data access
   
   **Note:** Without these files, the application will not run. The system needs the patent database and pre-computed embeddings for similarity search.
   
   **Important:** The model must be retrained with 10 features (BM25 and CPC features removed based on ablation study). If you have an older model trained with 13 features, you'll need to retrain it using the training scripts in `scripts/training/`.

5. **Configure API Keys (Optional - for online search)**
   Create a `.env` file:
   ```bash
   SERPAPI_KEY=your_key_here
   ```

6. **Run Application**
   ```bash
   streamlit run app.py
   ```

For detailed data setup and model training instructions, see `docs/project_documentation.md`.

## Video Links

- **Demo Video:** [Link TBD]
- **Technical Walkthrough:** [Link TBD]

## Evaluation

### Model Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Ensemble Model (Best)** | **91.77%** | **92.63%** | **90.60%** | **0.9160** | **0.9716** |
| MLP Classifier | 91.48% | 93.00% | 89.54% | 0.9124 | 0.9713 |
| PyTorch Neural Network | 90.71% | 93.23% | 87.61% | 0.9033 | 0.9682 |
| Logistic Regression | 90.93% | 92.45% | 88.99% | 0.9063 | 0.9674 |
| Cosine Similarity (heuristic) | 84.27% | 81.23% | 86.42% | 0.8353 | 0.9170 |
| Random Guessing | 49.78% | 49.80% | 49.80% | 0.4950 | 0.5032 |
| Majority Class | 50.47% | 0.00% | 0.00% | 0.0000 | 0.5000 |

**Note:** The Ensemble Model combines MLP and PyTorch classifiers using stacking with probability calibration, achieving the best overall performance.

### Individual Model Details

#### Ensemble Model (Production - Best Performing)
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 91.77% |
| **Precision** | 92.63% |
| **Recall** | 90.60% |
| **F1 Score** | 0.9160 |
| **ROC-AUC** | 0.9716 |
| **Brier Score** | 0.0625 |

#### MLP Classifier
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 91.48% |
| **Precision** | 93.00% |
| **Recall** | 89.54% |
| **F1 Score** | 0.9124 |
| **ROC-AUC** | 0.9713 |
| **Brier Score** | 0.0639 |

#### PyTorch Neural Network
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 90.71% |
| **Precision** | 93.23% |
| **Recall** | 87.61% |
| **F1 Score** | 0.9033 |
| **ROC-AUC** | 0.9682 |

### Ablation Study and Feature Selection

**Initial Feature Set:** The model was initially trained with 13 engineered features covering embedding similarity, text overlap, metadata, and lexical matching.

**Ablation Study Results (13-feature model):**

| Configuration | Accuracy | ROC-AUC | F1 Score | Δ AUC |
|---------------|----------|---------|----------|-------|
| Full Model (13 features) | 91.53% | 0.9713 | 0.9136 | -- |
| Without Claim Features | 87.24% | 0.9445 | 0.8667 | -0.0268 |
| Without Embedding Features | 90.57% | 0.9663 | 0.9035 | -0.0050 |
| Without Text Similarity | 91.11% | 0.9701 | 0.9089 | -0.0012 |
| Without Metadata Features | 90.43% | 0.9634 | 0.9008 | -0.0079 |
| Without BM25 Features | 91.64% | 0.9714 | 0.9145 | +0.0001 |
| Without CPC Features | 91.57% | 0.9713 | 0.9142 | 0.0000 |

**Feature Selection Process:**

Based on the ablation study, we systematically evaluated each feature group's contribution:

1. **Removed Features (2):**
   - **BM25 Features** (`bm25_doc_score`, `bm25_best_claim_score`): Removed because ablation showed a slight performance improvement (+0.0001 ROC-AUC) when removed, indicating these features introduced noise or redundancy.
   - **CPC Features** (`cpc_jaccard`): Removed due to neutral impact (0.0000 Δ AUC), suggesting CPC codes don't provide additional discriminative power beyond other features.

2. **Retained Features (10):**
   - **Claim Similarity** (`claim_similarity`): Most critical feature (-0.0268 ROC-AUC drop if removed)
   - **Embedding Features** (`cosine_doc_similarity`, `cosine_max_claim_similarity`, `embedding_diff_mean`, `embedding_diff_std`): Important for semantic matching (-0.0050 ROC-AUC drop)
   - **Metadata Features** (`year_diff`, `abstract_length_ratio`, `claim_count_ratio`): Important for maintaining high recall (-0.0079 ROC-AUC drop)
   - **Text Similarity** (`title_jaccard`, `shared_rare_terms_ratio`): Helpful for lexical matching (-0.0012 ROC-AUC drop)

**Final Model:** The production model uses **10 features** (reduced from 13), achieving comparable or slightly better performance (ROC-AUC: 0.9714) with a simpler, more interpretable feature set.

**Key Findings:**
- **Most Critical Feature:** Claim similarity features (removal causes largest performance drop: -0.0268)
- **Removed Features:** BM25 features (2 features, slightly harmful) and CPC (neutral impact)
- **Feature Engineering Success:** 5 out of 6 feature groups are helpful, contributing ~0.041 ROC-AUC improvement

### Model Architecture Details

**Ensemble Model:**
- Base Models: MLP Classifier (hidden_layer_sizes=(64,), alpha=1e-5, learning_rate=0.005) + PyTorch Neural Network (hidden_dims=[128, 64, 32], dropout=0.3, batch normalization)
- Meta-learner: Logistic Regression with probability calibration
- Method: Stacking with CalibratedClassifierCV

**MLP Classifier:**
- Architecture: Single hidden layer (64 neurons)
- Regularization: L2 (alpha=1e-5)
- Learning Rate: 0.005
- Optimizer: Adam

**PyTorch Neural Network:**
- Architecture: [128, 64, 32] with residual connections
- Regularization: Dropout (0.3), Batch Normalization, L2 weight decay (1e-4)
- Learning Rate: 0.001
- Optimizer: AdamW

### Inference Performance

- **Single Prediction Latency:** 0.15ms (mean), 0.14ms (median)
- **Batch Throughput:** 1.4M+ predictions/second
- **LLM Explanation Generation:** ~30-60 seconds via Ollama

## Individual Contributions

*Individual project - single contributor*


