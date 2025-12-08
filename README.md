# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines PatentSBERTa embeddings, FAISS indexing, PyTorch neural network classification, and LLM-based explainability to assess patent novelty.

## What it Does

This system helps researchers and inventors quickly assess the novelty of patent applications by comparing them against a corpus of 200,000 USPTO patents (2021-2025). Given a query patent (title, abstract, and claims), the system performs hybrid retrieval combining local FAISS similarity search with online Google Patents search (via SerpAPI). LLM-powered keyword extraction (Phi-3) generates optimized search terms. A trained PyTorch neural network scores each candidate based on 10 engineered features (reduced from 13 via ablation study) including embedding similarity, text overlap metrics, and metadata features. Finally, Phi-3 LLM generates human-readable explanations citing specific evidence from the prior art, helping users understand why certain patents may pose novelty concerns.

## Demo

<div align="center">
  <img src="docs/demo_novelty_assessment.png" alt="Novelty Assessment Interface" width="600"/>
  <p><em>Novelty Assessment Interface - Shows analysis results with novelty score, similar patents, and detailed explanations</em></p>
</div>

<div align="center">
  <img src="docs/demo_analysis_pipeline.png" alt="Analysis Pipeline" width="600"/>
  <p><em>Analysis Pipeline - Demonstrates the complete workflow from patent input to novelty assessment</em></p>
</div>

## Quick Start

For complete setup instructions, see [SETUP.md](SETUP.md).

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd CS372-final-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required data files (see SETUP.md)
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

For detailed end-to-end setup instructions, see [SETUP.md](SETUP.md).

For technical documentation, see `docs/PROJECT_DOCUMENTATION.md`.

## Video Links

- **Demo Video:** [Link TBD]
- **Technical Walkthrough:** [Link TBD]

## Evaluation

### Model Performance Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **PyTorch Neural Network (Best)** | **91.73%** | **92.83%** | **90.27%** | **0.9153** | **0.9720** |
| MLP Classifier | 87.58% | 89.28% | 85.16% | 0.8717 | 0.9453 |
| Logistic Regression | 90.93% | 92.45% | 88.99% | 0.9063 | 0.9674 |
| Cosine Similarity (heuristic) | 84.27% | 81.23% | 86.42% | 0.8353 | 0.9170 |
| Random Guessing | 49.78% | 49.80% | 49.80% | 0.4950 | 0.5032 |
| Majority Class | 50.47% | 0.00% | 0.00% | 0.0000 | 0.5000 |

**Note:** The PyTorch Neural Network achieves the best overall performance with 10 engineered features.

### Individual Model Details

#### PyTorch Neural Network (Production - Best Performing)
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 91.73% |
| **Precision** | 92.83% |
| **Recall** | 90.27% |
| **F1 Score** | 0.9153 |
| **ROC-AUC** | 0.9720 |
| **PR-AUC** | 0.9747 |
| **Average Precision** | 0.9747 |
| **Brier Score** | 0.0627 |
| **ECE** | 0.0089 |

#### MLP Classifier
| Metric | Test Set |
|--------|----------|
| **Accuracy** | 87.58% |
| **Precision** | 89.28% |
| **Recall** | 85.16% |
| **F1 Score** | 0.8717 |
| **ROC-AUC** | 0.9453 |
| **Brier Score** | 0.0905 |
| **Brier Score** | 0.0639 |


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
| Without BM25 Features | 90.80% | 0.9680 | 0.9077 | +0.0001 |
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

**PyTorch Neural Network (Production Model):**
- Architecture: Multi-layer perceptron with residual connections
- Hidden Layers: [256, 128] neurons
- Regularization: Dropout (0.3), Batch Normalization, L2 weight decay (1e-5)
- Learning Rate: 0.002
- Batch Size: 256
- Features: 10 engineered features (embedding similarity, text overlap, metadata)

**MLP Classifier:**
- Architecture: Single hidden layer (64 neurons)
- Regularization: L2 (alpha=1e-5)
- Learning Rate: 0.005
- Optimizer: Adam

**PyTorch Neural Network:**
- Architecture: [256, 128] with residual connections
- Regularization: Dropout (0.3), Batch Normalization, L2 weight decay (1e-5)
- Learning Rate: 0.001
- Optimizer: AdamW

### Additional Analyses

**Baseline Comparison:**
The model was compared against several baseline approaches to establish performance benchmarks:
- **Random Guessing:** 49.78% accuracy (ROC-AUC: 0.50) - serves as lower bound
- **Majority Class:** 50.47% accuracy (ROC-AUC: 0.50) - demonstrates class imbalance handling
- **Cosine Similarity Heuristic:** 84.27% accuracy (ROC-AUC: 0.92) - simple embedding-based baseline
- **Logistic Regression:** 90.93% accuracy (ROC-AUC: 0.97) - linear baseline for comparison

The PyTorch Neural Network (91.73% accuracy) significantly outperforms all baselines, demonstrating the value of engineered features and non-linear modeling.

**Hard Negatives Analysis:**
The model was tested on 500 hard negative pairs (patents with high semantic similarity but different novelty classifications). Results show:
- **100% accuracy** on hard negatives (0 false positives)
- **Low prediction probabilities** (mean: 0.019, max: 0.036) indicating high confidence in "not novel" predictions
- **Robust discrimination:** Model correctly identifies semantically similar patents as non-novel, demonstrating strong generalization

**Input Length Sensitivity:**
Analysis of model performance across different input text lengths confirmed:
- **Stable performance** across typical patent document lengths
- **Robust to varying input sizes** due to fixed-size feature engineering approach
- Model maintains consistent accuracy regardless of document length variations

**Ablation Study:**
Systematic feature removal analysis (detailed above) identified the most critical features and led to feature selection from 13 to 10 features, improving model interpretability while maintaining performance.

### Inference Performance

- **Single Prediction Latency:** 2.66 ms mean (1.71 ms median) — measured on Apple M2 (mps) using `scripts/evaluation/benchmark_inference.py`
- **Batch Throughput:** ~614k predictions/second at batch size 4,096 (mps)
- **LLM Explanation Generation:** ~30–60 seconds via Ollama (Phi-3) for long-form explanations
- **Benchmark Artifact:** `results/analysis/inference_benchmark.json`

## Documentation

- **[SETUP.md](SETUP.md)** - Complete end-to-end installation and setup instructions
- **[ATTRIBUTION.md](ATTRIBUTION.md)** - Detailed attributions of all data sources, models, and AI assistance
- **docs/PROJECT_DOCUMENTATION.md** - Technical documentation and architecture details
- **notebooks/pipeline.ipynb** - Example usage and pipeline demonstration

## Individual Contributions

*Individual project - single contributor*

