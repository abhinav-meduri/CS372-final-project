# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines BM25 lexical search, PatentSBERTa embeddings, FAISS indexing, PyTorch neural network classification, and LLM-based explainability to assess patent novelty.

## What it Does

This system helps researchers and inventors quickly assess the novelty of patent applications by comparing them against a corpus of 200,000 USPTO patents (2021-2025). Given a query patent (title, abstract, and claims), the system performs hybrid retrieval combining local FAISS similarity search with online Google Patents search (via SerpAPI). LLM-powered keyword extraction (Phi-3) generates optimized search terms. A trained PyTorch neural network scores each candidate based on 13 engineered features including embedding similarity, text overlap metrics, and metadata features. Finally, Phi-3 LLM generates human-readable explanations citing specific evidence from the prior art, helping users understand why certain patents may pose novelty concerns.

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
   - `data/features/feature_names_v2.json` - Feature names (included in repo)
   - `models/pytorch_nn/pytorch_model.pt` - Trained PyTorch model
   - `models/pytorch_nn/scaler_pytorch.pkl` - Feature scaler
   - `models/pytorch_nn/training_history_pytorch.json` - Training metadata
   
   **Options:**
   - Download from a shared location (Google Drive, Dropbox, etc.)
   - Generate from raw PatentsView data using scripts in `scripts/data/`
   - Contact repository maintainer for data access
   
   **Note:** Without these files, the application will not run. The system needs the patent database and pre-computed embeddings for similarity search.

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

### PyTorch Neural Network Performance (Production Model)

| Metric | Test Set |
|--------|----------|
| **Accuracy** | 91.56% |
| **ROC-AUC** | 0.9714 |
| **Precision** | 92.41% |
| **Recall** | 90.39% |
| **F1 Score** | 0.9139 |

### Baseline Comparison

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Random Guessing | 49.8% | 0.495 | 0.500 |
| Majority Class | 50.5% | 0.000 | 0.500 |
| Title Jaccard Heuristic | 75.4% | 0.737 | N/A |
| Logistic Regression | 90.8% | 0.905 | 0.968 |
| **PyTorch Neural Network (Ours)** | **91.56%** | **0.9139** | **0.9714** |

### Ablation Study

Ablation study results for the PyTorch Neural Network (13 features with real BM25):

| Configuration | ROC-AUC | Δ AUC |
|---------------|---------|-------|
| Full Model (13 features) | 0.9714 | -- |
| Without Embedding Features | TBD | TBD |
| Without BM25 Features | TBD | TBD |
| Without Metadata Features | TBD | TBD |
| Without Text Similarity | TBD | TBD |

**Note:** Ablation study is being re-run on the trained PyTorch model with real BM25 features. Results will be updated shortly.

### Model Architecture Comparison

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Gradient Boosting | 91.9% | 0.9717 |
| MLP (32) | 91.5% | 0.9713 |
| **PyTorch NN (128-64-32) [Production]** | **91.56%** | **0.9714** |
| MLP (128-64-32) | 91.8% | 0.9709 |
| Random Forest | 91.5% | 0.9709 |

### Inference Performance

- **Single Prediction Latency:** 0.15ms (mean), 0.14ms (median)
- **Batch Throughput:** 1.4M+ predictions/second
- **LLM Explanation Generation:** ~30-60 seconds via Ollama

## System Architecture

```
Query Patent → LLM Keyword Extraction (Phi-3) → Hybrid Search
                                              ↓
                    ┌─────────────────────────┴─────────────────────────┐
                    ↓                                                   ↓
        Local FAISS Search (200K)                    Online Search (SerpAPI)
                    ↓                                                   ↓
                    └─────────────────────────┬─────────────────────────┘
                                              ↓
                               Top-K Similar Patents (Merged & Deduplicated)
                                              ↓
                               Feature Extraction (13 features)
                                              ↓
                               PyTorch NN Novelty Scoring
                                              ↓
                               Phi-3 LLM Explanation
                                              ↓
                               Streamlit UI Output
```

## Individual Contributions

*Individual project - single contributor*

---

⚠️ **DISCLAIMER:** This is a research prototype for educational purposes. Not legal advice. Always consult a patent attorney for legal novelty opinions.


