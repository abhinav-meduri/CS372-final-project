# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines BM25 lexical search, PatentSBERTa embeddings, FAISS indexing, PyTorch neural network classification (91.82% accuracy), and LLM-based explainability to assess patent novelty.

## What it Does

This system helps researchers and inventors quickly assess the novelty of patent applications by comparing them against a corpus of 200,000 USPTO patents (2021-2025). Given a query patent (title, abstract, and claims), the system performs hybrid retrieval combining local FAISS similarity search with online Google Patents search (via SerpAPI). LLM-powered keyword extraction (Phi-3) generates optimized search terms. A trained PyTorch neural network (91.82% accuracy) scores each candidate based on 13 engineered features including embedding similarity, text overlap metrics, and metadata features. Finally, Phi-3 LLM generates human-readable explanations citing specific evidence from the prior art, helping users understand why certain patents may pose novelty concerns.

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
| **Accuracy** | 91.82% |
| **ROC-AUC** | 0.972 |
| **Precision** | 92.1% |
| **Recall** | 91.4% |
| **F1 Score** | 0.917 |

### Baseline Comparison

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Random Guessing | 49.8% | 0.495 | 0.500 |
| Majority Class | 50.5% | 0.000 | 0.500 |
| Title Jaccard Heuristic | 75.4% | 0.737 | N/A |
| Logistic Regression | 90.8% | 0.905 | 0.968 |
| **PyTorch Neural Network (Ours)** | **91.82%** | **0.913** | **0.972** |

### Ablation Study

| Configuration | ROC-AUC | Δ AUC |
|---------------|---------|-------|
| Full Model (13 features) | 0.9709 | -- |
| Without Embedding Features | 0.9671 | -0.0038 |
| Without BM25 Features | 0.9715 | +0.0006 |
| Without Metadata Features | 0.9642 | -0.0067 |
| Without Text Similarity | 0.9702 | -0.0007 |

### Model Architecture Comparison

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Gradient Boosting | 91.9% | 0.9717 |
| MLP (32) | 91.5% | 0.9713 |
| **PyTorch NN (64-32) [Production]** | **91.82%** | **0.9721** |
| MLP (128-64-32) | 91.8% | 0.9709 |
| Random Forest | 91.5% | 0.9709 |

### Inference Performance

- **Single Prediction Latency:** 0.15ms (mean), 0.14ms (median)
- **Batch Throughput:** 1.4M+ predictions/second
- **LLM Explanation Generation:** ~30-60 seconds via Ollama

## System Architecture

```
Query Patent → PatentSBERTa Embedding → FAISS Similarity Search
                                              ↓
                               Top-K Similar Patents
                                              ↓
                               Feature Extraction (13 features)
                                              ↓
                               PyTorch NN Novelty Scoring (91.82% accuracy)
                                              ↓
                               Phi-3 LLM Explanation
                                              ↓
                               Streamlit UI Output
```

## Individual Contributions

*Individual project - single contributor*

---

⚠️ **DISCLAIMER:** This is a research prototype for educational purposes. Not legal advice. Always consult a patent attorney for legal novelty opinions.


