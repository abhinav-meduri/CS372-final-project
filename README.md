# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines PatentSBERTa embeddings, FAISS vector search, PyTorch neural network classification, and LLM-based explainability to assess whether a patent application is novel by comparing it against 200,000 USPTO patents from 2021-2025.

---

## What it Does

In this project, I explored patent prior-art retrieval and novelty assessment across a corpus of 200,000 USPTO patents from 2021-2025. The system combines local FAISS semantic search with online Google Patents search to find similar patents. A PyTorch neural network classifier, trained on patent citation pairs, scores candidate patents based on 10 engineered features including PatentSBERTa cosine similarity, TF-IDF overlap, Jaccard similarity, claim count ratio, abstract length ratio, year difference, assignee match, and CPC code overlap. This is a positive-unlabeled (PU) learning problem because while citation relationships definitively identify related patent pairs, we cannot definitively identify all unrelated pairs, making high recall critical to avoid missing relevant prior art. The model learns to predict patent relatedness, which inversely indicates novelty. The system generates explanations using Phi-3 LLM to help users understand potential novelty concerns.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/abhinavmeduri/CS372-final-project.git
cd CS372-final-project

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download required data files (~7 GB) from Duke Box
# Go to: https://duke.box.com/s/4y6mjf1965d15gnltnkqnk0dkedbttqh
# Extract and place files as follows:
#   - patent_embeddings.npy -> data/embeddings/
#   - patent_ids.json -> data/embeddings/
#   - patents_sampled.jsonl -> data/sampled/
#   - pytorch_model.pt -> models/pytorch_nn/
#   - scaler_pytorch.pkl -> models/pytorch_nn/
# (See SETUP.md for detailed instructions)

# 4. Install and configure Ollama (for LLM explanations)
brew install ollama                    # macOS
# OR: curl -fsSL https://ollama.ai/install.sh | sh  # Linux
brew services start ollama
ollama pull phi3

# 5. Set your SerpAPI key (for online patent search)
export SERPAPI_KEY=your_serpapi_key_here
# Get free API key from: https://serpapi.com/ (100 searches/month)

# 6. Run the application
streamlit run app.py
```

For complete setup instructions, see [SETUP.md](SETUP.md).

---

## Required Data Files

**Important:** This repository does not include the large data files (~7 GB) required to run the application due to GitHub size limits.

**You must download these files from Duke Box:**

**[Download Required Data Files (Duke Box)](https://duke.box.com/s/4y6mjf1965d15gnltnkqnk0dkedbttqh)**

**What's included:**
- `patent_embeddings.npy` (~3.2 GB) - 200K pre-computed PatentSBERTa embeddings
- `patent_ids.json` (~15 MB) - Patent ID to embedding index mappings
- `patents_sampled.jsonl` (~3.8 GB) - Patent metadata database (200K patents from 2021-2025)
- `pytorch_model.pt` (~2 MB) - Trained PyTorch neural network classifier
- `scaler_pytorch.pkl` (~20 KB) - Feature normalization scaler

**File placement after download:**
```
CS372-final-project/
  data/
    embeddings/
      patent_embeddings.npy    (Place here - 3.2 GB)
      patent_ids.json          (Place here - 15 MB)
    sampled/
      patents_sampled.jsonl    (Place here - 3.8 GB)
  models/
    pytorch_nn/
      pytorch_model.pt         (Place here - 2 MB)
      scaler_pytorch.pkl       (Place here - 20 KB)
```

See [SETUP.md](SETUP.md) for detailed extraction and placement instructions.

---

## Video Links

- [Demo Video](https://drive.google.com/drive/folders/12FspQzWt7QM_nqvoML0M5z0BfGV2CLUY?usp=sharing)
- [Technical Walkthrough](https://drive.google.com/drive/folders/12FspQzWt7QM_nqvoML0M5z0BfGV2CLUY?usp=sharing)

---

## Evaluation

### Classification Performance

**Training Data and Problem Formulation:**

The classifier was trained on patent citation pairs from USPTO data. This is a positive-unlabeled (PU) learning problem because while we can definitively identify related patent pairs through citations, we cannot definitively enumerate all unrelated pairs. Any two patents that don't cite each other might still be related in ways not captured by the citation graph. Therefore, our "negative" examples are actually unlabeled pairs sampled randomly from non-citing patents, not confirmed negatives.

Citation relationships serve as a reliable proxy for patent relatedness because when patent A cites patent B, B is confirmed relevant prior art. The model learns to predict relatedness (similarity) between patent pairs, which inversely indicates novelty: high similarity to existing patents suggests lower novelty. The model was trained on 28,557 positive pairs (citations) and 28,557 negative pairs (random non-citing pairs), with architecture and hyperparameters selected through 3-fold cross-validation grid search over 54 configurations, achieving a best validation ROC-AUC of 97.17%.

**Test Set Metrics (8,568 patent pairs):**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 91.20% | Overall correctness on balanced test set |
| **Precision** | 93.98% | When predicting similar, correct 94.0% of time |
| **Recall** | 87.87% | Captures 87.9% of actually similar pairs |
| **F1 Score** | 90.82% | Balanced precision-recall trade-off |
| **ROC-AUC** | 97.01% | Strong ranking ability across thresholds |
| **PR-AUC** | 97.27% | High precision-recall curve area |
| **Expected Calibration Error** | 2.19% | Probabilities closely match actual rates |
| **Brier Score** | 0.063 | Low probabilistic prediction error |

The calibration error indicates that predicted probabilities approximate the true likelihood of similarity.

### Baseline Comparison

The PyTorch neural network was compared against multiple baselines including random guessing, logistic regression, cosine similarity heuristic, and scikit-learn MLP to evaluate performance gains.

**Model Comparison (Test Set):**

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **PyTorch NN (Residual MLP)** | **91.20%** | **90.82%** | **97.01%** |
| **Logistic Regression** | 90.93% | 90.63% | 96.74% |
| **scikit-learn MLP** | 87.58% | 87.17% | 94.53% |
| **Cosine Similarity (heuristic)** | 84.27% | 83.53% | 91.70% |
| **Random Guessing** | 49.78% | 49.50% | 50.32% |

The custom PyTorch architecture with residual connections and batch normalization outperforms all baselines. The gains over logistic regression and the MLP baseline demonstrate that the additional model capacity and architectural choices improve performance on this task.

### Retrieval Performance

**Why Retrieval Metrics Matter for PU Learning:**

In addition to standard classification metrics, retrieval metrics (Recall@K and Mean Reciprocal Rank) are critical for evaluating performance on this PU learning problem. Because we cannot definitively label all negative pairs, ranking-based metrics better reflect real-world usage: the system retrieves a ranked list of potentially similar patents, and users examine top results. For prior-art search, recall metrics are prioritized over precision since missing a relevant patent can be more costly than retrieving irrelevant ones.

Retrieval was evaluated using patent citation relationships as ground truth: if patent A cites patent B, then B should appear in the top-k results when querying with A.

**Recall@K Metrics (3,551 test queries with known citations):**

| Metric | Score | What This Means |
|--------|-------|-----------------|
| **Recall@1** | 83.60% | 83.6% of queries return the cited patent as top result |
| **Recall@5** | 97.62% | 97.6% of cited patents appear in top 5 results |
| **Recall@10** | 99.91% | 99.9% of cited patents appear in top 10 results |
| **Recall@20** | 100.00% | All cited patents found within top 20 results |
| **Mean Reciprocal Rank** | 99.96% | On average, cited patents rank extremely high (near position 1) |

The retrieval system surfaces relevant prior art in the top results, with cited patents typically appearing as the top or second result when they exist in the database.

### Outcomes

**Strengths:**
- High recall (99.9% @ k=10) ensures relevant prior art is surfaced
- Hybrid retrieval (local + online) balances speed with coverage
- LLM-generated explanations provide readable justifications

**Limitations:**
- Local database limited to 200K patents from 2021-2025
- Online search requires SerpAPI subscription and returns only patent snippets rather than full abstracts
- Performance may vary on patents outside the 2021-2025 timeframe

### Key Takeaways

1. **Positive-Unlabeled Learning:** This problem is inherently PU learning because we can identify related patent pairs through citations (positive examples) but cannot definitively enumerate all unrelated pairs. Citations serve as a reliable proxy for relatedness since they represent confirmed relevant prior art relationships.

2. **Recall Over Precision:** High recall (99.9% @ k=10) aligns with the practical requirement of not missing relevant prior art, which is critical in PU learning where false negatives are more costly than false positives.

3. **Feature Engineering:** PatentSBERTa embeddings capture semantic similarity, while metadata features provide additional signal for distinguishing related from potentially unrelated patents.

4. **Hybrid Approach:** Combining local search with online search and ML scoring with LLM explanations balances performance and interpretability for real-world patent novelty assessment.

---

## Individual Contributions

This is a solo project completed by Abhinav Meduri for CS 372: Introduction to Machine Learning at Duke University.

---

*For detailed setup instructions, see [SETUP.md](SETUP.md).*  
*For attribution and licensing, see [ATTRIBUTION.md](ATTRIBUTION.md).*

*Last Updated: December 2025*
