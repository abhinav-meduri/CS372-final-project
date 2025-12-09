# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines PatentSBERTa embeddings, FAISS vector search, PyTorch neural network classification, and LLM-based explainability to assess whether a patent application is novel by comparing it against 200,000 USPTO patents from 2021-2025.

---

## What it Does

In this project, I explored patent prior-art retrieval and novelty assessment across a corpus of 200,000 USPTO patents from 2021-2025. The system combines local FAISS semantic search with online Google Patents search to find similar patents. A PyTorch neural network classifier, trained on patent citation pairs, scores candidate patents based on 10 engineered features including PatentSBERTa cosine similarity, TF-IDF overlap, Jaccard similarity, claim count ratio, abstract length ratio, year difference, assignee match, and CPC code overlap. This is a positive-unlabeled (PU) learning problem because while citation relationships definitively identify related patent pairs, we cannot definitively identify all unrelated pairs, making high recall critical to avoid missing relevant prior art. The model learns to predict patent relatedness, which inversely indicates novelty. The system generates explanations using Phi-3 LLM to help users understand potential novelty concerns.

---

## Quick Start

```bash
git clone https://github.com/abhinavmeduri/CS372-final-project.git
cd CS372-final-project
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Download data from Box: https://duke.box.com/s/4y6mjf1965d15gnltnkqnk0dkedbttqh
# Extract and place in data/ and models/ (see SETUP.md)

brew install ollama
brew services start ollama
ollama pull phi3

export SERPAPI_KEY=your_serpapi_key_here

streamlit run app.py
```

For complete setup instructions, see [SETUP.md](SETUP.md).

---

## Video Links

- **Demo Video:** `[INSERT LINK HERE]`
- **Technical Walkthrough:** `[INSERT LINK HERE]`

---

## Evaluation

### Classification Performance

**Training Data and Problem Formulation:**

The classifier was trained on patent citation pairs from USPTO data. This is a positive-unlabeled (PU) learning problem because while we can definitively identify related patent pairs through citations, we cannot definitively enumerate all unrelated pairs. Any two patents that don't cite each other might still be related in ways not captured by the citation graph. Therefore, our "negative" examples are actually unlabeled pairs sampled randomly from non-citing patents, not confirmed negatives.

Citation relationships serve as a reliable proxy for patent relatedness because when patent A cites patent B, B is confirmed relevant prior art. The model learns to predict relatedness (similarity) between patent pairs, which inversely indicates novelty: high similarity to existing patents suggests lower novelty. The model was trained on 28,557 positive pairs (citations) and 28,557 negative pairs (random non-citing pairs), with architecture and hyperparameters selected through 3-fold cross-validation grid search over 54 configurations, achieving a best validation ROC-AUC of 97.17%.

**Test Set Metrics (8,568 patent pairs):**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 91.73% | Overall correctness on balanced test set |
| **Precision** | 92.83% | When predicting similar, correct 92.8% of time |
| **Recall** | 90.27% | Captures 90.3% of actually similar pairs |
| **F1 Score** | 91.53% | Balanced precision-recall trade-off |
| **ROC-AUC** | 97.20% | Strong ranking ability across thresholds |
| **PR-AUC** | 97.47% | High precision-recall curve area |
| **Expected Calibration Error** | 0.89% | Probabilities closely match actual rates |
| **Brier Score** | 0.063 | Low probabilistic prediction error |

The calibration error indicates that predicted probabilities approximate the true likelihood of similarity.

### Baseline Comparison

The PyTorch neural network was compared against multiple baselines including random guessing, logistic regression, cosine similarity heuristic, and scikit-learn MLP to evaluate performance gains.

**Model Comparison (Test Set):**

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **PyTorch NN (Residual MLP)** | **91.73%** | **91.53%** | **97.20%** |
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
