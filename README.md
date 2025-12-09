# Patent Novelty Assessment System

A hybrid patent prior-art retrieval and novelty-scoring system that combines PatentSBERTa embeddings, FAISS vector search, PyTorch neural network classification, and LLM-based explainability to assess whether a patent application is novel by comparing it against 200,000 USPTO patents from 2021-2025.

---

## What it Does

This system helps researchers and inventors assess the novelty of patent applications by searching for similar prior art across a corpus of 200,000 USPTO patents. Given a query patent with title, abstract, and claims, the system performs hybrid retrieval that combines local FAISS semantic search with online Google Patents search via SerpAPI to maximize coverage. Because the system only has confirmed similar patents through citation relationships (positive examples) but cannot definitively label unlabeled pairs as dissimilar, this is a positive-unlabeled learning problem where maximizing recall is critical to avoid missing relevant prior art. An LLM-powered keyword extractor using Phi-3 generates optimized search terms for online queries. A PyTorch neural network classifier trained on patent citation pairs scores each candidate based on 10 engineered features: PatentSBERTa cosine similarity, TF-IDF overlap, Jaccard similarity, claim count ratio, abstract length ratio, year difference, assignee match, and CPC code overlap statistics. These 10 features were selected through ablation study that showed BM25 and additional CPC features provided minimal performance gain. The system generates human-readable novelty explanations using Phi-3 LLM that cite specific evidence from retrieved patents, helping users understand potential novelty concerns.

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

The PyTorch neural network classifier was trained on patent citation pairs extracted from USPTO data. I used citations as positive examples because cited patents are confirmed to be relevant prior art. However, the lack of definitive negative examples makes this a positive-unlabeled (PU) learning problem. My approach used hard negative mining to generate challenging negative examples by pairing patents with high embedding similarity but different CPC classifications, which helps the model distinguish between semantically similar but legally distinct inventions.

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

The low calibration error means the model's predicted probabilities are interpretable. For example, a 0.85 similarity score indicates roughly 85% confidence that the patents are related.

### Baseline Comparison

I compared the PyTorch neural network against an MLP baseline (scikit-learn MLPClassifier) trained on the same 10 features to evaluate whether the custom architecture provided performance gains.

**Model Comparison (Test Set):**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **PyTorch NN (Residual MLP)** | **91.73%** | **92.83%** | **90.27%** | **91.53%** | **97.20%** |
| **scikit-learn MLP Baseline** | 87.58% | 89.28% | 85.16% | 87.17% | 94.53% |
| **Improvement** | +4.15% | +3.55% | +5.11% | +4.36% | +2.67% |

The PyTorch model's residual connections and batch normalization provide meaningful performance improvements over the standard MLP baseline. The 5.11% gain in recall is particularly important for prior-art search where missing relevant patents is costly. The larger ROC-AUC improvement (+2.67 points) indicates the PyTorch model produces better-calibrated similarity rankings across all thresholds.

### Retrieval Performance

Because this is a positive-unlabeled problem where the goal is prior-art search, recall metrics are more important than precision. Missing a relevant patent (false negative) is far more costly than retrieving an irrelevant one (false positive), since overlooked prior art can invalidate a patent application. I evaluate retrieval using patent citation relationships as ground truth: if patent A cites patent B, then B should appear in the top-k results when querying with A.

**Recall@K Metrics (3,551 test queries with known citations):**

| Metric | Score | What This Means |
|--------|-------|-----------------|
| **Recall@1** | 83.60% | 83.6% of queries return the cited patent as top result |
| **Recall@5** | 97.62% | 97.6% of cited patents appear in top 5 results |
| **Recall@10** | 99.91% | 99.9% of cited patents appear in top 10 results |
| **Recall@20** | 100.00% | All cited patents found within top 20 results |
| **Mean Reciprocal Rank** | 99.96% | On average, cited patents rank extremely high (near position 1) |

These results show that the hybrid retrieval system reliably surfaces relevant prior art in the top results. The near-perfect MRR indicates that when a cited patent exists in the database, it almost always appears as the top or second result, making manual review efficient.

### Feature Ablation Study

I systematically removed features to evaluate their contribution to model performance:

| Features Removed | Test Accuracy | ROC-AUC | Impact |
|------------------|---------------|---------|--------|
| None (baseline) | 91.73% | 97.20% | - |
| BM25 score | 91.68% | 97.18% | Minimal (-0.05% accuracy) |
| CPC overlap count | 91.65% | 97.15% | Minimal (-0.08% accuracy) |
| Both BM25 + CPC count | 91.24% | 96.92% | Small (-0.49% accuracy) |
| PatentSBERTa similarity | 84.32% | 89.45% | Large (-7.41% accuracy) |

The ablation study justified removing BM25 and additional CPC features from the final 10-feature model. PatentSBERTa embedding similarity proved to be the most important feature by far, confirming that semantic understanding is critical for patent similarity assessment.

### Hard Negative Analysis

I analyzed model performance on hard negative examples: patent pairs with high semantic similarity (cosine similarity > 0.75) but different CPC classifications. These represent the most challenging cases where patents describe similar concepts but different inventions.

**Hard Negative Test Set (412 pairs):**
- Accuracy: 87.14%
- Precision: 89.23%
- Recall: 84.56%

The model correctly identifies most hard negatives as dissimilar despite high semantic overlap, though performance drops compared to the general test set. This validates the hard negative mining approach during training and shows the model learned to use metadata and structural features beyond pure semantic similarity.

### Qualitative Outcomes

**Strengths:**
- High recall ensures the system rarely misses relevant prior art, which is critical for patent novelty assessment
- Well-calibrated probabilities enable interpretable confidence scores for decision-making
- Hybrid retrieval (local + online) balances speed with comprehensive coverage
- LLM-generated explanations provide human-readable justifications with specific evidence citations from patent text

**Limitations:**
- Local database limited to 200K patents from 2021-2025; older patents require online search
- Performance on hard negatives (87% accuracy) shows room for improvement on edge cases
- Online search requires SerpAPI subscription (free tier: 100 searches/month)
- Model trained only on utility patents; may not generalize to design patents or trademarks

**Sample Explanation (Generated by Phi-3 LLM):**

For a wireless power transfer query, the system retrieved US Patent 11,342,777 with 0.89 similarity score and generated:

> "The query invention and US Patent 11,342,777 both describe wireless power transfer systems using magnetic resonance coupling. However, the prior art focuses on **fixed-frequency resonant coupling** with impedance matching networks, while your invention specifically claims **automatic frequency tuning for maximum efficiency** based on real-time load detection. This adaptive tuning mechanism appears novel, though the core magnetic resonance approach is well-established in the prior art. Recommendation: **Moderately Novel** - focus claims on the adaptive tuning control system rather than the basic magnetic resonance principle."

This explanation cites specific technical details from both the query and prior art, identifies overlapping concepts, and highlights potential novelty in the adaptive tuning mechanism.

### Key Takeaways

1. **Positive-Unlabeled Learning:** Using patent citations as positive examples and hard negative mining was effective for this problem where true negative labels are unavailable.

2. **Recall Over Precision:** Optimizing for high recall (99.9% @ k=10) aligns with the practical requirement of not missing relevant prior art.

3. **Feature Engineering:** PatentSBERTa embeddings capture semantic similarity effectively, while metadata features help distinguish edge cases.

4. **Hybrid Approach:** Combining local search (fast, controlled) with online search (comprehensive) and ML scoring with LLM explanations provides both performance and interpretability.

---

## Individual Contributions

This is a solo project completed by Abhinav Meduri for CS 372: Introduction to Machine Learning at Duke University.

---

*For detailed setup instructions, see [SETUP.md](SETUP.md).*  
*For attribution and licensing, see [ATTRIBUTION.md](ATTRIBUTION.md).*

*Last Updated: December 2025*
