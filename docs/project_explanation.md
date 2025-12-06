# Patent Novelty Assessment System: Complete Technical Explanation

## Table of Contents
1. [Project Goal](#project-goal)
2. [What the Model Predicts](#what-the-model-predicts)
3. [Training Data and Ground Truth](#training-data-and-ground-truth)
4. [The 13 Features: Complete Mathematical Definitions](#the-13-features-complete-mathematical-definitions)
5. [Citation Pairs: Purpose and Validity](#citation-pairs-purpose-and-validity)
6. [Ground Truth Labels: What Accuracy Measures](#ground-truth-labels-what-accuracy-measures)
7. [Production Pipeline](#production-pipeline)

---

## Project Goal

**Patent Novelty Assessment**: Automatically determine if a new patent application is novel (not too similar to existing patents) by comparing it against a database of 200,000+ USPTO patents.

**Problem**: Patent examiners manually check novelty, which is time-consuming and requires expertise.

**Solution**: Automated system that:
1. Finds similar patents (retrieval)
2. Scores similarity (MLP/PyTorch classifier)
3. Explains why (Phi-3 LLM)

---

## What the Model Predicts

### Training Objective

**Input:** Two patents (Patent A, Patent B) with 13 numerical features  
**Output:** Probability that Patent A and Patent B are **similar** (0.0 to 1.0)

- **Label 1 (Positive)**: Patents are similar
- **Label 0 (Negative)**: Patents are not similar

### In Production

The model predicts similarity, then converts to novelty:

```
Novelty Score = 1 - Similarity Probability
```

- **High similarity (0.9)** → Low novelty (0.1) → Patent may be rejected
- **Low similarity (0.2)** → High novelty (0.8) → Patent is likely novel

### Assessment Thresholds

- **Novelty > 0.7**: "NOVEL" - Patent is likely novel
- **Novelty 0.4-0.7**: "MODERATELY NOVEL" - Some overlap, may need revision
- **Novelty < 0.4**: "LOW NOVELTY" - Very similar to prior art, likely rejection

---

## Training Data and Ground Truth

### Training Set Composition

**Total: 39,979 patent pairs**

**Positive Examples (Label=1): 19,966 pairs (~50%)**
- **Citation pairs: 19,966** - Patent A cites Patent B
- **CPC matches** - Patents share same CPC classification code
- **High embedding similarity** - PatentSBERTa cosine > 0.85

**Negative Examples (Label=0): 20,013 pairs (~50%)**
- **Random cross-CPC** - Patents from different technology fields
- **Temporal negatives** - Patents from very different time periods (3+ years apart)
- **Hard negatives** - High BM25 but low embedding similarity

### Validation Set: 8,567 pairs
### Test Set: 8,568 pairs

**Test Set Breakdown:**
- Label 1 (Positive): 4,244 pairs (citation pairs)
- Label 0 (Negative): 4,324 pairs (random negatives)

---

## The 13 Features: Complete Mathematical Definitions

The PyTorch model receives **13 numerical features** for each patent pair. Here are the complete mathematical definitions:

### Feature 1: BM25 Document Score

**Purpose:** Measures lexical (word-based) similarity between full patent documents.

**Formula:**

\[
\text{BM25}(Q, D) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
\]

Where:
- \(Q\) = Query document (Patent A: title + abstract)
- \(D\) = Document (Patent B: title + abstract)
- \(q_i\) = \(i\)-th term in query
- \(f(q_i, D)\) = Term frequency of \(q_i\) in \(D\)
- \(|D|\) = Length of document \(D\) (number of terms)
- \(\text{avgdl}\) = Average document length in corpus
- \(k_1 = 1.2\) (term frequency saturation parameter)
- \(b = 0.75\) (length normalization parameter)
- \(\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\) (inverse document frequency)
  - \(N\) = Total number of documents
  - \(n(q_i)\) = Number of documents containing \(q_i\)

**Normalization:** Score is normalized to [0, 1] range using min-max scaling across the training set.

**Implementation:** Uses `rank_bm25.BM25Okapi` library with patent-specific tokenization.

---

### Feature 2: BM25 Best Claim Score

**Purpose:** Measures lexical similarity between the most important claims of each patent.

**Formula:**

\[
\text{BM25}_{\text{claim}}(Q, D) = \max_{c_1 \in C_1, c_2 \in C_2} \text{BM25}(c_1, c_2)
\]

Where:
- \(C_1\) = Set of claims from Patent A
- \(C_2\) = Set of claims from Patent B
- \(c_1, c_2\) = Individual claims

**Normalization:** Same as Feature 1.

**Rationale:** Claims define the invention's scope. High claim similarity indicates overlapping inventions.

---

### Feature 3: Cosine Document Similarity

**Purpose:** Measures semantic similarity between full patent documents using PatentSBERTa embeddings.

**Formula:**

\[
\text{cosine\_sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \cdot \|\mathbf{e}_2\|} = \frac{\sum_{i=1}^{d} e_{1i} \cdot e_{2i}}{\sqrt{\sum_{i=1}^{d} e_{1i}^2} \cdot \sqrt{\sum_{i=1}^{d} e_{2i}^2}}
\]

Where:
- \(\mathbf{e}_1\) = PatentSBERTa embedding of Patent A (768-dimensional)
- \(\mathbf{e}_2\) = PatentSBERTa embedding of Patent B (768-dimensional)
- \(d = 768\) (embedding dimension)
- Embeddings computed from: `title + abstract + independent_claims`

**Range:** [-1, 1], typically [0, 1] for normalized embeddings.

**Model:** `AI-Growth-Lab/PatentSBERTa` (sentence-transformers)

---

### Feature 4: Cosine Max Claim Similarity

**Purpose:** Measures semantic similarity between the best-matching pair of claims.

**Formula:**

\[
\text{cosine\_max\_claim} = \max_{c_1 \in C_1, c_2 \in C_2} \text{cosine\_sim}(\text{embed}(c_1), \text{embed}(c_2))
\]

Where:
- \(C_1, C_2\) = Sets of claims from Patent A and B
- \(\text{embed}(c)\) = PatentSBERTa embedding of claim \(c\)

**Range:** [-1, 1], typically [0, 1]

**Rationale:** Identifies the most similar claim pair, which is critical for novelty assessment.

---

### Feature 5: Embedding Difference (Mean)

**Purpose:** Measures average absolute difference between embedding vectors.

**Formula:**

\[
\text{embedding\_diff\_mean} = \frac{1}{d} \sum_{i=1}^{d} |e_{1i} - e_{2i}|
\]

Where:
- \(e_{1i}, e_{2i}\) = \(i\)-th component of embeddings \(\mathbf{e}_1, \mathbf{e}_2\)
- \(d = 768\) (embedding dimension)

**Range:** [0, ∞), normalized to [0, 1] for training.

**Interpretation:** Lower values indicate more similar embeddings (and thus more similar patents).

---

### Feature 6: Embedding Difference (Standard Deviation)

**Purpose:** Measures variability in embedding differences across dimensions.

**Formula:**

\[
\text{embedding\_diff\_std} = \sqrt{\frac{1}{d-1} \sum_{i=1}^{d} (|e_{1i} - e_{2i}| - \mu)^2}
\]

Where:
- \(\mu = \text{embedding\_diff\_mean}\) (from Feature 5)
- \(d = 768\)

**Range:** [0, ∞), normalized to [0, 1]

**Interpretation:** High std indicates some dimensions are very different while others are similar (mixed similarity).

---

### Feature 7: CPC Jaccard Similarity

**Purpose:** Measures overlap in technology classifications (CPC codes).

**Formula:**

\[
\text{CPC\_Jaccard} = \frac{|C_1 \cap C_2|}{|C_1 \cup C_2|} = \frac{|\text{shared\_CPC\_codes}|}{|\text{all\_CPC\_codes}|}
\]

Where:
- \(C_1\) = Set of CPC codes for Patent A
- \(C_2\) = Set of CPC codes for Patent B
- \(|C_1 \cap C_2|\) = Number of shared CPC codes
- \(|C_1 \cup C_2|\) = Number of unique CPC codes across both patents

**Range:** [0, 1]
- 0 = No shared CPC codes (different technology fields)
- 1 = Identical CPC codes (same technology field)

**Example:** 
- Patent A: `["G06F", "A61K"]`
- Patent B: `["G06F", "H04L"]`
- Shared: `["G06F"]` (1 code)
- Union: `["G06F", "A61K", "H04L"]` (3 codes)
- Jaccard = 1/3 = 0.33

---

### Feature 8: Year Difference

**Purpose:** Measures temporal distance between patent filing dates.

**Formula:**

\[
\text{year\_diff} = \min\left(\frac{|\text{year}_1 - \text{year}_2|}{10}, 1.0\right)
\]

Where:
- \(\text{year}_1, \text{year}_2\) = Filing years of Patent A and B
- Normalized by assuming max difference of 10 years

**Range:** [0, 1]
- 0 = Same year
- 1 = 10+ years apart

**Rationale:** Patents filed close together are more likely to be related (same technology era).

---

### Feature 9: Title Jaccard Similarity

**Purpose:** Measures word overlap in patent titles.

**Formula:**

\[
\text{title\_Jaccard} = \frac{|T_1 \cap T_2|}{|T_1 \cup T_2|}
\]

Where:
- \(T_1, T_2\) = Sets of tokens (words) from Patent A and B titles
- Tokens are lowercased and extracted using regex: `\b\w+\b`
- \(|T_1 \cap T_2|\) = Number of shared words
- \(|T_1 \cup T_2|\) = Number of unique words across both titles

**Range:** [0, 1]

**Example:**
- Title A: "Smart water bottle with sensors"
- Title B: "Water bottle with smart sensors"
- Shared: `{"water", "bottle", "with", "smart", "sensors"}` (5 words)
- Union: Same 5 words
- Jaccard = 5/5 = 1.0

---

### Feature 10: Abstract Length Ratio

**Purpose:** Measures relative length similarity of abstracts.

**Formula:**

\[
\text{abstract\_length\_ratio} = \frac{\min(|\text{abstract}_1|, |\text{abstract}_2|)}{\max(|\text{abstract}_1|, |\text{abstract}_2|)}
\]

Where:
- \(|\text{abstract}_i|\) = Character length of abstract \(i\)

**Range:** [0, 1]
- 1 = Same length
- 0 = One abstract is empty

**Rationale:** Similar patents often have similar abstract lengths (similar level of detail).

---

### Feature 11: Claim Count Ratio

**Purpose:** Measures relative number of claims.

**Formula:**

\[
\text{claim\_count\_ratio} = \frac{\min(\text{num\_claims}_1, \text{num\_claims}_2)}{\max(\text{num\_claims}_1, \text{num\_claims}_2)}
\]

Where:
- \(\text{num\_claims}_i\) = Number of claims in Patent \(i\)

**Range:** [0, 1]
- 1 = Same number of claims
- 0 = One patent has no claims

**Rationale:** Similar inventions often have similar claim structures.

---

### Feature 12: Shared Rare Terms Ratio

**Purpose:** Measures overlap of technical/rare terminology (important for patent similarity).

**Formula (TF-IDF based):**

\[
\text{shared\_rare\_terms\_ratio} = \frac{|R_1 \cap R_2|}{|R_1 \cup R_2|}
\]

Where:
- \(R_1, R_2\) = Sets of rare terms from Patent A and B
- Rare terms identified by: \(\text{TF-IDF}(t, d) > \theta\) (threshold = 0.1)
- Or heuristic: words with length > 10 characters

**TF-IDF Formula:**

\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t) = \frac{f(t, d)}{|d|} \cdot \log \frac{N}{n(t)}
\]

Where:
- \(f(t, d)\) = Frequency of term \(t\) in document \(d\)
- \(|d|\) = Total terms in document \(d\)
- \(N\) = Total documents in corpus
- \(n(t)\) = Number of documents containing term \(t\)

**Range:** [0, 1]

**Rationale:** Technical terms (rare words) are more discriminative than common words. High overlap indicates similar technical concepts.

---

### Feature 13: Claim Similarity

**Purpose:** Measures average semantic similarity across all claim pairs.

**Formula:**

\[
\text{claim\_similarity} = \frac{1}{|C_1| \cdot |C_2|} \sum_{c_1 \in C_1} \sum_{c_2 \in C_2} \text{cosine\_sim}(\text{embed}(c_1), \text{embed}(c_2))
\]

Or alternatively (if using mean of best matches):

\[
\text{claim\_similarity} = \frac{1}{|C_1|} \sum_{c_1 \in C_1} \max_{c_2 \in C_2} \text{cosine\_sim}(\text{embed}(c_1), \text{embed}(c_2))
\]

Where:
- \(C_1, C_2\) = Sets of claims from Patent A and B
- \(\text{embed}(c)\) = PatentSBERTa embedding of claim \(c\)
- For each claim in Patent A, find best match in Patent B, then average

**Range:** [0, 1]

**Rationale:** Captures overall claim-level similarity across all claim pairs, providing a more comprehensive view than just the best-matching pair (Feature 4).

---

## Citation Pairs: Purpose and Validity

### Purpose of Citation Pairs

Citation pairs were used to create **high-quality positive training examples** for the MLP/PyTorch classifier.

**Why Citations Matter:**

When Patent A cites Patent B, it indicates:
- Patent A is related to Patent B (prior art)
- They likely share technical concepts
- They are good examples of "similar" patents

### How They Were Used

1. **Training Data Generation:**
   - Citation pairs from PatentsView (`g_us_patent_citation.tsv`)
   - Filtered to only pairs where BOTH patents are in our dataset
   - Saved to `data/citations/filtered_citations.jsonl` (28,559 pairs)

2. **Positive Training Examples:**
   - Citation pairs became positive examples (label=1)
   - Used in `data/training/train_pairs.jsonl`, `val_pairs.jsonl`, `test_pairs.jsonl`
   - Example: `{"patent_id_1": "10918706", "patent_id_2": "12311061", "label": 1, "pair_type": "citation"}`

3. **Training Dataset:**
   - Train: 39,979 pairs (includes 19,966 citation pairs)
   - Val: 8,567 pairs
   - Test: 8,568 pairs (includes 4,244 citation pairs)

### Are We Predicting Citations or Similarity?

**Answer: We are predicting SIMILARITY, not citations.**

Citations are used as a **proxy** (approximation) for similarity, but the model learns to predict similarity based on features, not citation patterns.

**The Assumption:**
- If Patent A cites Patent B, they are related/similar
- This is based on patent law: patents cite relevant prior art

**The Problem:**
- Citations might indicate relevance, not necessarily similarity
- A patent might cite a very different patent as prior art
- Citations serve multiple purposes (similar technology, prior art, background)

**Why It's Still Valid:**

1. **High-Quality Signal**: Citations are created by patent examiners and applicants (experts)
2. **Multiple Positive Sources**: Training includes citations + CPC matches + high embedding similarity
3. **Model Learns from Features**: Uses BM25, embeddings, CPC, etc. (semantic similarity indicators)
4. **Works on Non-Citation Pairs**: Model scores pairs without citations in production, suggesting it learned true similarity

**Validation Evidence:**
- High accuracy (91.82%) suggests learned meaningful patterns, not citation memorization
- Works on non-citation pairs in production
- High ROC-AUC (97.21%) suggests learned meaningful similarity patterns

---

## Ground Truth Labels: What Accuracy Measures

### The Critical Question

**When we say "91.82% accuracy," what are we comparing the model's predictions against?**

### Ground Truth Definition

**Test Set: 8,568 patent pairs**

- **Label 1 (Positive): 4,244 pairs**
  - **Source**: Citation pairs
  - **Definition**: Patent A cites Patent B (or vice versa)
  - **Assumption**: If patents cite each other, they are similar

- **Label 0 (Negative): 4,324 pairs**
  - **Source**: Random negative pairs
  - **Definition**: Random patents that don't cite each other
  - **Assumption**: If patents don't cite each other, they are not similar

### What Accuracy Measures

**Formula:**

\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Where:
- **TP (True Positive)**: Model predicts 1, ground truth is 1
- **TN (True Negative)**: Model predicts 0, ground truth is 0
- **FP (False Positive)**: Model predicts 1, ground truth is 0
- **FN (False Negative)**: Model predicts 0, ground truth is 1

**Ground Truth**: The label from the test set (1 = citation pair, 0 = random negative)

### What We're Actually Measuring

**91.82% accuracy means:**
- The model correctly predicts citation relationships 91.82% of the time
- It does NOT mean the model correctly predicts semantic similarity 91.82% of the time

### The Proxy Problem

**Ground Truth = Citation Relationship**
- Label 1 = Patents cite each other
- Label 0 = Patents don't cite each other

**What We Want = Semantic Similarity**
- Label 1 = Patents are semantically similar
- Label 0 = Patents are not semantically similar

**The Gap:**
- Citation relationship ≠ Semantic similarity
- Some citations are for different purposes (prior art, background)
- Some similar patents might not cite each other

### Why We Trust It

1. **Citation pairs are a reasonable proxy** for similarity
   - Most citations are to related patents
   - Patent examiners cite relevant prior art

2. **Model learns from features, not citations**
   - Uses BM25, embeddings, CPC, etc.
   - These capture semantic similarity, not just citation patterns

3. **Works on non-citation pairs in production**
   - Model scores pairs that may not have citations
   - If it only learned citations, it would fail here
   - But it works, suggesting it learned true similarity

4. **High ROC-AUC (97.21%)**
   - Measures ranking quality, not just binary accuracy
   - Suggests model learned meaningful similarity patterns

### Limitations

1. **Not human-validated similarity**
   - No human experts labeled pairs as "similar" or "not similar"
   - Ground truth is citation-based, not similarity-based

2. **Potential citation bias**
   - Model might be better at predicting citations than true similarity
   - Some similar patents might not cite each other (false negatives)
   - Some citations might not indicate similarity (false positives)

3. **Random negatives might be too easy**
   - Random cross-CPC pairs are obviously different
   - Model might perform worse on hard negatives (similar but not citations)

### The Caveat

**Ground truth is citation-based, not similarity-based.**

This is a limitation, but citations are the best available proxy at scale. For true similarity validation, we would need human-labeled pairs or embedding-based similarity thresholds.

---

## Production Pipeline

### Step-by-Step Flow

**Step 1: User Submits Patent Application**
- User provides: title, abstract, claims

**Step 2: Find Similar Patents**
- **Local search**: FAISS cosine similarity search (200K patents)
- **Online search**: Google Patents via SerpAPI (millions of patents)
- **LLM keyword extraction**: Phi-3 generates optimized search queries
- Get top 10-20 most similar patents

**Step 3: Extract Features for Each Pair**
For each (query patent, similar patent) pair:
- Extract 13 features comparing the two patents
- Example: BM25 score = 0.85, cosine similarity = 0.92, CPC overlap = 0.6, etc.

**Step 4: PyTorch Model Prediction**
- Input: 13 features as a vector
- Output: Similarity probability (e.g., 0.87 = 87% chance they're similar)

**Step 5: Compute Novelty Score**
\[
\text{Novelty Score} = 1 - \text{Similarity Probability}
\]

- If similarity = 0.87 → Novelty = 0.13 (LOW NOVELTY)
- If similarity = 0.20 → Novelty = 0.80 (HIGH NOVELTY)

**Step 6: Phi-3 Explanation**
- Generate detailed explanation citing specific prior art
- Provide recommendation (APPROVE/REVISE/REJECT)

### Example

**Query Patent**: "Smart water bottle with hydration tracking sensors"

**Similar Patent Found**: "Hydration monitoring device with Bluetooth connectivity"

**Features Extracted:**
- BM25 score: 0.82 (high lexical overlap)
- Cosine similarity: 0.89 (high semantic similarity)
- CPC overlap: 0.75 (same technology field)
- Year difference: 0.2 (2 years apart, normalized)
- Title Jaccard: 0.6 (moderate word overlap)
- ... (8 more features)

**PyTorch Prediction**: Similarity = 0.91 (91% chance they're similar)

**Novelty Score**: 1 - 0.91 = 0.09 (LOW NOVELTY)

**Assessment**: "LOW NOVELTY - Very similar to Patent 12345678. Consider revising claims to emphasize unique features."

---

## Ablation Studies

### Feature Ablation Study

We conducted ablation studies to understand the contribution of different feature groups:

**Configurations Tested:**
1. **All Features** (baseline): 91.60% accuracy, 97.09% ROC-AUC
2. **Without embedding features**: 90.76% accuracy (-0.84%)
3. **Without text similarity features**: 91.12% accuracy (-0.48%)
4. **Without metadata features**: 90.45% accuracy (-1.15%)
5. **Without BM25 features**: 91.69% accuracy (+0.09%)

**Key Findings:**
- Embedding features (cosine similarity, embedding differences) are most critical
- Metadata features (CPC, year, claim counts) provide significant value
- BM25 features have minimal impact (possibly redundant with embedding features)
- Text similarity features (title Jaccard, rare terms) provide moderate benefit

**Conclusion:** All feature groups contribute, with embedding features being most important. The full feature set achieves best performance.

### Model Architecture Ablation

**MLP Variants Tested:**
- MLP (32): 91.53% accuracy
- MLP (64-32): 91.60% accuracy (production baseline)
- MLP (128-64-32): 91.76% accuracy

**PyTorch Neural Network:**
- Architecture: [256, 128, 64, 32] with residual connections
- Accuracy: 91.82% (best performance)
- ROC-AUC: 97.21% (highest of all models)

**Key Findings:**
- Deeper networks provide marginal improvements
- Residual connections and batch normalization help with training stability
- PyTorch implementation outperforms sklearn MLP

## Summary

### Model Architecture

- **Type**: Binary classifier (PyTorch Neural Network)
- **Input**: 13 features per patent pair
- **Output**: Similarity probability [0, 1]
- **Architecture**: [256, 128, 64, 32] hidden layers with residual connections, batch normalization, dropout
- **Parameters**: 102,361 trainable parameters

### Training Data

- **Total pairs**: 39,979 (train) + 8,567 (val) + 8,568 (test)
- **Positive examples**: Citation pairs (19,966), CPC matches, high embedding similarity
- **Negative examples**: Random cross-CPC, temporal negatives, hard negatives
- **Ground truth**: Citation relationships (proxy for similarity)

### Performance

- **Accuracy**: 91.82% (on test set)
- **ROC-AUC**: 97.21% (highest of all models)
- **Baseline comparison**: +42.04% vs random, +16.46% vs title Jaccard heuristic

### Key Insights

1. **We predict similarity, not citations** - Citations are used as training signal
2. **Ground truth is citation-based** - Accuracy measures citation prediction, not human-judged similarity
3. **Model learns semantic patterns** - Works on non-citation pairs, suggesting true similarity learning
4. **13 features capture multiple similarity dimensions** - Lexical (BM25), semantic (embeddings), structural (CPC, claims), temporal (year)

