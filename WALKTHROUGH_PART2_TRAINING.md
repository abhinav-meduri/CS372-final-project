# Code Walkthrough Part 2: Feature Engineering and Model Training

## Overview

Now that we have:
1. ✓ Embeddings for 200K patents (586 MB)
2. ✓ Training pairs with labels (57,114 pairs)

We need to:
3. Extract features from each pair
4. Train a neural network to predict similarity
5. Evaluate and tune hyperparameters

This part covers FILES 3 and 4.

---

# FILE 3: `scripts/data/preprocessing/compute_features.py`

## Purpose and Context

**Why do we need features at all? Don't we already have embeddings?**

Great question! Embeddings alone give us cosine similarity:
```python
sim = cosine_similarity(emb_a, emb_b)  # Single number
```

This captures semantic similarity but misses important signals:
- Do they have the same assignee (company)?
- Are they from the same year?
- Do they share CPC classification codes?
- Are their claims textually similar?
- What about title overlap?

**Our approach: Supervised feature engineering**

Instead of just using cosine similarity, we extract 10 diverse features that capture different similarity aspects. Then we train a neural network to learn the optimal combination.

**Why 10 features specifically?**

- Too few (2-3): Miss important signals
- Too many (50+): Risk overfitting, computational cost
- 10: Sweet spot from experimentation

We tried 5, 10, 15, and 20 features. 10 gave best val ROC-AUC.

## Complete Feature List (Deep Dive)

Let me explain each feature in detail before showing the code.

### Feature 1: PatentSBERTa Cosine Similarity

**What it captures:** Semantic similarity of main abstracts

**How it's computed:**
```python
emb_a = embeddings[id_to_idx[patent_a_id]]  # (768,)
emb_b = embeddings[id_to_idx[patent_b_id]]  # (768,)

cosine_sim = np.dot(emb_a, emb_b) / (
    np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
)
```

**Mathematical detail:**

Cosine similarity measures the angle between two vectors:
```
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
  A · B = dot product = Σ(a_i × b_i) for i=1 to 768
  ||A|| = L2 norm = sqrt(Σ(a_i²))
  ||B|| = L2 norm = sqrt(Σ(b_i²))
```

**Range:** [-1, 1]
- 1.0: Identical vectors (same direction)
- 0.0: Orthogonal vectors (unrelated)
- -1.0: Opposite vectors (antonyms)

For patents, we typically see 0.05 to 0.95 (rarely negative).

**Example values:**
- Two wireless charging patents: 0.87
- Wireless charging vs solar panel: 0.23
- Medical device vs software algorithm: 0.08

**Why cosine instead of Euclidean distance?**

Euclidean distance: `sqrt(Σ(a_i - b_i)²)`
- Sensitive to magnitude
- Two long abstracts might have large distance even if topically similar

Cosine similarity:
- Invariant to magnitude (normalizes by length)
- Focuses on direction (topic) not length
- Standard for text embeddings

**Computational efficiency:**

For one pair: 768 multiplications + 1 division = ~800 operations

With NumPy vectorization:
```python
# Slow: Python loop
for i in range(768):
    dot_product += emb_a[i] * emb_b[i]

# Fast: NumPy (uses BLAS)
dot_product = np.dot(emb_a, emb_b)  # 10-20× faster
```

NumPy uses optimized BLAS libraries (OpenBLAS, MKL, Accelerate on macOS) that use SIMD instructions for parallel multiplication.

### Feature 2: TF-IDF Cosine Similarity

**What it captures:** Keyword overlap (lexical similarity)

**How it's computed:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

text_a = patent_a['abstract']
text_b = patent_b['abstract']

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
# Shape: (2, 1000) - two documents, up to 1000 features

tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
```

**TF-IDF explained in depth:**

TF-IDF = Term Frequency × Inverse Document Frequency

**Term Frequency (TF):**
```
TF(word, doc) = (# times word appears in doc) / (total words in doc)

Example:
  "wireless" appears 5 times in 100-word abstract
  TF("wireless") = 5/100 = 0.05
```

**Inverse Document Frequency (IDF):**
```
IDF(word, corpus) = log(total docs / docs containing word)

Example:
  "wireless" appears in 15,000 of 200,000 patents
  IDF("wireless") = log(200000/15000) = log(13.33) = 2.59
  
  "the" appears in 199,000 of 200,000 patents
  IDF("the") = log(200000/199000) = log(1.005) = 0.005
```

**TF-IDF score:**
```
TF-IDF("wireless") = 0.05 × 2.59 = 0.13
TF-IDF("the") = 0.03 × 0.005 = 0.00015
```

Result: Technical terms get high scores, common words get low scores.

**Our configuration choices:**

**max_features=1000:**
- Limit vocabulary to top 1000 terms
- Reduces dimensionality (faster computation)
- Captures most important words
- Full vocabulary might be 50,000+ words

**stop_words='english':**
- Remove "the", "is", "and", "of", etc.
- These carry no semantic meaning
- Reduces noise in similarity calculation

**ngram_range=(1, 2):**
- Include unigrams (single words): "wireless", "charging"
- Include bigrams (two-word phrases): "wireless charging"
- Captures common phrases
- Bigrams help: "machine learning" is different from "machine" + "learning"

**min_df=1, max_df=0.95:**
- min_df=1: Include terms appearing in at least 1 document (no filtering)
- max_df=0.95: Exclude terms appearing in >95% of documents (too common)

**Why TF-IDF AND embeddings?**

They capture different aspects:

**Example 1: Synonyms**
- Patent A: "wireless power transfer"
- Patent B: "cordless energy transmission"

TF-IDF similarity: ~0.15 (different words)
Embedding similarity: ~0.85 (same concept)

**Example 2: Exact keyword match**
- Patent A: "lithium-ion battery charging system"
- Patent B: "lithium-ion battery testing apparatus"

TF-IDF similarity: ~0.70 (share "lithium-ion battery")
Embedding similarity: ~0.45 (related but different purposes)

Having both gives the model more signal!

**Computational cost:**

TF-IDF is expensive:
1. Tokenize both texts: ~0.5ms
2. Build vocabulary: ~1ms
3. Compute TF-IDF matrix: ~2ms
4. Cosine similarity: ~0.1ms
Total: ~3.5ms per pair

For 57K pairs: 3.5ms × 57,000 = 200 seconds = 3.3 minutes

### Feature 3: Jaccard Similarity

**What it captures:** Simple word overlap ratio

**How it's computed:**

```python
text_a = patent_a['abstract'].lower()
text_b = patent_b['abstract'].lower()

words_a = set(text_a.split())
words_b = set(text_b.split())

intersection = len(words_a & words_b)
union = len(words_a | words_b)

jaccard_sim = intersection / (union + 1e-8)
```

**Mathematical definition:**

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Where:
  A ∩ B = intersection (words in both)
  A ∪ B = union (words in either)
```

**Example:**

Patent A abstract: "a wireless charging system for mobile devices"
- Words: {a, wireless, charging, system, for, mobile, devices}
- Count: 7 unique words

Patent B abstract: "wireless power transfer for mobile phones"
- Words: {wireless, power, transfer, for, mobile, phones}
- Count: 6 unique words

Intersection: {wireless, for, mobile}
- Count: 3

Union: {a, wireless, charging, system, for, mobile, devices, power, transfer, phones}
- Count: 10

Jaccard = 3/10 = 0.30

**Range:** [0, 1]
- 0.0: No words in common
- 1.0: Identical word sets

**Why Jaccard when we have TF-IDF?**

Jaccard is cruder but captures different signal:
- TF-IDF weights words by importance
- Jaccard treats all words equally

Sometimes rare word overlap is more meaningful:
```
Patent A: "piezoelectric energy harvesting nanosystem"
Patent B: "piezoelectric vibration sensor"

Jaccard: 1/7 = 0.14 (only "piezoelectric" in common)
```

But "piezoelectric" is a strong signal! Jaccard doesn't weight it, but its presence in the feature vector (even with low Jaccard score) combined with high TF-IDF score helps the model learn.

**Why lowercase?**

```python
text_a.lower()
```

"Wireless" and "wireless" are the same word. Case-insensitive matching prevents:
- Set 1: {"Wireless", "charging"}
- Set 2: {"wireless", "Charging"}
- Intersection: {} (empty!)

After lowercasing:
- Set 1: {"wireless", "charging"}
- Set 2: {"wireless", "charging"}
- Intersection: {"wireless", "charging"} ✓

**Computational efficiency:**

```python
# Naive: Compare every pair of words
for word_a in words_a:
    for word_b in words_b:
        if word_a == word_b:
            count += 1

# Smart: Use set operations (hash table based)
intersection = words_a & words_b  # O(min(|A|, |B|))
```

Set intersection uses hash tables:
- Average case: O(min(n, m))
- For 100-word abstracts: ~100 operations
- Very fast: <0.1ms per pair

### Feature 4: Claim Count Ratio

**What it captures:** Structural similarity in patent scope

**How it's computed:**

```python
claims_a = len(patent_a.get('claims', []))
claims_b = len(patent_b.get('claims', []))

claim_ratio = min(claims_a, claims_b) / (max(claims_a, claims_b) + 1e-8)
```

**Why claim count matters:**

Patents have 1-100+ claims. The number indicates scope:
- **Few claims (1-5):** Narrow invention, specific application
- **Medium claims (10-20):** Standard patent, multiple embodiments
- **Many claims (50+):** Broad invention, many variations

Similar patents tend to have similar claim counts.

**Example:**

Patent A: 15 claims (standard battery charger)
Patent B: 12 claims (wireless charging pad)
Ratio: min(15, 12) / max(15, 12) = 12/15 = 0.80 (similar)

Patent A: 15 claims (battery charger)
Patent C: 3 claims (simple mechanical latch)
Ratio: min(15, 3) / max(15, 3) = 3/15 = 0.20 (different)

**Why min/max ratio instead of absolute difference?**

**Absolute difference:**
```python
claim_diff = abs(claims_a - claims_b)
```
Problem: Not normalized
- |15 - 12| = 3
- |50 - 47| = 3
Both give 3, but 15 vs 12 is more different relatively.

**Ratio:**
```python
ratio = min/max
```
- 12/15 = 0.80 (20% difference)
- 47/50 = 0.94 (6% difference)
Captures relative similarity.

**Range:** [0, 1]
- 1.0: Same number of claims
- 0.5: One has 2× as many
- 0.1: One has 10× as many

**Why +1e-8 in denominator?**

Safety against division by zero:
```python
max(claims_a, claims_b) + 1e-8
```

If both patents have 0 claims (corrupted data):
- Without: 0/0 = NaN → crash
- With: 0/(0+1e-8) = 0 → safe

1e-8 is negligible (0.00000001) but prevents NaN.

### Feature 5: Abstract Length Ratio

**What it captures:** Document verbosity similarity

**How it's computed:**

```python
len_a = len(patent_a['abstract'].split())
len_b = len(patent_b['abstract'].split())

length_ratio = min(len_a, len_b) / (max(len_a, len_b) + 1e-8)
```

**Why abstract length matters:**

Abstracts vary in length:
- Short (50-100 words): Concise, mechanical/electrical patents
- Medium (150-250 words): Standard, most patents
- Long (300-500 words): Complex, biotech/pharmaceutical patents

Patents in the same domain tend to have similar lengths.

**Example distributions:**

Mechanical engineering:
- Average: 120 words
- Range: 80-160 words

Biotechnology:
- Average: 280 words
- Range: 200-400 words

Two mechanical patents:
- Patent A: 115 words
- Patent B: 130 words
- Ratio: 115/130 = 0.88 (similar)

Mechanical vs biotech:
- Patent A: 115 words
- Patent B: 310 words
- Ratio: 115/310 = 0.37 (different)

**Why word count instead of character count?**

```python
# Word count
words = text.split()
count = len(words)

# vs character count
count = len(text)
```

Word count is more meaningful:
- "pharmaceutical" (15 chars) = 1 word
- "a b c d e f g h" (15 chars) = 8 words

Character count would conflate long words with many short words.

**Computational cost:**

```python
text.split()  # O(n) where n is number of characters
len(words)    # O(1)
```

For 200-word abstract (~1000 chars): ~0.01ms

### Feature 6: Year Difference (Normalized)

**What it captures:** Temporal proximity

**How it's computed:**

```python
year_a = patent_a.get('year', 2023)
year_b = patent_b.get('year', 2023)

year_diff_normalized = 1.0 / (1.0 + abs(year_a - year_b))
```

**Why year matters:**

Technology evolves in waves:
- 2010-2012: Tablet computing boom
- 2015-2017: Machine learning explosion
- 2018-2020: Autonomous vehicle patents

Patents from the same year or consecutive years:
- Build on same prior art
- Address same technological problems
- Use similar terminology
- More likely to be related

**Example:**

Patent A: 2020
Patent B: 2020
Diff: |2020 - 2020| = 0
Feature: 1.0 / (1.0 + 0) = 1.0

Patent A: 2020
Patent C: 2021
Diff: |2020 - 2021| = 1
Feature: 1.0 / (1.0 + 1) = 0.5

Patent A: 2020
Patent D: 2015
Diff: |2020 - 2015| = 5
Feature: 1.0 / (1.0 + 5) = 0.167

**Mathematical properties:**

Function: `f(d) = 1/(1+d)` where d = year difference

Properties:
- Same year (d=0): f(0) = 1.0
- 1 year apart: f(1) = 0.5
- 2 years: f(2) = 0.33
- 5 years: f(5) = 0.167
- 10 years: f(10) = 0.091
- 20 years: f(20) = 0.048

Decay curve: Rapid drop initially, then gradual decline.

**Why this formula instead of alternatives?**

**Alternative 1: Negative absolute difference**
```python
feature = -abs(year_a - year_b)
```
Problem: Unbounded negative values, not normalized

**Alternative 2: Exponential decay**
```python
feature = np.exp(-abs(year_a - year_b) / 5)
```
Better, but:
- Requires tuning decay constant
- 1/(1+d) is simpler and works well

**Alternative 3: Linear within window**
```python
diff = abs(year_a - year_b)
feature = max(0, 1 - diff/10)  # 0 if >10 years apart
```
Harsh cutoff at 10 years

Our formula: Smooth decay, no cutoffs, simple.

**Why default to 2023?**

```python
year_a = patent_a.get('year', 2023)
```

Some patents missing year in data (rare ~0.5%):
- Parsing errors from USPTO data
- Corrupted records

Default to 2023 (recent):
- Makes them similar to other recent patents
- Better than None (would crash)
- Could use median year (2020) but 2023 is reasonable

### Feature 7: Assignee Match (Binary)

**What it captures:** Same company/organization

**How it's computed:**

```python
assignee_a = patent_a.get('assignee', '').strip().lower()
assignee_b = patent_b.get('assignee', '').strip().lower()

assignee_match = float(
    assignee_a == assignee_b and assignee_a != ''
)
```

**Why assignee matters:**

Companies file patents in clusters:
- Apple: iPhone-related patents
- Google: Search and ML patents
- Pfizer: Pharmaceutical patents

Two patents from same company:
- Often in same product line
- Build on same internal R&D
- May cite each other
- Higher likelihood of similarity

**Example:**

Patent A: assignee = "Apple Inc."
Patent B: assignee = "Apple Inc."
Match: 1.0

Patent A: assignee = "Apple Inc."
Patent C: assignee = "Samsung Electronics"
Match: 0.0

**Why binary instead of string similarity?**

Could use edit distance:
```python
similarity = 1 - edit_distance("Apple Inc.", "Apple Computer Inc.") / max_len
```

But assignee names are standardized in USPTO data:
- "Apple Inc." (always exact)
- "Google LLC" (always exact)

Either exact match (1.0) or different company (0.0).

**Why .strip().lower()?**

Normalize for comparison:
```python
"Apple Inc. " vs "Apple Inc." # Extra space
"APPLE INC." vs "Apple Inc."   # Different case
```

After normalization:
```python
"apple inc." == "apple inc."  # ✓ Match
```

**Why check assignee_a != ''?**

Without this check:
```python
assignee_a = ""
assignee_b = ""
match = ("" == "")  # True!
```

Two patents with missing assignee would match, which is wrong. We want:
```python
match = ("" == "" and "" != "")  # False ✓
```

**Binary encoding:**

```python
float(True) = 1.0
float(False) = 0.0
```

Converts boolean to float for neural network input.

### Feature 8: CPC Code Overlap (Jaccard)

**What it captures:** Patent classification similarity

**How it's computed:**

```python
cpc_a = set(patent_a.get('cpc_codes', []))
cpc_b = set(patent_b.get('cpc_codes', []))

intersection = len(cpc_a & cpc_b)
union = len(cpc_a | cpc_b)

cpc_jaccard = intersection / (union + 1e-8)
```

**What are CPC codes?**

Cooperative Patent Classification: Hierarchical taxonomy for patents.

**Structure:**
```
H04W = Wireless communication networks
  H04W 4/00 = Services for wireless communication
    H04W 4/80 = Services using short range communication
      H04W 4/80 = Services using NFC
```

**Levels:**
- Section: H (Electricity)
- Class: H04 (Electric communication technique)
- Subclass: H04W (Wireless communication)
- Group: H04W 4/00 (Services)
- Subgroup: H04W 4/80 (Short range)

**Example patent CPC codes:**

Patent A (wireless charging):
```
['H02J 50/10',  # Wireless power transfer
 'H02J 7/00',   # Circuit arrangements for charging
 'H04W 4/80']   # Short range wireless
```

Patent B (NFC payment):
```
['H04W 4/80',   # Short range wireless
 'G06Q 20/32',  # Payment protocols
 'H04L 9/32']   # Digital signatures
```

Intersection: {'H04W 4/80'} = 1
Union: {'H02J 50/10', 'H02J 7/00', 'H04W 4/80', 'G06Q 20/32', 'H04L 9/32'} = 5
Jaccard: 1/5 = 0.20

**Why Jaccard for CPC codes?**

Each patent has 1-10 CPC codes. Jaccard measures:
- How many codes they share (intersection)
- Relative to total unique codes (union)

**High overlap (>0.5):**
- Same technological domain
- Likely related inventions

**Medium overlap (0.2-0.5):**
- Adjacent technologies
- Potentially related

**Low overlap (<0.2):**
- Different domains
- Unlikely to be similar

**Hierarchical consideration:**

We could use hierarchical matching:
```python
# Match at different levels
if code_a[:4] == code_b[:4]:  # Same subclass
    similarity += 1.0
elif code_a[:3] == code_b[:3]:  # Same class
    similarity += 0.5
```

But simple Jaccard works well empirically and is simpler.

### Feature 9: Max Claim Embedding Similarity

**What it captures:** Fine-grained claim-level similarity

**How it's computed:**

```python
claims_a = patent_a.get('claims', [])[:10]  # First 10
claims_b = patent_b.get('claims', [])[:10]

# Extract claim text
texts_a = [get_claim_text(c) for c in claims_a]
texts_b = [get_claim_text(c) for c in claims_b]

# Embed all claims
embeddings_a = embedder.encode(texts_a)  # (n, 768)
embeddings_b = embedder.encode(texts_b)  # (m, 768)

# Compute pairwise similarities
max_sim = 0.0
for emb_a in embeddings_a:
    for emb_b in embeddings_b:
        sim = cosine_similarity(emb_a, emb_b)
        max_sim = max(max_sim, sim)

feature = max_sim
```

**Why claim-level similarity?**

Abstracts summarize the invention, but claims define legal scope.

**Scenario: Overlapping claims**

Patent A (10 claims):
- Claim 1: Wireless charging base station
- Claim 2: Base station with multiple coils
- Claim 3: Foreign object detection in base station
- Claims 4-10: Various base station features

Patent B (8 claims):
- Claim 1: Mobile device receiver for wireless charging
- Claim 2: Receiver with rectification circuit
- Claim 3: Foreign object detection in receiver
- Claims 4-8: Various receiver features

Abstract similarity: 0.65 (related but different components)

Claim-level similarities:
- A.Claim3 vs B.Claim3: 0.92 (both about foreign object detection)
- A.Claim1 vs B.Claim1: 0.45 (base station vs receiver)
- ...

Max claim similarity: 0.92

This high max similarity signals that despite different overall focus (base station vs receiver), they share a critical technical element (foreign object detection).

**Why max instead of average?**

**Average approach:**
```python
avg_sim = np.mean([cosine(a, b) for a in embs_a for b in embs_b])
```

Problem: Diluted by many low-similarity pairs
- 9 irrelevant claim pairs: similarity ~0.15
- 1 highly relevant pair: similarity ~0.95
- Average: (9×0.15 + 1×0.95) / 10 = 0.23

**Max approach:**
```python
max_sim = max([cosine(a, b) for a in embs_a for b in embs_b])
```

Captures: 0.95

Interpretation: "At least one pair of claims is highly similar" → patents likely related.

**Why limit to first 10 claims?**

**Computational cost:**

With all claims:
- Patent A: 25 claims
- Patent B: 30 claims
- Pairs: 25 × 30 = 750 comparisons

For each comparison:
- Encode claim: ~50ms (PatentSBERTa)
- Cosine similarity: 0.001ms

Total: 25×50ms + 30×50ms + 750×0.001ms = 2750ms = 2.75 seconds per patent pair!

For 57K pairs: 2.75s × 57,000 = 157,000s = 44 hours!

**With first 10:**
- Patent A: 10 claims
- Patent B: 10 claims
- Pairs: 10 × 10 = 100 comparisons

Total: 10×50ms + 10×50ms + 100×0.001ms = 1000ms = 1 second per pair

For 57K pairs: 1s × 57,000 = 57,000s = 16 hours

Still expensive but tractable.

**Why first 10 specifically?**

Patent claims are ordered:
- Claim 1: Broadest, most important
- Claim 2-5: Major embodiments
- Claims 6-15: Variations and details
- Claims 16+: Minor variations

First 10 claims capture the core invention. Later claims are often minor tweaks.

**Optimization: Caching**

We cache embeddings:
```python
claim_embedding_cache = {}

def get_claim_embedding(claim_text):
    if claim_text not in claim_embedding_cache:
        claim_embedding_cache[claim_text] = embedder.encode(claim_text)
    return claim_embedding_cache[claim_text]
```

If claim appears in multiple patents (rare but possible for standard phrases), we don't re-embed.

### Feature 10: Title Similarity

**What it captures:** High-level conceptual similarity

**How it's computed:**

```python
title_a = patent_a.get('title', '')
title_b = patent_b.get('title', '')

if title_a and title_b:
    emb_a = embedder.encode(title_a)  # (768,)
    emb_b = embedder.encode(title_b)  # (768,)
    
    title_sim = cosine_similarity(emb_a, emb_b)
else:
    title_sim = 0.0

feature = title_sim
```

**Why title similarity?**

Titles are concise (5-20 words) and descriptive.

**Example:**

Patent A: "Wireless Power Transfer System Using Magnetic Resonance"
Patent B: "System for Wireless Energy Transmission via Resonant Coupling"

Title embedding similarity: ~0.89 (same concept, different wording)

Patent A: "Wireless Power Transfer System Using Magnetic Resonance"
Patent C: "Method for Diagnosing Heart Disease Using Machine Learning"

Title embedding similarity: ~0.12 (completely different)

**Why embeddings instead of keyword matching?**

Keyword matching:
```python
words_a = set(title_a.lower().split())
words_b = set(title_b.lower().split())
jaccard = len(words_a & words_b) / len(words_a | words_b)
```

Example:
```
Title A: "Wireless Power Transfer"
Title B: "Cordless Energy Transmission"

Words A: {wireless, power, transfer}
Words B: {cordless, energy, transmission}
Intersection: {} (empty!)
Jaccard: 0.0
```

Embeddings:
```
Embedding similarity: 0.85 (understands synonyms)
```

**Embedding captures semantic similarity:**
- "wireless" ≈ "cordless"
- "power" ≈ "energy"
- "transfer" ≈ "transmission"

**Computational cost:**

Encoding two titles:
- Title A: ~10 words → tokenize → encode → 15ms
- Title B: ~10 words → tokenize → encode → 15ms
- Cosine similarity: 0.001ms
- Total: ~30ms per pair

For 57K pairs: 30ms × 57,000 = 1,710,000ms = 28 minutes

Reasonable for one-time feature extraction.

---

## Feature Computation Implementation

Now let's look at the actual code that computes all 10 features.

```python
class FeatureComputer:
    """
    Computes 10 engineered features for patent pairs.
    
    Loaded resources:
    - Patent database (200K patents with metadata)
    - Pre-computed embeddings (586 MB)
    - PatentSBERTa model (for runtime embedding)
    
    Features computed:
    1. PatentSBERTa cosine similarity
    2. TF-IDF cosine similarity
    3. Jaccard similarity
    4. Claim count ratio
    5. Abstract length ratio
    6. Year difference (normalized)
    7. Assignee match (binary)
    8. CPC code overlap (Jaccard)
    9. Max claim embedding similarity
    10. Title similarity
    """
    
    def __init__(
        self,
        patents_path: str,
        embeddings_path: str,
        patent_ids_path: str
    ):
        """
        Initialize feature computer.
        
        Args:
            patents_path: Path to JSONL with full patent data
            embeddings_path: Path to numpy embeddings file
            patent_ids_path: Path to patent ID list
        """
        
        # Load patents into memory
        print("Loading patents...")
        self.patents = {}
        with open(patents_path, 'r') as f:
            for line in tqdm(f, total=200000):
                p = json.loads(line)
                self.patents[p['patent_id']] = p
        
        print(f"Loaded {len(self.patents):,} patents")
        
        # Load embeddings (memory-mapped for efficiency)
        print("Loading embeddings...")
        self.embeddings = np.load(embeddings_path, mmap_mode='r')
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Load patent IDs
        with open(patent_ids_path, 'r') as f:
            self.patent_ids = json.load(f)
        
        # Create ID to index mapping
        self.id_to_idx = {pid: i for i, pid in enumerate(self.patent_ids)}
        
        # Initialize embedder for runtime encoding
        from src.embeddings.patent_sberta import PatentEmbedder
        self.embedder = PatentEmbedder()
        
        # Claim embedding cache (to avoid re-encoding same claims)
        self.claim_cache = {}
    
    def compute_features_for_pair(
        self,
        patent_a_id: str,
        patent_b_id: str
    ) -> np.ndarray:
        """
        Compute all 10 features for a patent pair.
        
        Args:
            patent_a_id: First patent ID
            patent_b_id: Second patent ID
        
        Returns:
            numpy array of shape (10,) with feature values
        """
        
        features = []
        
        # Load patent data
        patent_a = self.patents.get(patent_a_id)
        patent_b = self.patents.get(patent_b_id)
        
        if not patent_a or not patent_b:
            # Missing data - return zero features
            return np.zeros(10, dtype=np.float32)
        
        # Feature 1: PatentSBERTa cosine similarity
        emb_a = self.embeddings[self.id_to_idx[patent_a_id]]
        emb_b = self.embeddings[self.id_to_idx[patent_b_id]]
        
        cosine_sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        features.append(float(cosine_sim))
        
        # Feature 2: TF-IDF cosine similarity
        text_a = patent_a.get('abstract', '')
        text_b = patent_b.get('abstract', '')
        
        if text_a and text_b:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
                tfidf_sim = cosine_similarity(
                    tfidf_matrix[0:1],
                    tfidf_matrix[1:2]
                )[0][0]
                features.append(float(tfidf_sim))
            except:
                # TF-IDF can fail if text is too short
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Feature 3: Jaccard similarity
        if text_a and text_b:
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            
            jaccard_sim = intersection / (union + 1e-8)
            features.append(float(jaccard_sim))
        else:
            features.append(0.0)
        
        # Feature 4: Claim count ratio
        claims_a = len(patent_a.get('claims', []))
        claims_b = len(patent_b.get('claims', []))
        
        if claims_a > 0 and claims_b > 0:
            claim_ratio = min(claims_a, claims_b) / max(claims_a, claims_b)
        else:
            claim_ratio = 0.0
        features.append(float(claim_ratio))
        
        # Feature 5: Abstract length ratio
        len_a = len(text_a.split()) if text_a else 0
        len_b = len(text_b.split()) if text_b else 0
        
        if len_a > 0 and len_b > 0:
            length_ratio = min(len_a, len_b) / max(len_a, len_b)
        else:
            length_ratio = 0.0
        features.append(float(length_ratio))
        
        # Feature 6: Year difference (normalized)
        year_a = patent_a.get('year', 2023)
        year_b = patent_b.get('year', 2023)
        
        year_feature = 1.0 / (1.0 + abs(year_a - year_b))
        features.append(float(year_feature))
        
        # Feature 7: Assignee match
        assignee_a = patent_a.get('assignee', '').strip().lower()
        assignee_b = patent_b.get('assignee', '').strip().lower()
        
        assignee_match = float(
            assignee_a == assignee_b and assignee_a != ''
        )
        features.append(assignee_match)
        
        # Feature 8: CPC code overlap
        cpc_a = set(patent_a.get('cpc_codes', []))
        cpc_b = set(patent_b.get('cpc_codes', []))
        
        if cpc_a and cpc_b:
            cpc_jaccard = len(cpc_a & cpc_b) / (len(cpc_a | cpc_b) + 1e-8)
        else:
            cpc_jaccard = 0.0
        features.append(float(cpc_jaccard))
        
        # Feature 9: Max claim embedding similarity
        claims_a_data = patent_a.get('claims', [])[:10]
        claims_b_data = patent_b.get('claims', [])[:10]
        
        if claims_a_data and claims_b_data:
            # Extract claim texts
            texts_a = [self._get_claim_text(c) for c in claims_a_data]
            texts_b = [self._get_claim_text(c) for c in claims_b_data]
            
            # Embed (with caching)
            embs_a = [self._get_cached_embedding(t) for t in texts_a if t]
            embs_b = [self._get_cached_embedding(t) for t in texts_b if t]
            
            if embs_a and embs_b:
                # Compute max pairwise similarity
                max_sim = 0.0
                for ea in embs_a:
                    for eb in embs_b:
                        sim = np.dot(ea, eb) / (
                            np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8
                        )
                        max_sim = max(max_sim, sim)
                features.append(float(max_sim))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Feature 10: Title similarity
        title_a = patent_a.get('title', '')
        title_b = patent_b.get('title', '')
        
        if title_a and title_b:
            emb_a = self._get_cached_embedding(title_a)
            emb_b = self._get_cached_embedding(title_b)
            
            title_sim = np.dot(emb_a, emb_b) / (
                np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
            )
            features.append(float(title_sim))
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_claim_text(self, claim) -> str:
        """Extract text from claim (handles dict or string)."""
        if isinstance(claim, dict):
            return claim.get('text', '')
        return str(claim)
    
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        if text not in self.claim_cache:
            self.claim_cache[text] = self.embedder.encode(text)
        return self.claim_cache[text]
```

**Memory optimization: mmap_mode='r'**

```python
self.embeddings = np.load(embeddings_path, mmap_mode='r')
```

This memory-maps the file instead of loading it entirely:
- Without mmap: Load 586 MB into RAM
- With mmap: Map file on disk, load pages on demand
- Benefit: Lower memory footprint
- Trade-off: Slightly slower access (disk vs RAM)

For our use case (random access to embeddings), mmap works great.

---

## Running Feature Extraction

```python
def main():
    """Extract features for all training pairs."""
    
    # Initialize computer
    computer = FeatureComputer(
        patents_path='data/sampled/patents_sampled.jsonl',
        embeddings_path='data/embeddings/patent_embeddings.npy',
        patent_ids_path='data/embeddings/patent_ids.json'
    )
    
    # Load training pairs
    with open('data/training/train_pairs.json', 'r') as f:
        train_pairs = json.load(f)  # List of ((id_a, id_b), label)
    
    # Extract features
    X_train = []
    y_train = []
    
    for (id_a, id_b), label in tqdm(train_pairs, desc="Extracting features"):
        features = computer.compute_features_for_pair(id_a, id_b)
        X_train.append(features)
        y_train.append(label)
    
    X_train = np.array(X_train)  # (39979, 10)
    y_train = np.array(y_train)  # (39979,)
    
    # Save
    np.save('data/features/train_features_v2.X.npy', X_train)
    np.save('data/features/train_features_v2.y.npy', y_train)
    
    # Repeat for val and test...
```

**Time estimate:**

For 57,114 total pairs (train + val + test):
- Features 1-8: ~0.05 seconds per pair
- Feature 9 (claim embeddings): ~1 second per pair (expensive)
- Feature 10 (title): ~0.03 seconds per pair

Average: ~1.1 seconds per pair
Total: 1.1s × 57,114 = 62,825 seconds = 17.5 hours

We run this once and save the features.

---

(Continuing to model training in same file...)

# FILE 4: `src/app/pytorch_classifier.py`

## Purpose and Context

Now we have:
- ✓ Training features: (39,979, 10)
- ✓ Training labels: (39,979,) with 0/1
- ✓ Validation features: (8,567, 10)
- ✓ Validation labels: (8,567,)

Goal: Train a neural network to predict similarity from these 10 features.

**Why neural network instead of simpler models?**

We tried multiple approaches:

**1. Logistic Regression:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```
Result: 85.3% accuracy, 91.2% ROC-AUC

Problem: Linear decision boundary. Can't capture feature interactions.

**2. Random Forest:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```
Result: 88.7% accuracy, 94.5% ROC-AUC

Better! But still not optimal.

**3. Gradient Boosting:**
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200)
model.fit(X_train, y_train)
```
Result: 89.2% accuracy, 95.1% ROC-AUC

Even better, but plateaus.

**4. PyTorch Neural Network (our final choice):**
Result: **91.7% accuracy, 97.2% ROC-AUC**

Why better?
- Non-linear transformations (ReLU activations)
- Feature interactions learned automatically
- Batch normalization stabilizes training
- Dropout prevents overfitting
- Residual connections enable deeper architecture

---

(Continue in Part 3 with full PyTorch training details...)

