# Complete Code Walkthrough Script - File by File

## Overview
This guide walks through the codebase in the exact order of the data → model → inference pipeline. For each file, I'll tell you what to explain and which code sections to focus on.

---

# PART 1: DATA PREPROCESSING PIPELINE

---

## FILE 1: `scripts/data/preprocessing/generate_embeddings.py`

**What to say:**
> "Let's start with the most critical preprocessing step - generating embeddings. This file takes 200,000 patents and converts them into 768-dimensional semantic vectors using PatentSBERTa. This took 11 hours to run."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 1-30: Imports and Setup
**What to explain:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
```

**Say:**
> "We're using sentence-transformers library which gives us access to PatentSBERTa - a BERT model that was fine-tuned specifically on 1.2 million patents. This model understands patent language and technical terminology."

#### Lines 32-56: `get_patent_text()` function
**Navigate to this function and explain:**

```python
def get_patent_text(patent: dict) -> str:
    """Extract text from patent for embedding."""
    if patent.get("abstract"):
        return patent["abstract"][:500]
    if patent.get("summary"):
        return patent["summary"][:500]
    claims = patent.get("claims", [])
    if claims:
        first = claims[0]
        if isinstance(first, dict):
            return (first.get("text") or "")[:500]
        if isinstance(first, str):
            return first[:500]
    return ""
```

**Say:**
> "This function extracts the text we'll embed. Notice the priority order: we prefer abstracts first because they're concise but comprehensive. We truncate to 500 characters because PatentSBERTa has a maximum context window of 512 tokens. If there's no abstract, we fall back to the summary, then the first claim."

**Point out:**
- The fallback logic (abstract → summary → claims)
- The 500 character truncation (why: model limitation)
- Returns empty string if nothing found (handles missing data)

#### Lines 59-86: `generate_embeddings_batch()` function
**This is the CORE function - spend time here:**

```python
def generate_embeddings_batch(
    patents: List[dict],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for a batch of patents."""
    
    # Extract text from all patents
    texts = []
    for patent in patents:
        text = get_patent_text(patent)
        texts.append(text)
    
    # Batch encode with the model
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # We'll normalize during search
    )
    
    return embeddings  # Shape: (N, 768)
```

**Say:**
> "Here's where the magic happens. We're calling model.encode() which runs the patent text through PatentSBERTa's BERT architecture. This processes the text through 12 transformer layers with multi-head attention, and outputs a 768-dimensional dense vector that captures the semantic meaning."

**Key points to emphasize:**
1. **batch_size=32**: Process 32 patents at once for GPU efficiency
2. **convert_to_numpy=True**: Get numpy arrays, not torch tensors
3. **normalize_embeddings=False**: We'll normalize later during cosine similarity search
4. **Output shape**: (N, 768) where N is number of patents, 768 is embedding dimension

**Show the mathematical concept:**
```
Input: "A method for wireless charging using magnetic resonance..."
  ↓
PatentSBERTa (BERT encoder with 12 layers)
  ↓
[CLS] token representation
  ↓
Output: [0.234, -0.567, 0.123, ..., 0.891]  (768 numbers)
```

#### Lines 90-150: `main()` function
**Navigate here to show the overall process:**

```python
def main():
    # Load the model
    print("Loading PatentSBERTa model...")
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    # Use MPS (Apple GPU) if available
    if torch.backends.mps.is_available():
        model = model.to('mps')
        print("Using Apple MPS (GPU acceleration)")
    
    # Load patents
    print("Loading patents from JSONL...")
    patents = []
    with open('data/sampled/patents_sampled.jsonl', 'r') as f:
        for line in tqdm(f, total=200000):
            patents.append(json.loads(line))
    
    # Generate embeddings in batches
    all_embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(patents), batch_size)):
        batch = patents[i:i+batch_size]
        batch_embeddings = generate_embeddings_batch(batch, model, batch_size)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)  # Shape: (200000, 768)
    
    # Save embeddings
    np.save('data/embeddings/patent_embeddings.npy', embeddings)
    
    # Save patent IDs for indexing
    patent_ids = [p['patent_id'] for p in patents]
    with open('data/embeddings/patent_ids.json', 'w') as f:
        json.dump(patent_ids, f)
```

**Say:**
> "The main function orchestrates everything. First, we load the PatentSBERTa model from HuggingFace. Then we detect if we have Apple Silicon and use MPS for GPU acceleration - this makes it about 3x faster than CPU. We load all 200,000 patents from the JSONL file, process them in batches of 32, and save the resulting 200,000 x 768 embedding matrix as a numpy array. This file is 586 megabytes."

**Key outputs to point out:**
1. `patent_embeddings.npy`: The 200K x 768 embedding matrix (~586 MB)
2. `patent_ids.json`: Maps index → patent ID (for lookup)

**Time complexity:**
> "This took 11 hours on my machine because we're running a transformer model with 110 million parameters 200,000 times. That's why we save the embeddings - we never want to regenerate them."

---

## FILE 2: `scripts/training/extract_citation_pairs.py`

**What to say:**
> "Now that we have embeddings, we need training data for our similarity model. This file creates labeled pairs of patents by analyzing the citation graph - if patent A cites patent B, they're definitely related. This is our positive training signal."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 1-25: Imports and Overview
**Show the docstring:**

```python
"""
Extract Citation Pairs for Training

Creates positive and negative training pairs from patent citation data.

Positive pairs: Patents that cite each other (confirmed relationship)
Negative pairs: Random patents that don't cite each other (assumed unrelated)

This is "positive-unlabeled" (PU) learning because:
- Positives are confirmed (citations are explicit)
- Negatives are unlabeled (no citation doesn't guarantee no relationship)
"""
```

**Say:**
> "This is a classic positive-unlabeled learning problem. We KNOW two patents are related if one cites the other - that's our ground truth. But we can't definitively say two patents are unrelated just because there's no citation. They might be similar but in different patent families. So our negative examples are actually 'unlabeled' - we assume they're different but can't be 100% sure."

#### Lines 64-113: `generate_negative_pairs()` function
**This is the key algorithmic decision:**

```python
def generate_negative_pairs(
    positive_pairs: List[Tuple[str, str]],
    citation_graph: Dict[str, List[str]],
    all_patent_ids: List[str],
    ratio: int = 1,
    random_seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Generate negative pairs by random sampling.
    
    For each positive pair (A, B):
    - Sample random patent C where:
      1. C != B (not the same patent)
      2. A does not cite C (no relationship)
    - Create negative pair (A, C)
    
    Args:
        ratio: Number of negative pairs per positive pair
    """
    random.seed(random_seed)
    negative_pairs = []
    
    for citing_patent, cited_patent in positive_pairs:
        # Get all patents that citing_patent cites
        cited_by_citing = set(citation_graph.get(citing_patent, []))
        
        # Sample patents NOT in that set
        attempts = 0
        while attempts < 100:  # Avoid infinite loop
            candidate = random.choice(all_patent_ids)
            
            if candidate != citing_patent and candidate not in cited_by_citing:
                negative_pairs.append((citing_patent, candidate))
                break
            
            attempts += 1
    
    return negative_pairs
```

**Say:**
> "Here's our negative sampling strategy. For each positive pair like (Patent A cites Patent B), we randomly sample a patent C that A does NOT cite, creating the negative pair (A, C). We use a 1:1 ratio - equal numbers of positive and negative pairs - to keep our classes balanced. This is important because imbalanced data would bias the model."

**Key points:**
1. **ratio=1**: 1 negative per positive (balanced classes)
2. **Random sampling**: Ensures diversity
3. **Exclusion logic**: Can't sample from citation set
4. **Attempt limit**: Safety valve to prevent infinite loops

**Show the numbers:**
> "We extracted 28,557 positive pairs from the citation graph. With 1:1 ratio, we generated 28,557 negative pairs, giving us 57,114 total training pairs."

#### Lines 150-220: Train/Val/Test Split
**Navigate to the split function:**

```python
def create_splits(
    positive_pairs: List[Tuple],
    negative_pairs: List[Tuple],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Stratified split maintaining 50/50 class balance.
    
    Total: 57,114 pairs
    Split:
    - Train: 70% = 39,979 pairs
    - Val: 15% = 8,567 pairs  
    - Test: 15% = 8,568 pairs
    
    Each split has exactly 50% positive, 50% negative.
    """
    # Shuffle with fixed seed
    random.seed(random_seed)
    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)
    
    # Calculate split points
    n_positive = len(positive_pairs)
    n_negative = len(negative_pairs)
    
    # Split positives
    val_size = int(n_positive * val_ratio)
    test_size = int(n_positive * test_ratio)
    
    pos_train = positive_pairs[:(n_positive - val_size - test_size)]
    pos_val = positive_pairs[(n_positive - val_size - test_size):(n_positive - test_size)]
    pos_test = positive_pairs[(n_positive - test_size):]
    
    # Split negatives (same way)
    neg_train = negative_pairs[:(n_negative - val_size - test_size)]
    neg_val = negative_pairs[(n_negative - val_size - test_size):(n_negative - test_size)]
    neg_test = negative_pairs[(n_negative - test_size):]
    
    # Combine and label
    train_pairs = [(p, 1) for p in pos_train] + [(n, 0) for n in neg_train]
    val_pairs = [(p, 1) for p in pos_val] + [(n, 0) for n in neg_val]
    test_pairs = [(p, 1) for p in pos_test] + [(n, 0) for n in neg_test]
    
    return train_pairs, val_pairs, test_pairs
```

**Say:**
> "We do a stratified 70/15/15 split. Stratified means each split maintains the same class distribution - 50% positive, 50% negative. This is crucial for fair evaluation. We use a fixed random seed of 42 so the split is reproducible. The training set gets 39,979 pairs, validation gets 8,567, and test gets 8,568."

**Emphasize:**
- **Fixed random seed**: Reproducibility
- **Stratified**: Same class balance across splits
- **No data leakage**: Patents in train don't appear in val/test

---

## FILE 3: `scripts/data/preprocessing/compute_features.py`

**What to say:**
> "Now we have our training pairs, but we can't just feed patent IDs to a neural network. We need to extract meaningful features. This file computes 10 engineered features for each pair that capture different aspects of similarity."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 50-180: `FeatureComputer` class initialization

```python
class FeatureComputer:
    def __init__(
        self,
        patents_path: str,
        embeddings_path: str,
        patent_ids_path: str,
        embedder: PatentEmbedder
    ):
        # Load patents
        self.patents = {}
        with open(patents_path, 'r') as f:
            for line in f:
                p = json.loads(line)
                self.patents[p['patent_id']] = p
        
        # Load embeddings (memory-mapped for efficiency)
        self.embeddings = np.load(embeddings_path, mmap_mode='r')
        
        # Load patent IDs
        with open(patent_ids_path, 'r') as f:
            self.patent_ids = json.load(f)
        
        # Create index mapping
        self.id_to_idx = {pid: i for i, pid in enumerate(self.patent_ids)}
        
        # Embedder for runtime embedding generation
        self.embedder = embedder
```

**Say:**
> "The FeatureComputer class loads everything it needs: the full patent database, the embeddings we generated, and the PatentSBERTa model for generating embeddings on-the-fly. We use memory-mapping for the embeddings - this means we don't load all 586 MB into RAM at once, only what we need."

#### Lines 200-400: `compute_features_for_pair()` - THE MOST IMPORTANT METHOD

**Go through each feature one by one:**

```python
def compute_features_for_pair(
    self,
    patent_a_id: str,
    patent_b_id: str
) -> np.ndarray:
    """Compute 10 features for a patent pair."""
    
    features = []
    
    # Load patent data
    patent_a = self.patents[patent_a_id]
    patent_b = self.patents[patent_b_id]
```

**Say:**
> "This method takes two patent IDs and computes 10 features. Let me walk through each feature and explain what it captures..."

### FEATURE 1: PatentSBERTa Cosine Similarity

```python
    # Feature 1: PatentSBERTa Cosine Similarity
    emb_a = self.embeddings[self.id_to_idx[patent_a_id]]
    emb_b = self.embeddings[self.id_to_idx[patent_b_id]]
    
    cosine_sim = np.dot(emb_a, emb_b) / (
        np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
    )
    features.append(cosine_sim)
```

**Say:**
> "Feature 1 is the cosine similarity between the PatentSBERTa embeddings. This captures semantic similarity - do these patents describe similar inventions? Cosine similarity measures the angle between two vectors in 768-dimensional space. If the vectors point in the same direction, they're similar (score near 1). If orthogonal, they're unrelated (score near 0)."

**Write on screen/whiteboard:**
```
cos(θ) = (A · B) / (||A|| × ||B||)

A = [0.2, 0.5, 0.1, ...]  (768 dims)
B = [0.3, 0.4, 0.2, ...]  (768 dims)

Similar patents: cos(θ) ≈ 0.85
Different patents: cos(θ) ≈ 0.15
```

### FEATURE 2: TF-IDF Cosine Similarity

```python
    # Feature 2: TF-IDF Cosine Similarity
    text_a = patent_a.get('abstract', '')
    text_b = patent_b.get('abstract', '')
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_sim = 0.0
    
    features.append(tfidf_sim)
```

**Say:**
> "Feature 2 is traditional TF-IDF similarity. Unlike PatentSBERTa which understands meaning, TF-IDF is pure word matching. It finds common important words. TF-IDF stands for Term Frequency-Inverse Document Frequency - it weights words by how often they appear in this document versus all documents. Common words like 'the' get low weight, technical terms like 'resonance' get high weight."

**Explain the difference:**
> "Why both semantic AND keyword similarity? Because they catch different things. Two patents about 'wireless charging' and 'inductive power transfer' will have high semantic similarity (same concept) but lower TF-IDF (different words). Having both gives the model more signal."

### FEATURE 3: Jaccard Similarity

```python
    # Feature 3: Jaccard Similarity (word overlap)
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    
    jaccard_sim = intersection / (union + 1e-8)
    features.append(jaccard_sim)
```

**Say:**
> "Feature 3 is Jaccard similarity - a simple set overlap metric. It answers: what fraction of unique words appear in both patents? Formula is intersection over union. If they share 50 words and have 200 unique words total, Jaccard is 0.25. This is cruder than TF-IDF but sometimes catches overlap that TF-IDF misses."

**Visual:**
```
Patent A words: {method, wireless, power, transfer, coil, ...}
Patent B words: {wireless, charging, coil, resonance, ...}

Intersection: {wireless, coil} = 2 words
Union: {method, wireless, power, transfer, coil, charging, resonance, ...} = 7 words

Jaccard = 2/7 = 0.286
```

### FEATURE 4-6: Metadata Features

```python
    # Feature 4: Claim Count Ratio
    claims_a = len(patent_a.get('claims', []))
    claims_b = len(patent_b.get('claims', []))
    claim_ratio = min(claims_a, claims_b) / (max(claims_a, claims_b) + 1e-8)
    features.append(claim_ratio)
    
    # Feature 5: Abstract Length Ratio
    len_a = len(text_a.split())
    len_b = len(text_b.split())
    length_ratio = min(len_a, len_b) / (max(len_a, len_b) + 1e-8)
    features.append(length_ratio)
    
    # Feature 6: Year Difference (normalized)
    year_a = patent_a.get('year', 2023)
    year_b = patent_b.get('year', 2023)
    year_diff = 1.0 / (1.0 + abs(year_a - year_b))
    features.append(year_diff)
```

**Say:**
> "Features 4, 5, and 6 are metadata features. Claim count ratio: similar patents often have similar numbers of claims. A patent with 20 claims is more likely related to another with 15 claims than one with 2 claims. Abstract length ratio: same idea. Year difference: patents from the same year or consecutive years are more likely to be in the same technological wave. We normalize year difference so patents from the same year get 1.0, one year apart gets 0.5, two years gets 0.33, etc."

### FEATURES 7-8: Patent Classification

```python
    # Feature 7: Assignee Match (binary)
    assignee_match = float(
        patent_a.get('assignee', '') == patent_b.get('assignee', '')
        and patent_a.get('assignee', '') != ''
    )
    features.append(assignee_match)
    
    # Feature 8: CPC Code Overlap (Jaccard)
    cpc_a = set(patent_a.get('cpc_codes', []))
    cpc_b = set(patent_b.get('cpc_codes', []))
    cpc_jaccard = len(cpc_a & cpc_b) / (len(cpc_a | cpc_b) + 1e-8)
    features.append(cpc_jaccard)
```

**Say:**
> "Feature 7 is assignee match - are they from the same company? Google patents might cite other Google patents. This is binary: 1 if same company, 0 if different. Feature 8 is CPC code overlap. CPC stands for Cooperative Patent Classification - it's like a library system for patents. For example, H04W (wireless communication) or G06F (computing). If two patents share CPC codes, they're in the same technical domain. We use Jaccard similarity on the CPC sets."

### FEATURE 9: Max Claim Similarity

```python
    # Feature 9: Max Claim Embedding Similarity
    claims_a = patent_a.get('claims', [])[:10]  # Limit to first 10
    claims_b = patent_b.get('claims', [])[:10]
    
    if claims_a and claims_b:
        # Embed all claims
        texts_a = [self._get_claim_text(c) for c in claims_a]
        texts_b = [self._get_claim_text(c) for c in claims_b]
        
        embs_a = self.embedder.encode(texts_a)
        embs_b = self.embedder.encode(texts_b)
        
        # Find maximum similarity across all pairs
        max_sim = 0.0
        for emb_a in embs_a:
            for emb_b in embs_b:
                sim = np.dot(emb_a, emb_b) / (
                    np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
                )
                max_sim = max(max_sim, sim)
        
        features.append(max_sim)
    else:
        features.append(0.0)
```

**Say:**
> "Feature 9 is the most computationally expensive but also very powerful. We embed individual claims from each patent using PatentSBERTa, then compute pairwise similarities between ALL claim pairs, and take the maximum. Why? Because two patents might have mostly different claims but ONE overlapping claim that's critical. The max captures that. We limit to first 10 claims for speed."

**Example:**
> "Patent A has 5 claims, Patent B has 8 claims. That's 5 × 8 = 40 similarity scores. We take the max. Maybe claims A.3 and B.6 are nearly identical (0.92 similarity) while others are different. That max of 0.92 is our feature."

### FEATURE 10: Title Similarity

```python
    # Feature 10: Title Similarity
    title_a = patent_a.get('title', '')
    title_b = patent_b.get('title', '')
    
    if title_a and title_b:
        emb_a = self.embedder.encode(title_a)
        emb_b = self.embedder.encode(title_b)
        title_sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        features.append(title_sim)
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)  # Shape: (10,)
```

**Say:**
> "Feature 10 is title similarity using embeddings. Titles are concise and informative - if two patents have very similar titles like 'Wireless Power Transfer System' and 'System for Wireless Energy Transfer', they're likely related. We return all 10 features as a numpy array."

### Summary of Feature Computation

**Say:**
> "So we extract 10 features that capture different aspects:
> - Features 1-3: Text similarity (semantic, keyword, word overlap)
> - Features 4-6: Metadata (claim count, length, year)
> - Features 7-8: Classification (company, CPC codes)  
> - Features 9-10: Fine-grained similarity (claims, title)
>
> These 10 numbers form the input to our neural network. For our 57,114 training pairs, we get a 57,114 × 10 feature matrix."

**Show the output files:**
- `data/features/train_features_v2.X.npy`: Shape (39,979, 10)
- `data/features/train_features_v2.y.npy`: Shape (39,979,) - labels (0 or 1)
- Same for val and test

---

# PART 2: MODEL TRAINING

---

## FILE 4: `src/app/pytorch_classifier.py`

**What to say:**
> "Now we have our features and labels. This file defines our custom PyTorch neural network that learns to predict whether two patents are similar. Let me walk through the architecture."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 26-66: `ResidualBlock` class

```python
class ResidualBlock(nn.Module):
    """
    Residual block with skip connections.
    
    Architecture:
        Input (in_features)
          ↓
        Linear → BatchNorm → ReLU → Dropout
          ↓
        Linear → BatchNorm → ReLU → Dropout
          ↓
        Add skip connection
          ↓
        Output (out_features)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float,
        bn_momentum: float = 0.1
    ):
        super().__init__()
        
        # Main path
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Skip connection
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features)
            self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        else:
            self.skip_connection = nn.Identity()
            self.skip_bn = nn.Identity()
```

**Say:**
> "The ResidualBlock is the building block of our network. It implements residual connections, which were popularized by ResNet. The key idea: we have a main path that transforms the input, and a skip connection that bypasses it. At the end, we ADD them together. This solves the vanishing gradient problem and allows us to train deeper networks."

**Draw on screen:**
```
     Input (x)
       ↓
    [Linear + BN + ReLU + Dropout]
       ↓
    [Linear + BN + ReLU + Dropout]
       ↓
     Output
       +  ← Skip connection (adds original input)
       ↓
    Final output = transformation(x) + x
```

**Continue:**
> "We also use Batch Normalization after each linear layer. This normalizes activations to have mean 0 and variance 1, which stabilizes training. Dropout randomly zeros 30% of activations during training to prevent overfitting."

#### Lines 69-139: `PatentNoveltyNet` class - THE NEURAL NETWORK

**Show the forward pass:**

```python
class PatentNoveltyNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        bn_momentum: float = 0.1
    ):
        super().__init__()
        
        # Input layer: 10 → 256
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0], momentum=bn_momentum)
        self.input_dropout = nn.Dropout(dropout)
        
        # Residual blocks: 256 → 128
        self.residual_blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.residual_blocks.append(
                ResidualBlock(in_dim, out_dim, dropout, bn_momentum)
            )
            in_dim = out_dim
        
        # Output layer: 128 → 1
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        # Input: (batch_size, 10)
        x = self.input_layer(x)      # → (batch_size, 256)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)              # → (batch_size, 128)
        
        # Output
        x = self.output_layer(x)     # → (batch_size, 1)
        x = torch.sigmoid(x)         # → Squash to [0, 1]
        
        return x
```

**Say:**
> "Here's our full architecture. We start with 10 input features. The input layer projects these to 256 dimensions - we're expanding the feature space. Then we have a residual block that transforms 256 → 128. Finally, an output layer reduces to 1 number, and sigmoid squashes it to a probability between 0 and 1."

**Draw the architecture:**
```
Input: 10 features
   ↓
Linear(10 → 256) + BN + ReLU + Dropout
   ↓
ResidualBlock(256 → 128)
   ↓
Linear(128 → 1)
   ↓
Sigmoid
   ↓
Output: probability [0, 1]
```

**Explain each layer:**
1. **Input layer (10 → 256)**: Expand feature space, capture interactions
2. **Residual block (256 → 128)**: Deep transformation with skip connection
3. **Output (128 → 1)**: Compress to single similarity score
4. **Sigmoid**: Convert to probability

**Why this architecture?**
> "We chose this architecture through hyperparameter tuning. We tried [512, 256], [128, 64], but [256, 128] gave the best validation ROC-AUC of 97.17%. The residual connection helps gradients flow backward during training."

#### Lines 142-502: `PyTorchPatentClassifier` - TRAINING LOGIC

**Navigate to the `fit()` method (lines 200-400):**

This is TOO LONG to show all at once. Break it into sections:

##### Section 1: Data Preparation

```python
def fit(self, X_train, y_train, X_val=None, y_val=None):
    """Train the model."""
    
    # Normalize features using StandardScaler
    X_train_scaled = self.scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = self.scaler.transform(X_val)
```

**Say:**
> "First, we normalize features using StandardScaler. This standardizes each feature to have mean 0 and standard deviation 1. Why? Because our features have different scales - cosine similarity is 0-1, year difference is 0.5-1.0, claim count might be 1-20. Normalizing puts them on equal footing so the neural network can learn effectively."

**Show the math:**
```
For each feature column:
x_normalized = (x - mean(x)) / std(x)

Example:
Claim count: [5, 10, 15, 20] 
Mean = 12.5, Std = 6.1
Normalized: [-1.2, -0.4, 0.4, 1.2]
```

##### Section 2: DataLoaders

```python
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.batch_size,  # 64
        shuffle=True
    )
```

**Say:**
> "We create PyTorch DataLoaders which handle batching and shuffling. Batch size is 64 - we process 64 patent pairs at once. Shuffling randomizes the order each epoch so the model doesn't memorize the sequence. This is standard PyTorch data pipeline."

##### Section 3: Optimizer and Scheduler

```python
    # Optimizer: AdamW with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        self.model.parameters(),
        lr=self.learning_rate,  # 0.002
        weight_decay=self.weight_decay  # 1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
```

**Say:**
> "We use AdamW optimizer - an improved version of Adam with better weight decay. Learning rate is 0.002, which we found through grid search. Weight decay is 1e-5 - this is L2 regularization that penalizes large weights to prevent overfitting.

> "The learning rate scheduler automatically reduces the learning rate by 50% if validation loss doesn't improve for 5 epochs. This helps the model converge better - start with bigger steps, then take smaller steps as we get close to the optimum."

##### Section 4: Training Loop

```python
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(self.max_epochs):  # Up to 100 epochs
        
        # TRAINING PHASE
        self.model.train()  # Enable dropout, batch norm in training mode
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)  # Move to GPU (MPS)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()  # Reset gradients
            predictions = self.model(batch_X)  # Get predictions
            loss = criterion(predictions, batch_y)  # Binary cross-entropy
            
            # Backward pass
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
```

**Say:**
> "This is the core training loop. For each epoch, we iterate through batches. The forward pass runs the batch through our network to get predictions. We compute binary cross-entropy loss - this measures how wrong our predictions are. Then backward() computes gradients using backpropagation, and step() updates the weights using those gradients. This is standard supervised learning."

**Explain Binary Cross-Entropy:**
> "Binary cross-entropy loss for a single example:
> Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
>
> If y=1 (similar patents) and ŷ=0.9, loss is small (good)
> If y=1 but ŷ=0.1, loss is large (bad)
>
> We average this across the batch to get the total loss."

##### Section 5: Validation and Early Stopping

```python
        # VALIDATION PHASE
        self.model.eval()  # Disable dropout, batch norm in eval mode
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():  # Don't compute gradients (saves memory)
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                val_losses.append(loss.item())
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # Compute metrics
        binary_preds = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        roc_auc = roc_auc_score(all_labels, all_preds)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(self.model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= self.patience:  # patience=15
                print(f"Early stopping at epoch {epoch}")
                break
```

**Say:**
> "After each training epoch, we validate. We set the model to eval mode which disables dropout, and use no_grad() to skip gradient computation. We compute validation loss and metrics: accuracy, F1 score, and ROC-AUC.

> "Then we do early stopping: if validation loss improves, we save the model and reset the patience counter. If it doesn't improve for 15 consecutive epochs, we stop training to prevent overfitting. This way we don't waste time and we use the model from the epoch with best validation performance."

**Final result:**
> "After training, our model achieved 91.73% accuracy and 97.20% ROC-AUC on the test set. That ROC-AUC means the model correctly ranks similar pairs higher than dissimilar pairs 97% of the time."

---

### TRANSITION TO INFERENCE

**Say:**
> "That's the training pipeline. Now let me show you how we use this trained model for inference - when a user enters a new patent and we need to assess its novelty."

---

# PART 3: INFERENCE PIPELINE

---

## FILE 5: `src/app/patent_analyzer.py`

**What to say:**
> "This is the orchestrator - the main class that coordinates everything during inference. When a user submits a patent, this file handles the end-to-end pipeline from embedding generation to final novelty report."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 75-160: `__init__()` and `load()` methods

**Show initialization:**

```python
class PatentAnalyzer:
    def __init__(
        self,
        patents_path='data/sampled/patents_sampled.jsonl',
        embeddings_path='data/embeddings/patent_embeddings.npy',
        patent_ids_path='data/embeddings/patent_ids.json',
        use_online_search=True,
        use_llm_keywords=True,
        serpapi_key=None
    ):
        self.patents_path = patents_path
        self.embeddings_path = embeddings_path
        self.patent_ids_path = patent_ids_path
        
        # Components (loaded lazily)
        self.embedder = None
        self.embeddings = None
        self.patent_ids = None
        self.patents = {}
        self.classifier = None
        self.feature_extractor = None
        self.explainer = None
        self.online_searcher = None
        self.keyword_extractor = None
```

**Say:**
> "The PatentAnalyzer initializes with paths to all our data. Notice all the components are None - we use lazy loading. We don't load the embeddings or models until someone calls load(). This saves memory."

**Show load():**

```python
def load(self, status_callback=None):
    """Load all components."""
    
    # Load embeddings (memory-mapped)
    if status_callback:
        status_callback("Loading embeddings...")
    self.embeddings = np.load(self.embeddings_path, mmap_mode='r')
    
    # Load patent IDs
    with open(self.patent_ids_path, 'r') as f:
        self.patent_ids = json.load(f)
    
    # Load PatentSBERTa
    if status_callback:
        status_callback("Loading PatentSBERTa model...")
    from src.embeddings.patent_sberta import PatentEmbedder
    self.embedder = PatentEmbedder()
    
    # Load PyTorch classifier
    if status_callback:
        status_callback("Loading PyTorch model...")
    self.classifier = PyTorchPatentClassifier()
    self.classifier.load('models/pytorch_nn')
    
    # Load Phi-3 explainer
    if status_callback:
        status_callback("Initializing Phi-3 explainer...")
    from src.app.phi3_explainer import Phi3OllamaExplainer
    self.explainer = Phi3OllamaExplainer()
    
    # Load online search if enabled
    if self.use_online_search:
        from data.api.online_search import GooglePatentsSearch
        self.online_searcher = GooglePatentsSearch(serpapi_key=self.serpapi_key)
```

**Say:**
> "The load() method initializes everything we need:
> 1. Memory-map the embeddings (586 MB but not fully loaded into RAM)
> 2. Load PatentSBERTa model (~400 MB)
> 3. Load our trained PyTorch classifier from disk
> 4. Connect to Phi-3 via Ollama
> 5. Initialize SerpAPI if we have a key
>
> The status_callback lets us show progress in the UI. Total load time is about 10-15 seconds."

#### Lines 400-600: `analyze()` method - THE MAIN INFERENCE FUNCTION

**This is THE MOST IMPORTANT method. Go through it step-by-step:**

```python
def analyze(
    self,
    query_text: str,
    top_k: int = 15,
    status_callback=None
) -> NoveltyReport:
    """
    Complete novelty assessment pipeline.
    
    Steps:
    1. Generate query embedding
    2. Local search (cosine similarity)
    3. Online search (LLM keywords + SerpAPI)
    4. Merge results
    5. Extract features
    6. Score with PyTorch model
    7. Generate LLM explanation
    8. Return NoveltyReport
    """
```

**Say:**
> "The analyze method is our main entry point. It takes the user's patent text and returns a complete novelty report. Let me walk through each step..."

### STEP 1: Generate Query Embedding

```python
    # Step 1: Generate embedding
    if status_callback:
        status_callback("🔄 Generating embeddings...")
    
    query_embedding = self.embedder.encode(query_text)
    # Returns: numpy array of shape (768,)
```

**Say:**
> "First, we embed the user's patent using PatentSBERTa. This converts their text into a 768-dimensional vector. This takes about 2-3 seconds because we're running a transformer model."

### STEP 2: Local Search

```python
    # Step 2: Local search
    if status_callback:
        status_callback("🔍 Searching local database (200K patents)...")
    
    local_results = self._find_similar(
        query_embedding,
        top_k=top_k,
        status_callback=status_callback
    )
```

**Say:**
> "Now we search our local database. Let me show you the _find_similar method..."

**Navigate to _find_similar (lines 633-662):**

```python
def _find_similar(
    self,
    query_embedding: np.ndarray,
    top_k: int = 10,
    status_callback=None
) -> List[Dict]:
    """Find similar patents using cosine similarity."""
    
    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    all_norms = self.embeddings / np.linalg.norm(
        self.embeddings, axis=1, keepdims=True
    )
    
    # Compute cosine similarities
    similarities = np.dot(all_norms, query_norm)
    # Shape: (200000,) - one score per patent
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Load patent metadata
    results = []
    for idx in top_indices:
        pid = self.patent_ids[idx]
        patent_data = self._load_patent(str(pid)) or {}
        results.append({
            'patent_id': pid,
            'similarity': float(similarities[idx]),
            'title': patent_data.get('title', 'N/A'),
            'abstract': patent_data.get('abstract', 'N/A'),
            'year': patent_data.get('year', 'N/A'),
            'source': 'local'
        })
    
    return results
```

**Say:**
> "Here's the local search algorithm. We normalize both the query embedding and all 200,000 database embeddings to unit length. Then we compute dot products - because vectors are normalized, dot product equals cosine similarity. We use np.dot which is highly optimized and runs in 1-2 seconds for 200K vectors. Then we use argsort to get the indices of the top-k highest scores and load those patents from disk."

**Visual:**
```
Query: [0.1, 0.2, ..., 0.3] (768 dims, normalized)
Database: 200,000 × 768 matrix (normalized)

Similarities = Database @ Query  
             = [0.82, 0.15, 0.91, ..., 0.23]  (200,000 scores)

Top-15: indices [2, 0, 15789, ...]
```

**Back to analyze() method:**

### STEP 3: Online Search

```python
    # Step 3: Online search
    online_results = []
    if self.use_online_search and self.online_searcher:
        if status_callback:
            status_callback("🌐 Extracting search keywords with LLM...")
        
        # Generate keywords using Phi-3
        keywords = []
        if self.use_llm_keywords and self.keyword_extractor:
            keywords = self.keyword_extractor.extract_keywords(query_text)
            if status_callback:
                status_callback(
                    f"Generated {len(keywords)} search terms: {keywords[:3]}..."
                )
        
        if status_callback:
            status_callback("🌐 Searching Google Patents online...")
        
        # Search Google Patents via SerpAPI
        online_results = self.online_searcher.search_multiple_terms(
            keywords,
            max_per_term=10
        )
```

**Say:**
> "If online search is enabled, we use Phi-3 to generate intelligent search terms. This LLM call takes 10-15 seconds. Then we query SerpAPI with each term, getting up to 10 results per term. With 5 terms, that's potentially 50 patents from Google Patents. This catches patents from before 2021 or international patents not in our local database."

### STEP 4: Merge Results

```python
    # Step 4: Merge results
    all_candidates = self._merge_results(local_results, online_results)
```

**Say:**
> "We merge local and online results, deduplicating by patent ID. Same patent might appear in both. We typically end up with 50-70 unique candidate patents."

### STEP 5: Feature Extraction and Scoring

```python
    # Step 5: Extract features and score
    if status_callback:
        status_callback(
            f"⚖️ Scoring {len(all_candidates)} candidates with PyTorch model..."
        )
    
    scored_patents = []
    for candidate in all_candidates:
        # Extract 10 features
        features = self.feature_extractor.extract_features(
            query_text,
            candidate
        )
        
        # Score with PyTorch model
        similarity_score = self.classifier.predict_proba(
            features.reshape(1, -1)
        )[0]
        
        candidate['model_similarity'] = float(similarity_score)
        candidate['model_novelty'] = 1 - similarity_score
        scored_patents.append(candidate)
```

**Say:**
> "For each candidate, we extract our 10 features - the same ones we computed during training. Then we run those features through our trained PyTorch model to get a similarity score between 0 and 1. This scoring takes 2-3 seconds for ~60 candidates because PyTorch batches them efficiently."

### STEP 6: Ranking and Novelty Score

```python
    # Step 6: Rank by score
    scored_patents.sort(key=lambda x: x['model_similarity'], reverse=True)
    top_scored = scored_patents[:20]  # Keep top 20
    
    # Compute overall novelty score
    mean_similarity = np.mean([p['model_similarity'] for p in top_scored])
    novelty_score = 1 - mean_similarity
```

**Say:**
> "We sort all candidates by similarity score and keep the top 20. Then we compute the overall novelty score as 1 minus the mean similarity of the top 20. If the top similar patents have average similarity of 0.75, novelty is 0.25 - low novelty, lots of prior art. If average similarity is 0.20, novelty is 0.80 - high novelty, likely patentable."

### STEP 7: LLM Explanation

```python
    # Step 7: Generate explanation
    if status_callback:
        status_callback("🤖 Generating explanation with Phi-3...")
    
    explanation = self.explainer.generate_explanation(
        query_patent={'text': query_text},
        similar_patents=top_scored[:10],  # Top 10 for explanation
        novelty_score=novelty_score
    )
```

**Say:**
> "Finally, we send the query patent, top 10 similar patents, and novelty score to Phi-3. The LLM generates a detailed explanation with executive summary, technical overlap analysis, and recommendations. This is the longest step - 30-45 seconds because we're generating 800-1200 tokens."

### STEP 8: Return Report

```python
    # Step 8: Create report
    report = NoveltyReport(
        query_text=query_text,
        novelty_score=novelty_score,
        rank_percentile=rank_percentile,
        similar_patents=top_scored,
        explanation=explanation,
        search_metadata={
            'local_results': len(local_results),
            'online_results': len(online_results),
            'total_candidates': len(all_candidates),
            'top_k_scored': len(top_scored)
        }
    )
    
    return report
```

**Say:**
> "We package everything into a NoveltyReport object and return it. Total time: 60-90 seconds. The Streamlit app then displays this to the user."

---

## FILE 6: `src/app/phi3_explainer.py`

**What to say:**
> "Let me show you how we generate the LLM explanation. This file interfaces with Phi-3 running locally via Ollama."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 75-200: `generate_explanation()` method

```python
def generate_explanation(
    self,
    query_patent: Dict,
    similar_patents: List[Dict],
    novelty_score: float
) -> str:
    """Generate novelty explanation using Phi-3."""
    
    # Build prompt
    prompt = self._build_prompt(query_patent, similar_patents, novelty_score)
    
    # Call Ollama API
    response = requests.post(
        self.api_endpoint,  # http://localhost:11434/api/generate
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 1200,  # Max tokens to generate
                "temperature": 0.4,   # Lower = more factual
                "top_p": 0.9
            }
        },
        timeout=180  # 3 minutes max
    )
    
    if response.status_code == 200:
        explanation = response.json()['response']
        return explanation
    else:
        return "Error generating explanation."
```

**Say:**
> "We call Ollama's API endpoint with our prompt. Temperature 0.4 makes the output more deterministic and factual - we don't want creative hallucinations. num_predict=1200 limits token generation. The request takes 30-45 seconds depending on hardware."

#### Lines 100-180: `_build_prompt()` - PROMPT ENGINEERING

**This is crucial - show the prompt structure:**

```python
def _build_prompt(self, query_patent, similar_patents, novelty_score):
    """Construct the LLM prompt."""
    
    # Determine novelty level
    if novelty_score > 0.7:
        assessment = "HIGH NOVELTY"
    elif novelty_score > 0.4:
        assessment = "MODERATE NOVELTY"
    else:
        assessment = "LOW NOVELTY"
    
    # Format similar patents
    prior_art_text = ""
    for i, patent in enumerate(similar_patents[:5], 1):
        similarity = patent.get('model_similarity', 0)
        prior_art_text += f"""
Patent {i}: {patent.get('patent_id', 'N/A')} (Similarity: {similarity*100:.1f}%)
Title: {patent.get('title', 'N/A')}
Abstract: {patent.get('abstract', 'N/A')[:300]}...
Year: {patent.get('year', 'N/A')}

"""
    
    # Construct full prompt
    prompt = f"""You are a patent examiner analyzing a patent application.

PATENT APPLICATION:
{query_patent.get('text', '')[:1500]}

TOP SIMILAR PRIOR ART:
{prior_art_text}

NOVELTY SCORE: {novelty_score:.3f} ({assessment})
Scale: 0.0 = Not Novel | 1.0 = Highly Novel

Provide assessment following this structure:

## EXECUTIVE SUMMARY
Brief determination (2-3 sentences).

## TECHNICAL OVERLAP ANALYSIS
For top 3 patents:
- Patent ID and similarity
- Overlapping concepts
- Key quotes from prior art
- Which claims affected

## NOVELTY CONCERNS
List specific novelty challenges.

## RECOMMENDATION
Overall patentability and suggestions.

Be specific. Cite patent IDs and scores.

Assessment:"""
    
    return prompt
```

**Say:**
> "This is our prompt. We're doing few-shot prompting - giving the LLM a role (patent examiner), context (the application and prior art), task (assess novelty), and format (structured sections). We include the novelty score so the LLM knows the quantitative assessment. We truncate the application to 1500 characters and each prior art abstract to 300 to fit in Phi-3's 8K context window. This prompt engineering took iteration to get right."

**Explain why local LLM:**
> "Why run Phi-3 locally instead of using GPT-4 API? Three reasons:
> 1. Privacy: Patent data never leaves the user's machine
> 2. Cost: No per-query fees (important for frequent use)
> 3. Control: We can tune parameters and prompts without API limits
>
> Trade-off is speed - GPT-4 would be 5-10 seconds, Phi-3 is 30-45 seconds. But the privacy benefit is worth it for patent applications."

---

## FILE 7: `data/api/online_search.py`

**What to say:**
> "This file has two classes: one that uses Phi-3 to generate search keywords, and one that queries Google Patents via SerpAPI. Let me show both."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 50-120: `LLMKeywordExtractor` class

```python
class LLMKeywordExtractor:
    def __init__(self, ollama_endpoint="http://localhost:11434"):
        self.endpoint = f"{ollama_endpoint}/api/generate"
        self.model = "phi3"
    
    def extract_keywords(self, patent_text: str, num_terms: int = 5):
        """Generate search terms using LLM."""
        
        prompt = f"""You are a patent search expert. Generate {num_terms} search terms for Google Patents.

Patent:
{patent_text[:1000]}

Generate {num_terms} diverse search terms using Google Patents syntax (AND, OR). Each should target a different aspect. Output only the terms, one per line.

Search terms:"""
        
        response = requests.post(
            self.endpoint,
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Some creativity
                    "num_predict": 200
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            text = response.json()['response']
            terms = [line.strip() for line in text.split('\n') if line.strip()]
            return terms[:num_terms]
        
        return []
```

**Say:**
> "We use Phi-3 again here, but with temperature 0.7 - slightly more creative because we want diverse search terms. We ask it to generate 5 terms using Google Patents boolean syntax. For example, given a wireless charging patent, it might generate:
> 1. 'wireless power transfer AND magnetic resonance'
> 2. 'inductive charging system OR resonant coupling'
> 3. 'near-field energy transmission'
>
> These are smarter than just extracting keywords from the text because the LLM understands synonyms and technical variations."

#### Lines 140-250: `GooglePatentsSearch` class

```python
class GooglePatentsSearch:
    def __init__(self, serpapi_key=None):
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY')
        self.use_serpapi = SERPAPI_AVAILABLE and bool(self.serpapi_key)
    
    def search(self, query: str, num_results: int = 10):
        """Search Google Patents via SerpAPI."""
        
        if not self.use_serpapi:
            return []
        
        params = {
            "engine": "google_patents",
            "q": query,
            "num": num_results,
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        patents = []
        for result in results.get('organic_results', []):
            patents.append({
                'patent_id': result.get('patent_id'),
                'title': result.get('title'),
                'abstract': result.get('snippet'),  # Just a snippet
                'year': self._parse_year(result.get('publication_date')),
                'url': result.get('link'),
                'source': 'online'
            })
        
        return patents
```

**Say:**
> "This queries SerpAPI's Google Patents endpoint. We pass the search query, number of results, and our API key. SerpAPI returns structured JSON with patent IDs, titles, snippets, and URLs. Note we only get snippets, not full abstracts, because Google Patents doesn't provide full text in search results. This is why local search with full embeddings is still valuable."

```python
    def search_multiple_terms(self, terms, max_per_term=10):
        """Search multiple terms and deduplicate."""
        
        seen_ids = set()
        all_patents = []
        
        for i, term in enumerate(terms, 1):
            logger.info(f"[{i}/{len(terms)}] Searching: '{term[:50]}...'")
            
            results = self.search(term, num_results=max_per_term)
            
            for patent in results:
                if patent['patent_id'] not in seen_ids:
                    all_patents.append(patent)
                    seen_ids.add(patent['patent_id'])
        
        return all_patents
```

**Say:**
> "For multiple terms, we search each one sequentially and deduplicate by patent ID. Same patent might appear for multiple queries. With 5 terms and 10 results each, we might get 50 results but only 45 unique after deduplication. This takes 15-25 seconds due to network latency - each API call is 3-5 seconds."

---

# PART 4: WEB APPLICATION

---

## FILE 8: `app.py`

**What to say:**
> "Finally, the web interface. This Streamlit app ties everything together into a user-friendly UI. Let me show you the key sections."

### OPEN THE FILE AND NAVIGATE TO:

#### Lines 20-35: Cached Model Loading

```python
@st.cache_resource
def load_analyzer(serpapi_key=None, use_online=True, use_keywords=True):
    """Load analyzer with caching."""
    
    analyzer = PatentAnalyzer(
        use_online_search=use_online,
        use_llm_keywords=use_keywords,
        serpapi_key=serpapi_key
    )
    analyzer.load()
    return analyzer
```

**Say:**
> "We use Streamlit's cache_resource decorator. This means we load the models once and keep them in memory across requests. Without this, every button click would reload PatentSBERTa and PyTorch - taking 10-15 seconds each time. With caching, subsequent analyses are instant because models stay loaded."

#### Lines 550-630: Sidebar Configuration

```python
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    serpapi_key = st.text_input(
        "SerpAPI Key (for online search)",
        value=st.session_state.get('serpapi_key', ''),
        type="password"
    )
    
    if serpapi_key:
        os.environ['SERPAPI_KEY'] = serpapi_key
        st.session_state['serpapi_key'] = serpapi_key
        st.success(f"API key configured ({len(serpapi_key)} chars)")
    
    # Search settings
    use_online = st.checkbox("Enable Online Search", value=True)
    use_keywords = st.checkbox("Use LLM Keywords", value=True)
    num_results = st.slider("Number of results", 5, 30, 15)
```

**Say:**
> "The sidebar lets users configure the system. They can paste their SerpAPI key directly in the browser - we store it in session state and environment variables. Checkboxes toggle online search and LLM keywords. The slider controls how many results to retrieve. This gives users flexibility without touching code."

#### Lines 600-750: Main Analysis Flow

```python
if st.button("🔍 Analyze Patent Novelty"):
    if not query_text.strip():
        st.error("Please enter patent text.")
        return
    
    # Load analyzer (cached)
    with st.spinner("Loading models..."):
        analyzer = load_analyzer(serpapi_key, use_online, use_keywords)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_status(msg):
        status_text.text(msg)
    
    # Run analysis
    try:
        result = analyzer.analyze(
            query_text,
            top_k=num_results,
            status_callback=update_status
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        display_results(result)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
```

**Say:**
> "When the user clicks 'Analyze Patent Novelty', we load the analyzer from cache, create a progress bar, and call analyze() with a status callback. The callback updates the status text in real-time - 'Generating embeddings...', 'Searching local database...', etc. Users see exactly what's happening during the 60-90 second wait. Then we display the results."

#### Lines 760-900: Results Display

```python
def display_results(result):
    # Novelty score card
    score = result.novelty_score
    
    if score > 0.7:
        color = "green"
        interpretation = "High Novelty - Likely Patentable"
    elif score > 0.4:
        color = "orange"
        interpretation = "Moderate Novelty"
    else:
        color = "red"
        interpretation = "Low Novelty - Prior Art Found"
    
    st.metric(
        label="Novelty Score",
        value=f"{score:.3f}",
        delta=interpretation
    )
    
    # Similar patents table
    st.subheader("📊 Similar Patents")
    df = pd.DataFrame([
        {
            'Patent ID': p['patent_id'],
            'Title': p['title'][:60],
            'Similarity': f"{p['model_similarity']:.3f}",
            'Year': p['year'],
            'Source': p['source']
        }
        for p in result.similar_patents[:15]
    ])
    st.dataframe(df)
    
    # AI Explanation
    st.subheader("🤖 AI Explanation")
    with st.expander("View Detailed Explanation"):
        st.markdown(result.explanation)
    
    # Download buttons
    st.download_button(
        "📄 Download Text Report",
        data=generate_text_report(result),
        file_name="novelty_report.txt"
    )
```

**Say:**
> "We display the novelty score with color coding - red for low, orange for moderate, green for high. Then a table of similar patents with their IDs, titles, similarity scores, years, and sources. The AI explanation is in an expandable section so it doesn't overwhelm the page. Finally, download buttons for text and JSON reports. It's a clean, informative interface."

---

# FINAL SUMMARY

**What to say to wrap up:**

> "Let me summarize the complete pipeline we've walked through:
>
> **Data Pipeline:**
> 1. `generate_embeddings.py`: Convert 200K patents to embeddings (11 hours, 586 MB output)
> 2. `extract_citation_pairs.py`: Create 57K training pairs from citation graph
> 3. `compute_features.py`: Extract 10 features per pair (57K × 10 matrix)
>
> **Model Training:**
> 4. `pytorch_classifier.py`: Train custom ResNet-style network (97.2% ROC-AUC)
>
> **Inference:**
> 5. `patent_analyzer.py`: Orchestrate entire pipeline (60-90s per query)
>    - Generate query embedding (2-3s)
>    - Local cosine search (1-2s)
>    - LLM keyword extraction (10-15s)
>    - Online SerpAPI search (15-25s)
>    - Feature extraction & scoring (2-3s)
>    - Phi-3 explanation (30-45s)
> 6. `phi3_explainer.py`: Generate interpretable explanations
> 7. `online_search.py`: Hybrid local + online retrieval
>
> **Application:**
> 8. `app.py`: Streamlit UI with real-time progress and results
>
> This is a complete machine learning system: from raw data to trained model to deployed web application. The hybrid RAG architecture gives us both speed (local search) and coverage (online search). The ML scoring is more accurate than pure similarity (97% ROC-AUC). And the local LLM provides privacy-preserving explanations.
>
> Questions?"

---

# APPENDIX: Quick Reference - Files in Order

1. **`scripts/data/preprocessing/generate_embeddings.py`** - Create 200K × 768 embeddings
2. **`scripts/training/extract_citation_pairs.py`** - Extract 57K training pairs  
3. **`scripts/data/preprocessing/compute_features.py`** - Compute 10 features per pair
4. **`src/app/pytorch_classifier.py`** - Define and train neural network
5. **`src/app/patent_analyzer.py`** - Main inference orchestrator
6. **`src/app/phi3_explainer.py`** - LLM explanation generator
7. **`data/api/online_search.py`** - LLM keywords + SerpAPI integration
8. **`app.py`** - Streamlit web application

**Total lines covered: ~2,500 lines of actual implementation code**

**Key metrics:**
- 200,000 patents in database
- 768-dimensional embeddings
- 57,114 training pairs
- 10 engineered features
- 97.20% test ROC-AUC
- 99.91% Recall@10 for retrieval
- 60-90 second inference time
- 15 rubric items, 99 points

---

END OF CODE WALKTHROUGH SCRIPT

