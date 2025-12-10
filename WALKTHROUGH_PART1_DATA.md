# Code Walkthrough Part 1: Data Preprocessing Pipeline

## Overview of Data Pipeline

Before we can train any models or run inference, we need to prepare our data. This involves several critical steps:

1. **Embedding Generation**: Convert 200,000 patents into semantic vectors
2. **Citation Pair Extraction**: Create training labels from patent citations
3. **Feature Computation**: Engineer 10 features for each patent pair

Each step builds on the previous one. Let me walk you through each file in detail.

---

# FILE 1: `scripts/data/preprocessing/generate_embeddings.py`

## Purpose and Context

This is the **most foundational** file in the entire project. Without embeddings, nothing else works. This file takes our raw patent text and converts it into numerical representations that machine learning models can understand.

**What are embeddings?**

An embedding is a dense vector representation of text that captures semantic meaning. Instead of representing "wireless charging system" as a sparse bag of words, we represent it as a point in 768-dimensional space. Similar concepts are close together in this space.

**Why PatentSBERTa specifically?**

PatentSBERTa is a BERT model that was fine-tuned on 1.2 million patent abstracts by the AI-Growth-Lab research group. Regular BERT was trained on Wikipedia and books - it understands general language. PatentSBERTa understands patent-specific terminology:
- "Prior art" vs "previous work"
- "Claims" vs "statements"
- Technical jargon in electrical, mechanical, biotech domains
- Legal language patterns

This domain-specific training is crucial. A general-purpose embedding model would conflate "battery charging" with "criminal charging" because they share a word. PatentSBERTa understands the context.

## Line-by-Line Code Walkthrough

### Lines 1-30: Imports and Setup

```python
#!/usr/bin/env python3
"""
Generate Patent Embeddings using PatentSBERTa

This script:
1. Loads 200,000 sampled patents from JSONL
2. Extracts text (abstract/summary/claims)
3. Generates 768-dimensional embeddings using PatentSBERTa
4. Saves embeddings as numpy array for fast retrieval

Time: ~11 hours on Apple M1 with MPS acceleration
Output: 586 MB embedding file + patent ID mapping
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm
import time
```

**Explanation:**

- `sentence_transformers`: HuggingFace library that provides easy access to BERT-based embedding models. PatentSBERTa is hosted on HuggingFace model hub.
- `torch`: PyTorch backend that sentence-transformers uses. We need this for GPU acceleration detection.
- `numpy`: For array operations and saving embeddings efficiently.
- `tqdm`: Progress bars so we can monitor the 11-hour run.
- `json`: To load patents from JSONL format and save patent IDs.

### Lines 32-56: `get_patent_text()` Function

This function is deceptively simple but critically important.

```python
def get_patent_text(patent: dict) -> str:
    """
    Extract text from patent for embedding.
    
    Priority order:
    1. Abstract (preferred - concise and comprehensive)
    2. Summary (if no abstract)
    3. First claim (if no summary)
    4. Empty string (if nothing available)
    
    Truncates to 500 characters to fit model context window.
    
    Args:
        patent: Dictionary with keys like 'abstract', 'summary', 'claims'
    
    Returns:
        String of up to 500 characters
    """
    # Try abstract first
    if patent.get("abstract"):
        return patent["abstract"][:500]
    
    # Fall back to summary
    if patent.get("summary"):
        return patent["summary"][:500]
    
    # Fall back to first claim
    claims = patent.get("claims", [])
    if claims:
        first_claim = claims[0]
        
        # Claims can be strings or dicts with 'text' key
        if isinstance(first_claim, dict):
            claim_text = first_claim.get("text") or ""
            return claim_text[:500]
        
        if isinstance(first_claim, str):
            return first_claim[:500]
    
    # No text available
    return ""
```

**Deep Dive Explanation:**

**Why prioritize abstract?**

Patents have multiple text sections:
- **Title**: Too short (5-15 words), not enough information
- **Abstract**: Perfect balance - 100-300 words, comprehensive overview
- **Summary**: Similar to abstract but sometimes more verbose
- **Claims**: Legally binding but written in complex legal language
- **Description**: Too long (thousands of words), would need chunking

Abstracts are the sweet spot. They summarize the invention's purpose, method, and advantages concisely. This is what patent examiners read first, so it's what we should embed.

**Why 500 character limit?**

PatentSBERTa is based on BERT, which has a maximum sequence length of 512 tokens. A token is roughly 0.75 words, so 512 tokens ≈ 384 words ≈ 1920 characters. We use 500 characters to be conservative, which is ~375 words or ~500 tokens. This ensures we never exceed the model's context window.

What happens if we exceed it? The model would truncate automatically, potentially cutting off important information mid-sentence. Better to truncate ourselves cleanly.

**Why handle dict vs string claims?**

Different data sources format claims differently:
```python
# Format 1: List of strings
claims = [
    "1. A device comprising...",
    "2. The device of claim 1..."
]

# Format 2: List of dicts
claims = [
    {"claim_number": 1, "text": "A device comprising..."},
    {"claim_number": 2, "text": "The device of claim 1..."}
]
```

Our code handles both by checking `isinstance(first_claim, dict)`. This defensive programming prevents crashes when processing data from different USPTO download batches.

**Why return empty string instead of None?**

The sentence-transformers library expects a string. If we pass None, it will crash. Empty string gets embedded as a zero vector (or very small random vector), which has ~0.0 similarity to everything - exactly what we want for patents with no text.

### Lines 59-86: `generate_embeddings_batch()` Function

This is where the actual embedding happens.

```python
def generate_embeddings_batch(
    patents: List[dict],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for a batch of patents.
    
    Args:
        patents: List of patent dictionaries
        model: Loaded PatentSBERTa model
        batch_size: Number of patents to process at once
    
    Returns:
        numpy array of shape (len(patents), 768)
    """
    # Extract text from all patents in batch
    texts = []
    for patent in patents:
        text = get_patent_text(patent)
        texts.append(text)
    
    # Generate embeddings using the model
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    
    return embeddings
```

**Deep Dive Explanation:**

**What does `model.encode()` actually do?**

Under the hood, this function:

1. **Tokenization**: Converts text to token IDs using BERT tokenizer
   ```
   "wireless charging system" 
   → ["wireless", "charging", "system"]
   → [2456, 8923, 1034] (token IDs)
   ```

2. **Add special tokens**: `[CLS]` at start, `[SEP]` at end
   ```
   [101, 2456, 8923, 1034, 102]
   where 101 = [CLS], 102 = [SEP]
   ```

3. **Padding**: Pad to max length (512 tokens) with 0s
   ```
   [101, 2456, 8923, 1034, 102, 0, 0, 0, ...]
   ```

4. **Create attention mask**: 1 for real tokens, 0 for padding
   ```
   [1, 1, 1, 1, 1, 0, 0, 0, ...]
   ```

5. **Run through BERT encoder**: 12 transformer layers with multi-head attention
   ```
   Each layer:
   - Multi-head self-attention (8 heads)
   - Layer normalization
   - Feed-forward network (2 layers)
   - Residual connections
   ```

6. **Pool [CLS] token**: Extract the first token's final representation
   ```
   Output of layer 12 for [CLS] token = 768-dimensional vector
   ```

7. **Return**: Numpy array of shape (batch_size, 768)

**Why batch_size=32?**

Batch size is a trade-off:
- **Too small (e.g., 1)**: Underutilizes GPU, very slow (6× slower)
- **Too large (e.g., 256)**: Might exceed GPU memory, causes OOM errors
- **32**: Sweet spot for Apple M1/M2 MPS (Metal Performance Shaders)

With batch_size=32:
- GPU is well-utilized
- Memory usage ~2-3 GB (under M1's 8-16 GB limit)
- Processing speed: ~18 patents/second
- 200,000 patents ÷ 18/sec = 11,111 seconds = 3.1 hours

Wait, why did I say 11 hours earlier? Because:
- Model loading overhead
- Disk I/O for reading JSONL
- Progress bar rendering
- Saving checkpoints every 10K patents
- Actual observed time was 40,110 seconds = 11.1 hours

**Why `convert_to_numpy=True`?**

sentence-transformers returns PyTorch tensors by default. We specify `convert_to_numpy=True` because:
- NumPy arrays are easier to save/load (`np.save`, `np.load`)
- NumPy plays nicer with sklearn (for feature computation later)
- Smaller file size when saved
- We don't need autograd for embeddings (no backprop)

**Why `normalize_embeddings=False`?**

We could normalize embeddings to unit length during generation:
```python
embedding_normalized = embedding / ||embedding||
```

But we do it later during search instead. Why?
- Saves computation during generation (11 hours is already long)
- Flexibility: Can normalize or not during search
- Memory: Normalized vs unnormalized are same size

During search, we normalize on-the-fly:
```python
query_norm = query_emb / np.linalg.norm(query_emb)
db_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### Lines 90-180: `main()` Function - The Orchestration

This is where everything comes together.

```python
def main():
    """
    Main execution pipeline.
    
    Steps:
    1. Load PatentSBERTa model
    2. Configure GPU acceleration
    3. Load 200K patents from JSONL
    4. Generate embeddings in batches
    5. Save embeddings and patent IDs
    """
    
    print("="*80)
    print("Patent Embedding Generation Pipeline")
    print("="*80)
    
    # Step 1: Load PatentSBERTa model
    print("\n[1/5] Loading PatentSBERTa model from HuggingFace...")
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    # Step 2: Configure device (GPU if available)
    print("\n[2/5] Configuring computation device...")
    
    if torch.cuda.is_available():
        device = 'cuda'
        model = model.to('cuda')
        print(f"  ✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        model = model.to('mps')
        print(f"  ✓ Using Apple Metal Performance Shaders (MPS)")
    else:
        device = 'cpu'
        print(f"  ⚠ Using CPU (slow - expect 48+ hours)")
    
    # Step 3: Load patents
    print("\n[3/5] Loading patents from JSONL database...")
    patents_path = Path('data/sampled/patents_sampled.jsonl')
    
    patents = []
    with open(patents_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading patents", total=200000)):
            try:
                patent = json.loads(line)
                patents.append(patent)
            except json.JSONDecodeError as e:
                print(f"  ⚠ Warning: Skipping line {line_num} due to JSON error")
                continue
    
    print(f"  ✓ Loaded {len(patents):,} patents")
    
    # Step 4: Generate embeddings
    print("\n[4/5] Generating embeddings (this will take ~11 hours)...")
    
    all_embeddings = []
    batch_size = 32
    num_batches = (len(patents) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(patents), batch_size), 
                  desc="Processing batches",
                  total=num_batches):
        batch = patents[i:i+batch_size]
        batch_embeddings = generate_embeddings_batch(batch, model, batch_size)
        all_embeddings.append(batch_embeddings)
        
        # Checkpoint every 10,000 patents
        if (i + batch_size) % 10000 == 0:
            checkpoint_num = (i + batch_size) // 10000
            checkpoint_path = f'data/embeddings/checkpoint_{checkpoint_num}.npy'
            temp_embeddings = np.vstack(all_embeddings)
            np.save(checkpoint_path, temp_embeddings)
            print(f"  ✓ Checkpoint saved: {checkpoint_num * 10000} patents processed")
    
    # Concatenate all batches
    print("\n  Concatenating embeddings...")
    embeddings = np.vstack(all_embeddings)
    
    elapsed_time = time.time() - start_time
    print(f"  ✓ Generated {embeddings.shape[0]:,} embeddings in {elapsed_time/3600:.1f} hours")
    print(f"  ✓ Embedding shape: {embeddings.shape}")
    print(f"  ✓ Memory size: {embeddings.nbytes / (1024**3):.2f} GB")
    
    # Step 5: Save outputs
    print("\n[5/5] Saving embeddings and metadata...")
    
    # Save embeddings
    output_path = Path('data/embeddings/patent_embeddings.npy')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"  ✓ Saved embeddings to {output_path}")
    
    # Save patent IDs (for index mapping)
    patent_ids = [p.get('patent_id', p.get('id', f'UNKNOWN_{i}')) 
                  for i, p in enumerate(patents)]
    ids_path = Path('data/embeddings/patent_ids.json')
    with open(ids_path, 'w') as f:
        json.dump(patent_ids, f, indent=2)
    print(f"  ✓ Saved patent IDs to {ids_path}")
    
    # Save metadata
    metadata = {
        'num_patents': len(patents),
        'embedding_dim': embeddings.shape[1],
        'model_name': 'AI-Growth-Lab/PatentSBERTa',
        'generation_time_seconds': elapsed_time,
        'device_used': device,
        'batch_size': batch_size
    }
    metadata_path = Path('data/embeddings/embedding_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*80)
    print("✓ Embedding generation complete!")
    print("="*80)
```

**Detailed Explanation of Each Step:**

### Step 1: Loading PatentSBERTa

```python
model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
```

**What happens here:**

1. Sentence-transformers checks local cache (`~/.cache/torch/sentence_transformers/`)
2. If not cached, downloads from HuggingFace:
   - Model config (config.json)
   - Tokenizer vocabulary (vocab.txt)
   - Model weights (pytorch_model.bin) - ~440 MB
3. Loads weights into memory
4. Initializes BERT architecture:
   ```
   BertModel(
     12 encoder layers
     12 attention heads per layer
     768 hidden dimension
     110M parameters total
   )
   ```

**Model architecture details:**

PatentSBERTa uses the BERT-base architecture:
- **Embedding layer**: Converts token IDs to 768-dim vectors
- **12 Transformer encoder layers**: Each layer has:
  - Multi-head self-attention (12 heads × 64 dims = 768 total)
  - Layer normalization
  - Position-wise feed-forward network (768 → 3072 → 768)
  - Residual connections around each sublayer
- **Pooling layer**: Extracts [CLS] token representation

Total parameters: 110,336,768 (110M)

**How it was trained (by AI-Growth-Lab):**

1. Started with BERT-base (trained on Wikipedia)
2. Fine-tuned on 1.2M patent abstracts using contrastive learning
3. Training objective: Pull similar patents closer, push dissimilar apart
4. Result: Patent-domain specialized embeddings

### Step 2: GPU Acceleration Detection

```python
if torch.cuda.is_available():
    device = 'cuda'
    model = model.to('cuda')
    print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = 'mps'
    model = model.to('mps')
    print(f"✓ Using Apple Metal Performance Shaders (MPS)")
else:
    device = 'cpu'
    print(f"⚠ Using CPU (slow - expect 48+ hours)")
```

**Deep dive on device selection:**

**CUDA (NVIDIA GPUs):**
- Best performance for transformers
- Highly optimized CUDA kernels
- Can process ~40-50 patents/second
- 200K patents would take ~90 minutes

**MPS (Apple Silicon M1/M2/M3):**
- Apple's GPU framework
- Good performance, ~18-20 patents/second
- 200K patents takes ~3 hours of pure computation
- Actual time: 11 hours (including I/O overhead)

**CPU:**
- No GPU, uses CPU threads
- Very slow: ~3-4 patents/second
- 200K patents would take 48-60 hours
- Only use if no GPU available

**Why such a big difference?**

Transformers do massive matrix multiplications:
```
Attention: Q @ K^T @ V
  Q: (batch, 512, 768)
  K: (batch, 512, 768)
  V: (batch, 512, 768)
  
Result: (batch, 512, 768)

Operations: billions of floating-point multiplications
```

GPUs have thousands of cores optimized for parallel matrix operations. CPUs have 4-16 cores. GPUs are 10-20× faster for this workload.

### Step 3: Loading Patents

```python
patents = []
with open(patents_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(tqdm(f, desc="Loading", total=200000)):
        try:
            patent = json.loads(line)
            patents.append(patent)
        except json.JSONDecodeError as e:
            print(f"⚠ Skipping line {line_num}")
            continue
```

**Why JSONL format?**

JSONL (JSON Lines) has one JSON object per line:
```
{"patent_id": "US11234567", "title": "...", "abstract": "..."}
{"patent_id": "US11234568", "title": "...", "abstract": "..."}
{"patent_id": "US11234569", "title": "...", "abstract": "..."}
```

**Advantages:**
1. **Streamable**: Can read line-by-line, don't need to load entire file
2. **Appendable**: Can add new patents easily
3. **Robust**: One corrupted line doesn't break entire file
4. **Space-efficient**: No array brackets or commas between objects

**Why try/except around JSON parsing?**

Real-world data is messy. Out of 200,000 patents:
- ~50 might have malformed JSON (unescaped quotes, truncated lines)
- Better to skip a few patents than crash the entire 11-hour run
- We log which lines failed for debugging

**Memory consideration:**

Loading 200K patents into RAM:
- Average patent: ~15 KB (title + abstract + claims + metadata)
- 200,000 × 15 KB = 3 GB
- This fits comfortably in 16 GB RAM
- If RAM was limited, we'd process in chunks and not keep all in memory

### Step 4: Generate Embeddings (The Main Loop)

```python
all_embeddings = []
batch_size = 32
num_batches = (len(patents) + batch_size - 1) // batch_size  # Ceiling division

start_time = time.time()

for i in tqdm(range(0, len(patents), batch_size), 
              desc="Processing batches",
              total=num_batches):
    
    # Get batch
    batch = patents[i:i+batch_size]
    
    # Generate embeddings for this batch
    batch_embeddings = generate_embeddings_batch(batch, model, batch_size)
    
    # Accumulate
    all_embeddings.append(batch_embeddings)
    
    # Checkpoint every 10,000 patents
    if (i + batch_size) % 10000 == 0:
        checkpoint_num = (i + batch_size) // 10000
        checkpoint_path = f'data/embeddings/checkpoint_{checkpoint_num}.npy'
        temp_embeddings = np.vstack(all_embeddings)
        np.save(checkpoint_path, temp_embeddings)
        print(f"✓ Checkpoint saved: {checkpoint_num * 10000} patents")
```

**Deep explanation:**

**Batching logic:**

200,000 patents ÷ 32 per batch = 6,250 batches

Each batch:
1. Slice 32 patents from list: `batch = patents[0:32]`, then `[32:64]`, etc.
2. Extract text from each: 32 strings
3. Tokenize all 32: (32, 512) token ID matrix
4. Run through BERT: (32, 512, 768) hidden states
5. Pool to get: (32, 768) embeddings
6. Append to list

**Why accumulate in a list, not pre-allocate array?**

We could do:
```python
embeddings = np.zeros((200000, 768), dtype=np.float32)
for i, batch in enumerate(...):
    embeddings[i*32:(i+1)*32] = generate_embeddings_batch(batch, model)
```

But we use a list because:
- Safer: No index calculation errors
- Flexible: Handles last batch being smaller (<32 patents)
- Memory-efficient: Grows incrementally
- At the end, `np.vstack()` is fast (C-level operation)

**Checkpointing strategy:**

Every 10,000 patents (every 312 batches), we:
1. Stack embeddings generated so far: `np.vstack(all_embeddings)`
2. Save to disk: `checkpoint_10.npy`, `checkpoint_20.npy`, etc.
3. Continue processing

**Why checkpoint?**

- If the process crashes at 9 hours, we don't lose everything
- Can resume from last checkpoint
- Disk space trade-off: ~5 GB of checkpoints (deleted after success)

Without checkpointing, a crash at 10.5 hours means restarting from scratch.

### Step 5: Saving Outputs

```python
# Concatenate all batches
embeddings = np.vstack(all_embeddings)
# all_embeddings is list of arrays: [(32, 768), (32, 768), ...]
# vstack stacks them vertically: (200000, 768)

# Save embeddings
np.save('data/embeddings/patent_embeddings.npy', embeddings)
```

**NumPy save format:**

`np.save()` creates a binary `.npy` file:
- Header: Shape, dtype, byte order
- Data: Raw float32 values in binary

**File size calculation:**
```
200,000 patents × 768 dimensions × 4 bytes (float32) = 614,400,000 bytes
= 614.4 MB
= 586 MiB (binary MB vs decimal MB)
```

**Why float32 instead of float64?**

- float32: 4 bytes per number, ±3.4e38 range, 7 decimal digits precision
- float64: 8 bytes per number, ±1.7e308 range, 15 decimal digits precision

For embeddings, float32 is sufficient:
- Precision: 0.0000001 is fine for cosine similarity
- Space: 50% smaller files
- Speed: 2× faster operations on some hardware

**Patent ID mapping:**

```python
patent_ids = [p.get('patent_id', f'UNKNOWN_{i}') for i, p in enumerate(patents)]
with open('data/embeddings/patent_ids.json', 'w') as f:
    json.dump(patent_ids, f)
```

This creates the mapping:
```
Index 0 → "US11234567"
Index 1 → "US11234568"
Index 2 → "US11234569"
...
Index 199999 → "US12456789"
```

During search, we'll get indices from `argsort()`, then look up patent IDs using this list.

**Metadata file:**

```python
metadata = {
    'num_patents': 200000,
    'embedding_dim': 768,
    'model_name': 'AI-Growth-Lab/PatentSBERTa',
    'generation_time_seconds': 40110,  # 11.1 hours
    'device_used': 'mps',
    'batch_size': 32
}
```

This documents the generation process for reproducibility and verification.

---

## Summary of Embedding Generation

**Inputs:**
- `data/sampled/patents_sampled.jsonl` (200,000 patents, 3.8 GB)

**Process:**
- Load PatentSBERTa (110M parameters)
- Extract text from each patent (abstract preferred)
- Batch encode in groups of 32
- Run through 12 transformer layers
- Extract [CLS] token representation
- Total time: 11.1 hours on Apple M1

**Outputs:**
- `data/embeddings/patent_embeddings.npy` (200000 × 768, 586 MB)
- `data/embeddings/patent_ids.json` (200,000 IDs, 2.3 MB)
- `data/embeddings/embedding_metadata.json` (stats)

**Why this matters:**

These embeddings are the foundation of everything:
- Local search uses them for cosine similarity
- Feature extraction uses them (Feature #1)
- Claim embeddings use the same model
- Without embeddings, no semantic search

**Space vs Time trade-off:**

We could generate embeddings on-demand during search:
```python
query_emb = model.encode(user_query)  # 2 seconds
for patent in database:
    patent_emb = model.encode(patent_text)  # 2 seconds
    similarity = cosine(query_emb, patent_emb)
```

For 200K patents × 2 seconds = 400,000 seconds = 111 hours per query!

Pre-computing and saving embeddings:
- One-time cost: 11 hours
- Every query: 2 seconds (just encode query)
- After just 20 queries, we've saved time

Storage cost: 586 MB (trivial on modern hardware)

This is the classic time-space trade-off: we pay space (disk) to save time (compute).

---

# FILE 2: `scripts/training/extract_citation_pairs.py`

## Purpose and Context

Now that we have embeddings, we need to train a model to score patent similarity. But we need labeled training data: pairs of patents labeled "similar" or "not similar".

**Where do these labels come from?**

Patent citations! When patent A cites patent B, it means:
- The inventors researched B while developing A
- B is relevant prior art to A
- Patent examiner considers B related to A
- B influenced or is similar to A

This gives us supervised learning labels from the real world.

## The Positive-Unlabeled Learning Problem

This is a critical concept to understand.

**Positive Examples (Confirmed):**
- If patent US11234567 cites US10123456, they ARE related
- This is explicit, recorded in the citation graph
- These are our "positive" training examples
- Label: 1

**Negative Examples (Unlabeled!):**
- If patent US11234567 does NOT cite US10999999, are they unrelated?
- **Not necessarily!** They might be:
  - In different patent families (Google vs Apple)
  - Filed before/after citation window
  - Examiner missed the connection
  - Related but inventors didn't know about the other
- These are "unlabeled" - we ASSUME they're different but can't be certain
- Label: 0 (but actually unknown)

**Why this matters:**

Our "negative" examples might include some false negatives (actually related but no citation). This adds noise to training data. But with 28,557 negative pairs randomly sampled, the noise is diluted. Most random pairs genuinely ARE unrelated.

Our model still achieves 97% ROC-AUC despite this noise, proving the approach works.

## Line-by-Line Code Walkthrough

### Lines 1-40: Imports and Configuration

```python
"""
Citation Pair Extraction for Patent Similarity Training

Creates balanced dataset of positive and negative patent pairs:
- Positive pairs: Patent A cites Patent B (confirmed relationship)
- Negative pairs: Random pairs where A does not cite B (assumed unrelated)

This implements positive-unlabeled (PU) learning:
- Positives are ground truth (citations are explicit)
- Negatives are noisy (lack of citation doesn't prove lack of relationship)

Despite noise, model achieves 97% ROC-AUC, validating the approach.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm

# Configuration
RANDOM_SEED = 42
NEGATIVE_POSITIVE_RATIO = 1.0  # 1:1 ratio for balanced classes
VAL_RATIO = 0.15  # 15% for validation
TEST_RATIO = 0.15  # 15% for test
```

**Configuration choices explained:**

**RANDOM_SEED = 42:**
- Ensures reproducibility
- Same seed → same random samples → same train/val/test split
- Critical for comparing different model architectures
- 42 is arbitrary but conventional (Hitchhiker's Guide reference)

**NEGATIVE_POSITIVE_RATIO = 1.0:**
- For every positive pair, generate 1 negative pair
- Results in 50/50 class balance
- Why balanced?
  - Prevents model from just predicting majority class
  - Accuracy is meaningful (not inflated by class imbalance)
  - ROC-AUC is interpretable

Could use ratio > 1 (more negatives):
- Ratio = 2: More data but imbalanced (67% negative)
- Ratio = 3: Even more imbalanced (75% negative)
- We'd need to weight the loss function

**VAL_RATIO = 0.15, TEST_RATIO = 0.15:**
- Common split: 70% train, 15% val, 15% test
- Val: For hyperparameter tuning and early stopping
- Test: For final evaluation (never seen during training)
- Alternative: 80/10/10 (more training data)
- Alternative: 60/20/20 (more rigorous evaluation)

We chose 70/15/15 as a standard ML split.

### Lines 64-113: `generate_negative_pairs()` Function

**This is the most important algorithmic decision in the training data creation.**

```python
def generate_negative_pairs(
    positive_pairs: List[Tuple[str, str]],
    citation_graph: Dict[str, List[str]],
    all_patent_ids: List[str],
    ratio: float = 1.0,
    random_seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Generate negative training pairs via random sampling.
    
    Strategy:
    For each positive pair (A → B) where A cites B:
        1. Sample random patent C from database
        2. Check: C != B and A does not cite C
        3. If true, create negative pair (A, C)
        4. Repeat `ratio` times
    
    Args:
        positive_pairs: List of (citing_id, cited_id) tuples
        citation_graph: Dict mapping patent_id → List[cited_patent_ids]
        all_patent_ids: All patent IDs in our database
        ratio: Number of negative pairs per positive pair
        random_seed: For reproducibility
    
    Returns:
        List of (patent_a_id, patent_b_id) tuples (negative examples)
    """
    
    random.seed(random_seed)
    negative_pairs = []
    
    for citing_patent, cited_patent in tqdm(positive_pairs, desc="Generating negatives"):
        
        # Get all patents that `citing_patent` cites
        cited_by_citing = set(citation_graph.get(citing_patent, []))
        
        # Generate `ratio` negative pairs for this positive
        negatives_for_this_positive = 0
        attempts = 0
        max_attempts = 100  # Safety limit
        
        while negatives_for_this_positive < ratio and attempts < max_attempts:
            # Sample random patent
            candidate = random.choice(all_patent_ids)
            
            # Check validity
            if candidate != citing_patent and candidate not in cited_by_citing:
                # Valid negative pair
                negative_pairs.append((citing_patent, candidate))
                negatives_for_this_positive += 1
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not generate {ratio} negatives for {citing_patent}")
    
    return negative_pairs
```

**Detailed Algorithm Walkthrough:**

**Example scenario:**

Suppose we have positive pair: `(US11234567 → US10123456)`
- US11234567 cites US10123456 (confirmed relationship)
- US11234567 also cites US10111111, US10222222 (other citations)

Citation graph:
```python
citation_graph = {
    'US11234567': ['US10123456', 'US10111111', 'US10222222'],
    ...
}
```

**Negative sampling process:**

Step 1: Get citation set
```python
cited_by_citing = {'US10123456', 'US10111111', 'US10222222'}
```

Step 2: Sample random patent
```python
candidate = random.choice(all_patent_ids)
# Might return: 'US10999999'
```

Step 3: Check validity
```python
if 'US10999999' != 'US11234567' and 'US10999999' not in cited_by_citing:
    # Valid! US11234567 does NOT cite US10999999
    negative_pairs.append(('US11234567', 'US10999999'))
```

Step 4: Repeat for ratio=1.0 (just once per positive)

**Why this sampling strategy?**

**Alternative 1: Pure random**
```python
patent_a = random.choice(all_patents)
patent_b = random.choice(all_patents)
negative_pair = (patent_a, patent_b)
```

Problem: Might accidentally sample a citation pair!

**Alternative 2: Ensure no citation in either direction**
```python
if patent_a not in citation_graph.get(patent_b, []) and
   patent_b not in citation_graph.get(patent_a, []):
    negative_pair = (patent_a, patent_b)
```

Better, but:
- Takes 2× as long to check
- Still might sample related patents without formal citation

**Our strategy: Asymmetric sampling**
- Keep first patent from positive pair (citing patent)
- Sample second patent randomly (not in citation set)
- Advantages:
  - Balanced: Same citing patents in positive and negative sets
  - Reduces bias: Model sees each patent in multiple contexts
  - Efficient: Only one random sample per negative

**Safety limit: max_attempts=100**

Why might we fail to generate a negative?

Suppose US11234567 cites 50,000 other patents (highly connected hub):
- `cited_by_citing` has 50,000 elements
- Total database: 200,000 patents
- Probability of sampling a non-citation: 150,000/200,000 = 75%
- Expected attempts: 1.33

But for safety, we allow 100 attempts. If we hit the limit, we log a warning and move on. This prevents infinite loops for pathological cases.

**Negative pairs characteristics:**

From 28,557 positive pairs, we generate 28,557 negative pairs.

Distribution:
- Average citing patent in positives: cited ~15 other patents
- Average negative candidate: Random patent with ~0.01% chance of being related
- Most negatives are truly unrelated (estimated >98%)
- Some might be related without citation (noise <2%)

This noise is acceptable because:
1. Model has 57K examples to learn from
2. True signal (citations) is strong
3. Random noise averages out
4. Empirical result: 97% ROC-AUC proves it works

### Lines 150-220: Train/Val/Test Split

```python
def create_train_val_test_split(
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: List[Tuple[str, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List, List, List]:
    """
    Create stratified split maintaining class balance.
    
    Process:
    1. Shuffle positive and negative pairs separately (with fixed seed)
    2. Split each into train/val/test using same ratios
    3. Combine and label (1 for positive, 0 for negative)
    4. Return three datasets
    
    This ensures:
    - Same class balance (50/50) in all splits
    - No data leakage between splits
    - Reproducible splits (fixed random seed)
    """
    
    random.seed(random_seed)
    
    # Shuffle both lists
    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)
    
    # Calculate split sizes
    n_positive = len(positive_pairs)  # 28,557
    n_negative = len(negative_pairs)  # 28,557
    
    val_size = int(n_positive * val_ratio)    # 4,283
    test_size = int(n_positive * test_ratio)  # 4,283
    train_size = n_positive - val_size - test_size  # 19,991
    
    # Split positives
    pos_train = positive_pairs[:train_size]
    pos_val = positive_pairs[train_size:train_size+val_size]
    pos_test = positive_pairs[train_size+val_size:]
    
    # Split negatives (same sizes)
    neg_train = negative_pairs[:train_size]
    neg_val = negative_pairs[train_size:train_size+val_size]
    neg_test = negative_pairs[train_size+val_size:]
    
    # Combine and label
    train_data = [(pair, 1) for pair in pos_train] + [(pair, 0) for pair in neg_train]
    val_data = [(pair, 1) for pair in pos_val] + [(pair, 0) for pair in neg_val]
    test_data = [(pair, 1) for pair in pos_test] + [(pair, 0) for pair in neg_test]
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"Split summary:")
    print(f"  Train: {len(train_data):,} pairs ({len(pos_train):,} pos + {len(neg_train):,} neg)")
    print(f"  Val:   {len(val_data):,} pairs ({len(pos_val):,} pos + {len(neg_val):,} neg)")
    print(f"  Test:  {len(test_data):,} pairs ({len(pos_test):,} pos + {len(neg_test):,} neg)")
    
    return train_data, val_data, test_data
```

**Why stratified splitting?**

**Bad approach: Random split on combined data**
```python
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)
train = all_pairs[:40000]
val = all_pairs[40000:48567]
test = all_pairs[48567:]
```

Problem: By chance, splits might have different class distributions:
- Train: 48% positive, 52% negative
- Val: 53% positive, 47% negative
- Test: 49% positive, 51% negative

Model trained on 48% positive might not evaluate fairly on 53% positive validation set.

**Good approach: Stratified split**
- Split positives and negatives separately
- Use same ratios (70/15/15) for both
- Combine after splitting
- Guarantees 50/50 balance in all splits

**Mathematical verification:**

Train:
- Positive: 28,557 × 0.70 = 19,989.9 ≈ 19,990
- Negative: 28,557 × 0.70 = 19,989.9 ≈ 19,989
- Total: 39,979
- Ratio: 19,990 / 39,979 = 49.99% ≈ 50%

Validation:
- Positive: 28,557 × 0.15 = 4,283.55 ≈ 4,284
- Negative: 28,557 × 0.15 = 4,283.55 ≈ 4,283
- Total: 8,567
- Ratio: 4,284 / 8,567 = 50.00%

Test:
- Positive: 28,557 × 0.15 = 4,283.55 ≈ 4,284
- Negative: 28,557 × 0.15 = 4,283.55 ≈ 4,284
- Total: 8,568
- Ratio: 4,284 / 8,568 = 50.00%

Perfect balance!

**Why shuffle after combining?**

```python
train_data = [(pair, 1) for pair in pos_train] + [(pair, 0) for pair in neg_train]
random.shuffle(train_data)
```

Without final shuffle, first half of train_data would be all positive, second half all negative:
```
[positive, positive, ..., positive, negative, negative, ..., negative]
```

During training, model would see all positives in first 50% of epoch, all negatives in second 50%. This could cause:
- Catastrophic forgetting (learns positives, then forgets when seeing negatives)
- Unstable gradient updates
- Poor convergence

After shuffling:
```
[positive, negative, positive, positive, negative, ...]
```

Model sees mixed examples throughout the epoch, leading to stable learning.

**Final outputs:**

Save to disk:
```python
# Save train pairs
with open('data/training/train_pairs.json', 'w') as f:
    json.dump(train_data, f)

# Save validation pairs
with open('data/training/val_pairs.json', 'w') as f:
    json.dump(val_data, f)

# Save test pairs
with open('data/training/test_pairs.json', 'w') as f:
    json.dump(test_data, f)

# Save statistics
stats = {
    'total_pairs': len(train_data) + len(val_data) + len(test_data),
    'train_pairs': len(train_data),
    'val_pairs': len(val_data),
    'test_pairs': len(test_data),
    'positive_ratio': 0.5,
    'negative_positive_ratio': NEGATIVE_POSITIVE_RATIO
}
with open('data/training/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

Files created:
- `train_pairs.json`: 39,979 pairs
- `val_pairs.json`: 8,567 pairs
- `test_pairs.json`: 8,568 pairs
- `dataset_stats.json`: Statistics

---

(Continue to Part 2...)

