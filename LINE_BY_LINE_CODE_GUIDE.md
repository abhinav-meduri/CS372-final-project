# Line-by-Line Code Walkthrough - Full Pipeline

**Purpose:** Explain every important file in your pipeline with clear line references and practical explanations.

**Pipeline Order:**
1. Embedding Generation → 2. Training Data → 3. Feature Engineering → 4. Hyperparameter Tuning → 5. Model Training → 6. Inference → 7. LLM Integration → 8. Web App

## Files Covered (10 Total)

**Data Pipeline:**
1. `scripts/data/preprocessing/generate_embeddings.py` - Convert 200K patents to embeddings
2. `scripts/training/extract_citation_pairs.py` - Create 57K training pairs from citations

**Feature Engineering:**
3. `src/features/feature_extract.py` - FeatureExtractor class (10 features defined)
4. `scripts/data/preprocessing/compute_features.py` - Compute features for all pairs

**Model Training:**
5. `scripts/evaluation/tuning/nn_tuning.py` - Hyperparameter tuning (54 configs, 3-fold CV)
6. `src/app/pytorch_classifier.py` - PyTorch model class + training logic

**Inference:**
7. `src/app/patent_analyzer.py` - Main analysis orchestrator
8. `src/app/phi3_explainer.py` - Phi-3 LLM for explanations
9. `data/api/online_search.py` - SerpAPI + LLM keyword extraction

**Application:**
10. `app.py` - Streamlit web interface

---

## File 1: `scripts/data/preprocessing/generate_embeddings.py`
**Pipeline Stage:** Data Preprocessing (Step 1)  
**Purpose:** Convert 200K patents into 768-dimensional embeddings using PatentSBERTa

### Lines 1-20: Imports and Setup
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
```
**What it does:** Import PatentSBERTa model (sentence_transformers), numpy for array operations, json for reading patent data.

### Lines 32-56: `get_patent_text()` Function
**Line 32:** Function definition - extracts text from a patent dictionary
**Lines 34-35:** Try to get abstract first (best for embedding - concise but comprehensive)
**Lines 37-38:** Fall back to summary if no abstract
**Lines 39-47:** Fall back to first claim if no summary (handles both dict and string formats)
**Line 48:** Return empty string if nothing found

**Why this matters:** PatentSBERTa works best on 100-500 word texts. Abstracts are perfect. We truncate at 500 chars to fit the model's 512 token limit.

### Lines 59-86: `generate_embeddings_batch()` Function
**Line 63:** Takes a list of patents and the model
**Lines 67-70:** Extract text from each patent using `get_patent_text()`
**Lines 73-80:** Call `model.encode()` - THIS IS THE KEY LINE
  - Takes list of texts
  - Runs each through PatentSBERTa (BERT with 12 transformer layers)
  - Returns (N, 768) numpy array where each row is one patent's embedding
  - `batch_size=32`: Process 32 at once for efficiency
  - `convert_to_numpy=True`: Get numpy array not torch tensor

**Lines 82:** Return the embeddings array

**Conceptually:** This function transforms text → numbers. "A wireless charging system..." becomes [0.23, -0.56, 0.12, ..., 0.89] (768 numbers that capture meaning).

### Lines 90-180: `main()` Function - The Full Pipeline
**Lines 94-96:** Load PatentSBERTa model from HuggingFace
**Lines 99-106:** Detect if GPU available (MPS for Apple, CUDA for NVIDIA, else CPU)
**Lines 109-118:** Load all 200K patents from JSONL file (one JSON per line)
**Lines 121-140:** Main processing loop:
  - **Line 125:** Process in batches (0→32, 32→64, 64→96, etc.)
  - **Line 127:** Generate embeddings for this batch
  - **Line 128:** Accumulate in list
  - **Lines 131-136:** Save checkpoint every 10K patents (in case of crash)

**Lines 143-144:** Stack all batch embeddings into single (200000, 768) array
**Lines 152-154:** Save embeddings as numpy file (586 MB)
**Lines 157-160:** Save patent IDs as JSON (for lookup later)

**Total Output:**
- `patent_embeddings.npy`: 200,000 × 768 float32 array
- `patent_ids.json`: List of 200K patent IDs

**Pipeline Connection:** These embeddings are used for:
1. Local similarity search (cosine similarity)
2. Feature #1 in feature extraction
3. Claim embedding in Feature #9

---

## File 2: `scripts/training/extract_citation_pairs.py`
**Pipeline Stage:** Training Data Creation (Step 2)  
**Purpose:** Create labeled pairs from patent citations

### Lines 64-113: `generate_negative_pairs()` Function
**Line 64:** Takes positive pairs (citations) and generates negative pairs
**Lines 74-76:** For each positive pair (patent A cites patent B):
**Line 79:** Get all patents that A cites (the citation set)
**Lines 82-93:** Sample random patents NOT in that citation set
  - **Line 86:** Pick random patent C from full database
  - **Line 89:** Check C is not in A's citation set
  - **Line 91:** If valid, add (A, C) as negative pair

**Why this works:** If A cites B, they're related (positive). If A doesn't cite C (and C is random), they're likely unrelated (negative).

**Result:** 28,557 positive pairs + 28,557 negative pairs = 57,114 total

### Lines 150-220: `create_train_val_test_split()` Function
**Lines 161-166:** Shuffle positive and negative pairs separately
**Lines 169-172:** Calculate split sizes (70% train, 15% val, 15% test)
**Lines 175-180:** Split positives into 3 sets
**Lines 183-188:** Split negatives into 3 sets (same sizes)
**Lines 191-194:** Combine positives + negatives, label them (1 for pos, 0 for neg)
**Lines 197-199:** Shuffle each split (so not all positives then all negatives)

**Result:**
- Train: 39,979 pairs (50/50 positive/negative)
- Val: 8,567 pairs (50/50)
- Test: 8,568 pairs (50/50)

**Pipeline Connection:** These pairs are the ground truth for training. We'll extract features for each pair and train the model to predict the label (similar=1, different=0).

---

## File 3A: `src/features/feature_extract.py`
**Pipeline Stage:** Feature Engineering (Step 3a)  
**Purpose:** Define FeatureExtractor class with 10 feature methods

## File 3B: `scripts/data/preprocessing/compute_features.py`
**Pipeline Stage:** Feature Engineering (Step 3b)  
**Purpose:** Run feature extraction on all train/val/test pairs

### Lines 20-80: `FeatureExtractor` Class Initialization
**Lines 28-35:** Load embeddings and create ID→index mapping
**Line 37:** Initialize PatentSBERTa model (for runtime embedding)
**Lines 39-40:** Cache for claim embeddings (avoid re-computing)

### Lines 82-350: `extract_features()` Method - THE CORE
**Line 82:** Takes two patent dictionaries, returns 10-element array

**Feature 1 (Lines 90-95): PatentSBERTa Cosine Similarity**
```python
emb_a = embeddings[id_to_idx[patent_a_id]]  # Lookup embedding
emb_b = embeddings[id_to_idx[patent_b_id]]
cosine_sim = np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))
```
**What it does:** Compute angle between embedding vectors. Similar patents have similar embeddings → cosine ≈ 1.

**Feature 2 (Lines 98-112): TF-IDF Cosine Similarity**
```python
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
tfidf_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
```
**What it does:** Traditional keyword matching. TF-IDF weights words by importance (rare words get higher weight). Then compute cosine similarity on these vectors.

**Feature 3 (Lines 115-123): Jaccard Similarity**
```python
words_a = set(text_a.lower().split())
words_b = set(text_b.lower().split())
jaccard = len(words_a & words_b) / len(words_a | words_b)
```
**What it does:** Simple word overlap. If they share 20 words out of 100 total unique words, Jaccard = 0.20.

**Feature 4 (Lines 126-132): Claim Count Ratio**
```python
claims_a = len(patent_a['claims'])
claims_b = len(patent_b['claims'])
ratio = min(claims_a, claims_b) / max(claims_a, claims_b)
```
**What it does:** Patents with similar scope have similar numbers of claims. 15 claims vs 12 claims → ratio = 0.80.

**Feature 5 (Lines 135-141): Abstract Length Ratio**
```python
len_a = len(text_a.split())
len_b = len(text_b.split())
ratio = min(len_a, len_b) / max(len_a, len_b)
```
**What it does:** Same-domain patents have similar abstract lengths. Biotech = 300 words, mechanical = 120 words.

**Feature 6 (Lines 144-149): Year Difference**
```python
year_diff = 1.0 / (1.0 + abs(year_a - year_b))
```
**What it does:** Patents from same year → 1.0. One year apart → 0.5. Five years apart → 0.167. Temporal proximity matters.

**Feature 7 (Lines 152-157): Assignee Match**
```python
match = float(assignee_a == assignee_b and assignee_a != '')
```
**What it does:** Binary feature. Same company = 1, different = 0. Apple patents cite other Apple patents.

**Feature 8 (Lines 160-167): CPC Code Overlap**
```python
cpc_a = set(patent_a['cpc_codes'])
cpc_b = set(patent_b['cpc_codes'])
jaccard = len(cpc_a & cpc_b) / len(cpc_a | cpc_b)
```
**What it does:** CPC codes are patent classifications (like H04W = wireless). Jaccard overlap of these sets.

**Feature 9 (Lines 170-195): Max Claim Embedding Similarity**
```python
claims_a_texts = [get_text(c) for c in claims_a[:10]]  # First 10 claims
claims_b_texts = [get_text(c) for c in claims_b[:10]]

embs_a = embedder.encode(claims_a_texts)  # Embed each claim
embs_b = embedder.encode(claims_b_texts)

max_sim = 0
for emb_a in embs_a:
    for emb_b in embs_b:
        sim = cosine_similarity(emb_a, emb_b)
        max_sim = max(max_sim, sim)  # Keep the highest
```
**What it does:** Embed individual claims, find maximum pairwise similarity. Catches if even ONE claim overlaps strongly (0.92 similarity) even if abstracts differ.

**Feature 10 (Lines 198-206): Title Similarity**
```python
title_emb_a = embedder.encode(title_a)
title_emb_b = embedder.encode(title_b)
sim = cosine_similarity(title_emb_a, title_emb_b)
```
**What it does:** Embed titles (5-20 words each), compute cosine similarity. Captures high-level conceptual similarity.

**Line 209:** Return all 10 features as numpy array `[f1, f2, ..., f10]`

**Pipeline Connection:** This class defines HOW to compute features. The next script actually runs it on all pairs.

---

### File 3B Continued: `scripts/data/preprocessing/compute_features.py`

**Lines 24-30: `load_patents()` Function**
```python
def load_patents(path):
    patents = {}
    with path.open("r") as f:
        for line in tqdm(f, desc="Loading patents"):
            patent = json.loads(line)
            patents[str(patent["patent_id"])] = patent
    return patents
```
**What it does:** Load all 200K patents from JSONL into dictionary (patent_id → patent_dict)

**Lines 41-57: `load_embeddings()` Function**
**Line 47:** Load embeddings numpy array (200K × 768)
**Lines 48-56:** Load corresponding patent IDs from JSON
**Line 57:** Return both (embeddings array + ID list)

**Lines 60-79: `compute_split_features()` Function**
**Line 64:** Loop through all pairs in this split (train/val/test)
**Lines 65-67:** Extract patent IDs and label from pair dict
**Lines 68-71:** Look up both patents, skip if either missing
**Line 72:** **KEY LINE:** `extractor.extract_features(p1, p2)` - compute 10 features
**Line 73:** Convert feature vector to array
**Line 74:** Accumulate labels
**Lines 77-78:** Stack into numpy arrays
**Result:** X matrix (N, 10) and y labels (N,)

**Lines 92-115: `main()` Function - The Pipeline**
**Line 93:** Load all 200K patents
**Line 94:** Load embeddings and IDs
**Line 96-98:** Create FeatureExtractor and give it embeddings
**Lines 103-110:** For each split (train, val, test):
  - Load pairs from `{split}_pairs.jsonl`
  - Compute features using `compute_split_features()`
  - Print statistics

**What this script outputs:**
- `train_features_v2.X.npy`: (39979, 10) - train features
- `train_features_v2.y.npy`: (39979,) - train labels
- `val_features_v2.X.npy`: (8567, 10) - validation features
- `val_features_v2.y.npy`: (8567,) - validation labels
- `test_features_v2.X.npy`: (8568, 10) - test features
- `test_features_v2.y.npy`: (8568,) - test labels

**Pipeline Connection:** These feature files are loaded by the training script.

---

## File 4A: `scripts/evaluation/tuning/nn_tuning.py`
**Pipeline Stage:** Hyperparameter Tuning (Step 4a)  
**Purpose:** Find best hyperparameters using 3-fold cross-validation

### Lines 20-34: `load_features()` Function
**Lines 23-28:** Load train/val/test features from numpy files
**Lines 29-33:** Load feature names from JSON
**What it does:** Load the feature files created by `compute_features.py`

### Lines 37-44: `build_param_grid()` Function
```python
return {
    "hidden_dims": [[128, 64], [128, 64, 32], [256, 128]],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [0.0005, 0.001, 0.002],
    "weight_decay": [1e-5, 1e-4],
    "batch_size": [256],
}
```
**What it does:** Define grid of hyperparameters to try
- 3 architectures × 3 dropout rates × 3 learning rates × 2 weight decays = 54 configurations

### Lines 62-115: `main()` Function - Grid Search
**Lines 64-67:** Load features and combine train+val for cross-validation
**Lines 70-71:** Generate all 54 hyperparameter combinations
**Line 73:** Create 3-fold cross-validation splitter
**Lines 79-99:** For each configuration:
  - **Line 87-97:** Create PyTorchPatentClassifier with these hyperparameters
  - **Line 98:** Train on this fold
  - **Line 99:** Evaluate and get ROC-AUC
  - **Line 100:** Accumulate scores across 3 folds
**Lines 101-109:** Average scores across folds, track best configuration

**Lines 112-125:** After trying all configs:
  - Train final model on best configuration
  - Evaluate on held-out test set
  - Save results

**What this outputs:**
```json
{
  "best_params": {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "learning_rate": 0.002,
    "weight_decay": 1e-5,
    "batch_size": 256
  },
  "best_cv_score": 0.9717,
  "test_metrics": {
    "accuracy": 0.9173,
    "roc_auc": 0.9720
  }
}
```

**Pipeline Connection:** The best hyperparameters found here are used in the final model.

---

## File 4B: `src/app/pytorch_classifier.py`
**Pipeline Stage:** Model Training (Step 4)  
**Purpose:** Train neural network to predict similarity from 10 features

### Lines 26-66: `ResidualBlock` Class
**Lines 43-46:** Main transformation path: Linear → BatchNorm → ReLU → Dropout
**Lines 49-54:** Skip connection: If dimensions change, project input; else identity
**Lines 56-66:** Forward pass: `output = transformation(x) + skip(x)`

**Why residual:** The `+ skip(x)` creates alternate gradient path, preventing vanishing gradients. Allows deeper networks.

### Lines 69-139: `PatentNoveltyNet` Class
**Line 90:** Input BatchNorm normalizes the 10 features
**Lines 92-103:** Build hidden layers (10→256, 256→128)
**Lines 107-108:** Output layer: 128→1, then sigmoid to [0,1]

**Forward pass (lines 121-138):**
1. Normalize input features
2. Pass through residual blocks (10→256→128)
3. Linear layer (128→1)
4. Sigmoid squashes to probability

**Architecture:** Input(10) → BN → ResBlock(10→256) → ResBlock(256→128) → BN → Linear(128→1) → Sigmoid

### Lines 239-379: `fit()` Method - Training Loop
**Lines 264-265:** Scale features with StandardScaler (mean=0, std=1)
**Lines 268-275:** Create model instance
**Lines 278-287:** Define loss (BCE), optimizer (AdamW), scheduler

**Training Loop (Lines 300-371):**
**Lines 302-325:** Training phase for each batch:
  - **Line 306:** Iterate through batches
  - **Lines 311-315:** Optional mixup augmentation (interpolate examples)
  - **Line 317:** Zero gradients
  - **Line 318:** Forward pass through model
  - **Line 319:** Compute binary cross-entropy loss
  - **Line 320:** Backward pass (compute gradients)
  - **Line 321:** Clip gradients (prevent explosion)
  - **Line 323:** Update weights with optimizer

**Lines 330-356:** Validation phase:
  - **Line 331:** Set model to eval mode (disables dropout)
  - **Line 336:** `with torch.no_grad()`: Don't track gradients (saves memory)
  - **Lines 337-346:** Forward pass on validation batches
  - **Lines 349-352:** Compute accuracy from predictions

**Lines 357:** Learning rate scheduler adjusts LR if plateau
**Lines 359-371:** Early stopping logic:
  - If val loss improves: save model, reset patience
  - If no improvement for 15 epochs: stop training
  - Restore best model

**Result:** Trained model achieving 97.2% ROC-AUC on test set

**Pipeline Connection:** This trained model (118K parameters, 462 KB) is saved to `models/pytorch_nn/pytorch_model.pt` and loaded during inference.

---

## File 5: `src/app/patent_analyzer.py`  
**Pipeline Stage:** Inference (Step 5)  
**Purpose:** Main orchestrator for analyzing user's patent

### Lines 120-193: `load()` Method
**Lines 133-137:** Load embeddings (memory-mapped, 586 MB)
**Lines 139-143:** Load PatentSBERTa model
**Lines 145-149:** Initialize Phi-3 explainer
**Lines 151-155:** Initialize LLM keyword extractor
**Lines 157-166:** Initialize SerpAPI searcher (if API key provided)
**Lines 168-186:** Load PyTorch model and feature extractor

**What it does:** Lazy loading of all components. Only loads when first called, not at initialization.

### Lines 400-650: `analyze()` Method - THE MAIN PIPELINE

**Step 1 - Generate Query Embedding (Lines 410-412):**
```python
query_embedding = self.st_model.encode(query_text)  # Returns (768,)
```
**What it does:** User's patent text → 768-dim vector via PatentSBERTa

**Step 2 - Local Search (Lines 415-420):**
```python
local_results = self._find_similar(query_embedding, top_k=15)
```
Jumps to `_find_similar()` method...

### Lines 633-662: `_find_similar()` Method
**Lines 636-638:** Normalize query and database embeddings to unit length
**Line 640:** Compute all 200K cosine similarities: `similarities = all_norms @ query_norm`
**Line 642:** Get indices of top 15: `top_indices = argsort(similarities)[-15:]`
**Lines 645-660:** Load patent metadata for these 15 patents, return as list

**What it does:** Fast similarity search over 200K patents in ~1 second using optimized numpy.

**Step 3 - LLM Keyword Extraction (Lines 425-435 in analyze()):**
```python
if self.use_llm_keywords:
    keywords = self.keyword_extractor.extract_keywords(query_text)
```
**What it does:** Phi-3 generates 5 smart search terms like "wireless power transfer AND magnetic resonance"

**Step 4 - Online Search (Lines 440-450):**
```python
online_results = self.online_searcher.search_multiple_terms(keywords, max_per_term=10)
```
**What it does:** Query Google Patents via SerpAPI for each keyword. Returns up to 50 online patents.

**Step 5 - Merge Results (Lines 455-460):**
```python
all_candidates = self._merge_results(local_results, online_results)
```
**What it does:** Combine local + online, deduplicate by ID. Typically 50-70 unique candidates.

**Step 6 - Feature Extraction (Lines 465-475):**
```python
for candidate in all_candidates:
    features = self.feature_extractor.extract_features(query_text, candidate)
    # Returns (10,) array
```
**What it does:** Compute 10 features for (user_patent, candidate) pair. Creates (N, 10) matrix.

**Step 7 - PyTorch Scoring (Lines 480-490):**
```python
scores = self.pytorch_model.predict_proba(features_matrix)
for i, candidate in enumerate(all_candidates):
    candidate['model_similarity'] = scores[i]
    candidate['model_novelty'] = 1 - scores[i]
```
**What it does:** Run all candidates through trained PyTorch model in batch. Get similarity probabilities [0,1].

**Step 8 - Ranking (Lines 495-500):**
```python
candidates.sort(key=lambda x: x['model_similarity'], reverse=True)
top_20 = candidates[:20]
novelty_score = 1 - mean([p['model_similarity'] for p in top_20])
```
**What it does:** Sort by similarity, keep top 20, compute overall novelty as 1 - average similarity.

**Step 9 - LLM Explanation (Lines 505-515):**
```python
explanation = self.explainer.generate_explanation(
    query_patent={'text': query_text},
    similar_patents=top_20[:10],
    novelty_score=novelty_score
)
```
**What it does:** Send to Phi-3 to generate human-readable report.

**Lines 520-535:** Package everything into `AnalysisResult` and return

**Total Pipeline Time:** 60-90 seconds (most time in LLM steps 3 and 9)

---

## File 6: `src/app/phi3_explainer.py`
**Pipeline Stage:** LLM Integration (Step 6a)  
**Purpose:** Generate explanations using local Phi-3 model

### Lines 76-157: `_build_prompt()` Method
**Lines 87-94:** Determine assessment based on novelty score (HIGH/MODERATE/LOW/NOT NOVEL)
**Lines 96-106:** Build prompt header with user's patent title and abstract
**Lines 108-120:** Add top 4 similar patents with their details
**Lines 123-156:** Structured prompt asking for:
  - Executive Summary (cite specific patents)
  - Technical Overlap Analysis (quote overlapping text)
  - Novel Elements (what's new)
  - Recommendation (approve/revise/reject)

**What it does:** Create detailed prompt that guides Phi-3 to generate structured novelty report.

### Lines 159-227: `generate_explanation()` Method
**Lines 176-197:** HTTP POST to Ollama API:
```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3",
        "prompt": prompt,
        "options": {
            "num_predict": 4000,  # Max tokens to generate
            "temperature": 0.4    # Low = factual, high = creative
        }
    }
)
```
**Line 201:** Extract generated text from response
**Lines 204-211:** Log timing info (tokens/second)

**What it does:** Call local Phi-3 model via Ollama, generate 800-1200 tokens in 30-45 seconds.

**Lines 229-283: `_parse_response()`** - Parse LLM output into structured report

**Pipeline Connection:** This explanation is shown to the user in the web app's "AI Explanation" section.

---

## File 7: `data/api/online_search.py`
**Pipeline Stage:** LLM Integration (Step 6b)  
**Purpose:** Online search via SerpAPI + keyword extraction

### Lines 65-148: `LLMKeywordExtractor` Class

**Lines 81-148: `generate_search_terms()` Method**
**Lines 88-109:** Build prompt asking Phi-3 to generate 5 search queries
**Lines 112-121:** POST to Ollama with temperature=0.3 (fairly focused)
**Lines 127-138:** Parse numbered list from LLM response:
```python
# LLM returns:
# 1. wireless power transfer AND magnetic resonance
# 2. inductive charging system OR resonant coupling
# ...

for line in text.split('\n'):
    match = re.match(r'^\d+[\.\)]\s*(.+)$', line)  # Match "1. query"
    if match:
        queries.append(match.group(1))
```

**What it does:** Use Phi-3 to generate smart search terms that use Google Patents boolean syntax.

### Lines 319-489: `GooglePatentsSearch` Class

**Lines 352-367: `search()` Method**
```python
def search(query, max_results=10):
    if self.use_serpapi:
        return self._search_serpapi(query, max_results)
    else:
        return []  # No API key
```

**Lines 401-477: `_search_serpapi()` Method - THE ACTUAL SEARCH**
**Lines 420-425:** Build SerpAPI parameters:
```python
params = {
    "engine": "google_patents",
    "q": query,
    "api_key": self.serpapi_key,
    "num": num_results  # Must be 10-100
}
```
**Lines 428-429:** Make API call: `search = GoogleSearch(params); results = search.get_dict()`
**Lines 437-468:** Parse results:
  - Extract patent_id, title, abstract from each result
  - Create `PatentSearchResult` objects
  - Add metadata (year, URL, assignee)

**Lines 369-399: `search_multiple_terms()` Method**
```python
for term in terms:  # e.g., 5 terms
    results = self.search(term, max_per_term=10)  # Get 10 per term
    results_by_term[term] = results
```
**What it does:** Search each of the 5 LLM-generated terms, collect all results (deduplicated later).

**Pipeline Connection:** Returns 40-50 patents from Google's millions of patents, complementing local 200K database.

---

## File 8: `app.py`
**Pipeline Stage:** Web Application (Step 7)  
**Purpose:** Streamlit UI that ties everything together

### Lines 20-35: `load_analyzer()` Function
```python
@st.cache_resource
def load_analyzer(serpapi_key, use_online, use_keywords):
    analyzer = PatentAnalyzer(
        use_online_search=use_online,
        use_llm_keywords=use_keywords,
        serpapi_key=serpapi_key
    )
    analyzer.load()  # Calls the load() method we saw earlier
    return analyzer
```
**What `@st.cache_resource` does:** Load models ONCE and reuse across requests. Without this, every button click would reload PatentSBERTa (10 seconds).

### Lines 550-630: Sidebar Configuration
**Lines 551-563:** SerpAPI key input (password field)
**Lines 566-568:** Checkboxes for online search and LLM keywords
**Lines 570-571:** Slider for number of results

**What it does:** Let user configure the system without touching code.

### Lines 600-750: Main Analysis Flow
**Lines 605-615:** Get user input (text area for patent description)
**Lines 620-625:** Load analyzer (cached, so fast after first load)
**Lines 630-640:** Run analysis:
```python
result = analyzer.analyze(
    query_text,
    top_k=num_results,
    status_callback=update_status  # Updates progress bar
)
```
**Lines 645-750:** Display results:
  - Novelty score card (color-coded green/orange/red)
  - Table of similar patents
  - AI explanation (expandable)
  - Download buttons for report

**What it does:** User interface that calls `analyzer.analyze()` and displays the result.

---

## COMPLETE PIPELINE FLOW

**User Input:** "A wireless charging system using magnetic resonance..."

**1. Generate Embedding** (`patent_analyzer.py` lines 410-412)
→ PatentSBERTa encodes text → [0.23, -0.56, ..., 0.89] (768 numbers)

**2. Local Search** (`patent_analyzer.py` lines 633-662)
→ Compute 200K cosine similarities in numpy
→ Return top 15: US11234567 (0.87 sim), US10987654 (0.76 sim), ...

**3. LLM Keywords** (`online_search.py` lines 81-148)
→ Phi-3 generates: "wireless power transfer AND magnetic resonance", "inductive charging", ...

**4. Online Search** (`online_search.py` lines 401-477)
→ Query Google Patents via SerpAPI for each keyword
→ Return 45 patents: WO2023123456, EP3456789, ...

**5. Merge** (`patent_analyzer.py` line 455)
→ Combine local + online, dedupe → 58 unique candidates

**6. Extract Features** (`feature_extract.py` lines 82-209)
→ For each candidate, compute 10 features
→ Creates 58×10 matrix

**7. PyTorch Scoring** (`pytorch_classifier.py` lines 381-390)
→ Batch inference through trained model
→ Get similarity scores: [0.92, 0.78, 0.85, ..., 0.23]

**8. Rank & Calculate Novelty** (`patent_analyzer.py` lines 495-500)
→ Sort by score, take top 20
→ Novelty = 1 - mean(top 20) = 1 - 0.75 = 0.25 (LOW)

**9. LLM Explanation** (`phi3_explainer.py` lines 159-227)
→ Phi-3 generates report: "LOW NOVELTY - Significant prior art..."

**10. Display** (`app.py` lines 645-750)
→ Show results in Streamlit UI

**Total Time:** 73 seconds

---

## Key Methods Summary with Line Numbers

**Embedding Generation:**
- `generate_embeddings.py` lines 73-80: `model.encode()` - text to vectors
- `generate_embeddings.py` lines 125-136: Main processing loop

**Training Data:**
- `extract_citation_pairs.py` lines 86-91: Negative sampling logic
- `extract_citation_pairs.py` lines 191-199: Stratified split

**Feature Engineering:**
- `feature_extract.py` lines 90-95: Feature 1 (embedding similarity)
- `feature_extract.py` lines 98-112: Feature 2 (TF-IDF)
- `feature_extract.py` lines 170-195: Feature 9 (max claim similarity)
- `compute_features.py` lines 64-79: Compute features for all pairs
- `compute_features.py` lines 103-110: Process train/val/test splits

**Hyperparameter Tuning:**
- `nn_tuning.py` lines 37-44: Define parameter grid (54 configs)
- `nn_tuning.py` lines 79-99: 3-fold cross-validation loop
- `nn_tuning.py` lines 112-125: Train final model on best config

**Model Training:**
- `pytorch_classifier.py` lines 318-323: Training step (forward→backward→update)
- `pytorch_classifier.py` lines 359-371: Early stopping logic

**Inference:**
- `patent_analyzer.py` lines 640: Cosine similarity search
- `patent_analyzer.py` lines 480-490: PyTorch batch scoring
- `patent_analyzer.py` lines 495-500: Novelty calculation

**LLM Integration:**
- `phi3_explainer.py` lines 184-197: Ollama API call
- `online_search.py` lines 428-429: SerpAPI call

**Web App:**
- `app.py` line 20: `@st.cache_resource` decorator (crucial for performance)
- `app.py` lines 630-640: Main analysis call

---

## What You Should Be Able to Explain

For **each file**, be ready to explain:

1. **What it does** in the pipeline (1 sentence)
2. **Key methods** with line numbers (3-5 important ones)
3. **Input/Output** (what goes in, what comes out)
4. **How it connects** to the next stage

**Example for `generate_embeddings.py`:**
- "Converts 200K patents into 768-dim embeddings using PatentSBERTa"
- Key methods: `get_patent_text()` (lines 32-56), `model.encode()` (line 73), main loop (lines 125-136)
- Input: `patents_sampled.jsonl` (3.8 GB, 200K patents)
- Output: `patent_embeddings.npy` (586 MB, 200K×768 array)
- Connects to: Used in local search and feature extraction

This is your **practical reference** - actual line numbers, actual code, actual pipeline flow.

