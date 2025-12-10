# Complete Technical Walkthrough - Presentation Script

**How to use this:** Read this script while scrolling through the actual code files. This tells you exactly what to say for each important section.

---

# INTRODUCTION

Hi! I'm Abhinav and I will be presenting the technical walkthrough of my patent novelty assessment system.

## Project Overview

This project is a hybrid retrieval-augmented generation (RAG) system that assesses whether a patent application is novel by comparing it against 200,000 USPTO patents from 2021-2025. The system combines multiple techniques:

1. **Local semantic search** using PatentSBERTa embeddings and cosine similarity over 200K patents
2. **Online search** via Google Patents API to access millions more patents
3. **Supervised machine learning** - a custom PyTorch neural network trained on patent citations to score similarity
4. **LLM explanations** using Phi-3 running locally via Ollama to generate human-readable reports

The complete pipeline has 5 stages:
- **Stage 1: Data Collection & Preprocessing** - Download USPTO data, sample 200K diverse patents
- **Stage 2: Embedding Generation** - Convert patents to 768-dimensional semantic vectors using PatentSBERTa
- **Stage 3: Training Data Creation** - Extract 57K labeled pairs from patent citations
- **Stage 4: Feature Engineering & Model Training** - Extract 10 features, train PyTorch classifier achieving 97.2% ROC-AUC
- **Stage 5: Inference & Deployment** - Web app that combines all components for real-time patent analysis

Let me walk through each component in detail, showing you the actual implementation.

---

# STAGE 1: DATA COLLECTION & PREPROCESSING

## Overview Narrative

Before we can do any machine learning, we need data. USPTO publishes patent data as large TSV files. I downloaded files from 2021-2025, which gave me approximately 800,000 granted utility patents. From this, I created a stratified random sample of 200,000 patents to ensure temporal diversity and computational feasibility.

The data comes in two main files per year:
- `g_brf_sum_text_YYYY.tsv` - Contains patent IDs, titles, abstracts, summaries
- `g_claims_YYYY.tsv` - Contains patent claims text

I also downloaded the citation graph file `g_us_patent_citation.tsv` which contains all citation relationships between patents.

## File: `scripts/training/sample_diverse_patents.py`

**What to say while showing this file:**

"This script performs stratified sampling to create a diverse, balanced dataset of 200,000 patents from the full USPTO database. Let me show you how it works."

### Lines 24-30: `count_patents_in_file()`
"First, we have a utility function that counts how many patents are in each year's file. This helps us plan our sampling strategy. It simply iterates through the JSONL file and counts lines."

### Lines 33-40: `load_all_patent_ids()`
"This function loads all patent IDs from a file along with their line numbers. We need the line numbers because we'll randomly sample some patents, and we need to know which lines to read without loading everything into memory."

**Implementation detail:** Notice at line 38-39, we're creating tuples of (patent_id, line_number). This is memory-efficient because we're not loading the full patent data, just the IDs.

### Lines 43-75: `sample_patents_from_file()`
"Here's where the actual random sampling happens."

**Line 58:** "We set a random seed for reproducibility - this is crucial for scientific work. Same seed always gives same sample."

**Lines 59-65:** "We check if we have enough patents in this file. If we're asking for 20,000 but the file only has 15,000, we take all 15,000. Otherwise, we use Python's `random.sample()` to randomly select without replacement."

**Lines 67-73:** "Now here's an efficiency trick - we only read the lines we actually selected. Instead of loading all 50,000 patents and then filtering, we know exactly which line numbers we need, so we skip the rest. This saves a lot of memory and time."

### Lines 78-203: `create_diverse_sample()` - The Main Algorithm
"This is the core function that creates our stratified sample. Let me explain the strategy."

**Lines 98-103:** "First, we find all the patent files. They're named like `patents_2021.jsonl`, `patents_2022.jsonl`, etc. We use glob pattern matching to find them all."

**Lines 107-113:** "We count how many patents we have per year. This gives us the distribution. For example, maybe 2021 has 180,000 patents, 2022 has 170,000, and so on."

**Lines 118-131:** "Here's the stratification logic. We want equal representation from each year to avoid temporal bias. So we divide our target of 200,000 by the number of years. If we have 5 years, that's 40,000 per year."

**Why this matters:** "If we just randomly sampled 200,000 from all years combined, we might get 80,000 from 2021 (a busy year) and only 15,000 from 2024. This would bias our model toward older patent writing styles and technologies."

**Lines 133-141:** "We handle edge cases where some years might have fewer patents than our target. Any remaining quota gets distributed to years that can handle more."

**Lines 148-162:** "Now we actually perform the sampling for each year. Notice at line 160, we add the year to the random seed. This ensures each year gets a different random sample, but it's still reproducible."

**Lines 164-166:** "After collecting all samples, we shuffle the combined list. This breaks any year-based ordering so the model doesn't learn spurious temporal patterns during training."

**Lines 168-174:** "We write everything to a JSONL file - JSON Lines format. Each line is one patent as a JSON object. This format is great because it's streamable - you can read one line at a time without loading the whole 3.8 GB file."

**Lines 176-185:** "Finally, we save metadata about how we sampled. This includes the random seed, samples per year, and source files. This is crucial for reproducibility - someone else can verify our sampling strategy."

**Lines 192-201:** "As a sanity check, we verify the year distribution in our final sample. We should see roughly equal percentages for each year."

### Result
"After running this script, we have `patents_sampled.jsonl` - 200,000 patents, stratified by year, ready for embedding and training. The file is 3.8 GB and contains all the metadata we need: titles, abstracts, claims, CPC codes, assignees, and years."

---

# STAGE 2: EMBEDDING GENERATION

## Overview Narrative

"Now that we have our 200,000 patents, we need to convert them into numerical representations that machine learning models can work with. This is where PatentSBERTa comes in."

**What are embeddings?** "An embedding is a dense vector representation of text that captures semantic meaning. Instead of representing 'wireless charging system' as a sparse bag of words like a traditional one-hot encoding, we represent it as a point in 768-dimensional space. The key insight is that similar concepts are close together in this space. Two patents about wireless charging might be separated by a distance of 0.3, while a wireless patent and a pharmaceutical patent might be 1.5 apart."

**Why PatentSBERTa specifically?** "PatentSBERTa is a BERT model that was fine-tuned by the AI-Growth-Lab research group on 1.2 million patent abstracts from the USPTO database. Regular BERT was trained on Wikipedia and books - it understands general language. PatentSBERTa understands patent-specific terminology and language patterns. For example, it knows that 'prior art' in patents means something very different from 'previous work' in academic papers. It understands 'claims' refers to legally binding statements, not just assertions. It recognizes technical jargon across electrical engineering, mechanical engineering, biotechnology, and software domains. It understands the legal writing style of patent documents."

"This domain-specific training is crucial. A general-purpose embedding model like regular BERT might conflate 'battery charging' with 'criminal charging' because they share a word. PatentSBERTa understands the context and knows these are completely different concepts."

## File: `scripts/data/preprocessing/generate_embeddings.py`

**What to say:**

"This script converts all 200,000 patents into 768-dimensional embedding vectors. This was the most computationally expensive preprocessing step - it took about 11 hours to run on my Apple M1 machine with MPS GPU acceleration. Without GPU, this would have taken 50-60 hours."

### Lines 1-30: Imports and Setup

"Let me show you the imports first. We're using the sentence-transformers library which provides easy access to BERT-based models. PatentSBERTa is hosted on HuggingFace's model hub, so sentence-transformers can download it automatically. We also import torch for GPU acceleration detection, numpy for array operations, json for loading patent data, and tqdm for progress bars during the 11-hour run."

**Key imports explained:**
- "sentence_transformers provides the SentenceTransformer class which wraps BERT models and makes encoding text very simple"
- "torch is the PyTorch backend - we need this to detect if we have MPS (Apple Silicon GPU) or CUDA (NVIDIA GPU) available"
- "numpy for efficient array operations - we'll save embeddings as a numpy array which is much faster than JSON"
- "tqdm gives us progress bars so we can monitor the 11-hour embedding generation process"
- "pathlib for cross-platform path handling"

### Lines 32-56: `get_patent_text()` Function - Text Extraction Strategy

"This function looks simple but it's critically important. It determines what text we embed for each patent."

**The priority order is carefully chosen:**

**Line 34-35: Abstract (First Priority)**
"We prefer abstracts because they're the perfect balance. Patent abstracts are typically 100-300 words and provide a comprehensive overview of the invention. They describe what the invention is, what problem it solves, how it works, and what advantages it provides. This is what patent examiners read first when evaluating novelty, so it's what we should embed."

"Patents have multiple text sections - the title is too short at only 5-15 words and doesn't contain enough information. The detailed description can be thousands of words and would require chunking and aggregation. Claims are legally binding but written in complex legal language with nested clauses. Abstracts are the sweet spot."

**Lines 37-38: Summary (Second Priority)**
"If there's no abstract, we fall back to the summary section. Summaries are similar to abstracts but sometimes more verbose. They still capture the essence of the invention."

**Lines 39-47: First Claim (Third Priority)**
"If neither abstract nor summary exists, we use the first claim. The first claim is usually the broadest independent claim that describes the core invention. Notice the defensive programming here - claims can be formatted as dictionaries with a 'text' key or as plain strings. Different USPTO download batches format claims differently, so we handle both cases by checking isinstance. This prevents crashes during processing."

**Line 35, 38, 47: Why truncate to 500 characters?**
"PatentSBERTa is based on BERT architecture, which has a maximum sequence length of 512 tokens. A token is roughly 0.75 words on average - it could be a whole word like 'charging' or a subword like '##less' in 'wireless'. So 512 tokens is approximately 384 words, which is roughly 1920 characters. We use 500 characters to be conservative and ensure we never exceed the model's context window. What happens if we exceed it? The model would truncate automatically, potentially cutting off important information mid-sentence. Better to truncate ourselves cleanly at a character boundary."

**Lines 48-49: Return Empty String as Fallback**
"If a patent has no abstract, no summary, and no claims, we return an empty string instead of None. Why? The sentence-transformers library expects a string. If we pass None, it will crash with a TypeError. An empty string gets embedded as a near-zero vector, which has ~0.0 cosine similarity to everything - exactly what we want for patents with no text. This is defensive programming to handle edge cases gracefully."

### Lines 59-86: `generate_embeddings_batch()` Function
"This is where the magic happens - converting text to embeddings."

**Lines 67-70:** "For each patent in the batch, we extract its text using the function we just discussed."

**Lines 73-80: THE KEY LINE - `model.encode()`**
"This single line does a tremendous amount of work. Let me break down what's happening inside model.encode():

1. **Tokenization**: The text is split into subword tokens using BERT's WordPiece tokenizer. For example, 'wireless charging system' might become ['wire', '##less', 'charging', 'system'].

2. **Special tokens**: BERT adds [CLS] at the beginning and [SEP] at the end. The [CLS] token will eventually contain the representation of the entire sequence.

3. **Input IDs**: Each token is converted to an integer ID from BERT's 30,000-word vocabulary.

4. **Attention masks**: A mask is created showing which tokens are real (1) vs padding (0).

5. **Transformer layers**: The input passes through 12 transformer encoder layers. Each layer has:
   - Multi-head self-attention (8 heads, 64 dimensions each = 768 total)
   - Layer normalization
   - Feed-forward network (768 → 3072 → 768)
   - Residual connections

6. **Pooling**: We extract the final hidden state of the [CLS] token from layer 12. This 768-dimensional vector is our embedding."

**Why batch_size=32?** "This is an efficiency trade-off. Too small (like 1) means we underutilize the GPU. Too large (like 256) might cause out-of-memory errors. 32 is the sweet spot for M1's Metal Performance Shaders - good GPU utilization without memory issues."

**convert_to_numpy=True:** "By default, model.encode returns PyTorch tensors. We convert to NumPy because NumPy is easier to save/load and works well with our later processing steps."

**normalize_embeddings=False:** "We'll normalize during search instead. This saves computation time during the 11-hour embedding generation."

### Lines 90-180: `main()` Function - The Pipeline

**Lines 94-96:** "Load PatentSBERTa from HuggingFace. First time this runs, it downloads ~440MB of model weights."

**Lines 99-106: GPU Detection**
"The code checks for available hardware acceleration:
- CUDA: NVIDIA GPUs (fastest, 50+ patents/second)
- MPS: Apple Metal (Apple Silicon, ~18 patents/second) 
- CPU: Fallback (slowest, ~3 patents/second)

For M1, we use MPS which makes this about 5x faster than CPU."

**Lines 109-118:** "Load all 200,000 patents from the JSONL file. We use tqdm for a progress bar. Notice we wrap the JSON parsing in a try-except at line 114-117 to handle any malformed lines gracefully."

**Lines 121-140: The Main Processing Loop**
"This is where we process 200,000 patents in batches of 32."

**Line 125:** "We iterate in steps of 32: process patents 0-31, then 32-63, then 64-95, and so on."

**Line 127:** "Call our batch embedding function for this group of 32 patents."

**Line 128:** "Accumulate the results. We're building a list of arrays: [(32,768), (32,768), (32,768), ...]"

**Lines 131-136: Checkpointing**
"Every 10,000 patents (312 batches), we save a checkpoint. Why? Because this takes 11 hours. If it crashes at hour 10, we don't want to start over. We can resume from the last checkpoint."

**Lines 143-144:** "After all batches, we stack them vertically using numpy's vstack. This concatenates [(32,768), (32,768), ...] into one big (200000, 768) matrix."

**Lines 152-154: Save Embeddings**
"We save the embedding matrix as a .npy file. NumPy's binary format is efficient:
- 200,000 rows × 768 columns × 4 bytes per float32 = 614 million bytes = 586 MB
- Loading this file takes ~1 second vs several seconds for JSON or CSV."

**Lines 157-160: Save Patent IDs**
"We also save the mapping of array index to patent ID. Index 0 corresponds to the first patent ID, index 1 to the second, etc. During search, we'll get indices from argsort() and use this list to look up the actual patent IDs."

**Lines 162-168: Save Metadata**
"We record everything about the generation process: how long it took, which device we used, what model version. This is important for reproducibility and debugging."

### Performance Analysis
"Let me give you the numbers from when I ran this:
- Total time: 40,110 seconds = 11.1 hours
- Processing rate: 200,000 / 40,110 = 4.99 patents/second (overall)
- Pure embedding time: ~18 patents/second
- The difference accounts for I/O, checkpoint saving, and overhead

This is a one-time cost. Once we have the embeddings, searches are instant."

### Outputs
"This script produces three files:
1. `patent_embeddings.npy` - 586 MB, (200000, 768) float32 array
2. `patent_ids.json` - 2.3 MB, list of 200,000 patent IDs
3. `embedding_metadata.json` - stats about generation

These embeddings are used in three places later:
- Local similarity search (cosine similarity)
- Feature #1 in feature extraction
- Feature #9 for claim-level similarity"

---

# STAGE 3: TRAINING DATA CREATION

## Overview Narrative

"Now we have embeddings, but we need labeled training data to train a supervised classifier. The question is: where do we get labels for 'similar' vs 'different' patents?

The answer: patent citations! When patent A cites patent B, it means:
- The inventors explicitly acknowledged B as prior art
- Patent examiners verified the relevance
- There's a documented relationship

So citations give us positive examples. For negative examples, we randomly sample pairs that don't cite each other. This is called Positive-Unlabeled (PU) learning because we can't be 100% certain the negatives are truly unrelated - they might be related in ways not captured by citations. But random sampling ensures most are genuinely different."

## File: `scripts/training/extract_citation_pairs.py`

**What to say:**

"This script extracts training pairs from the USPTO citation graph and creates balanced train/validation/test splits."

### Lines 1-40: Setup and Citation Loading
"The script starts by loading the citation graph. This is a large TSV file from USPTO containing millions of citation relationships."

### Lines 64-113: `generate_negative_pairs()` - THE KEY ALGORITHM

"This function is crucial - it's how we create negative training examples."

**Lines 74-76:** "For each positive pair where patent A cites patent B, we're going to create a negative pair where A doesn't cite some other patent C."

**Line 79:** "We get the full citation set for patent A - all the patents it cites. Let's say A cites [B, X, Y, Z]."

**Lines 82-93: The Sampling Logic**
"Now we randomly sample from the full database until we find a patent that A does NOT cite."

**Line 86:** "Pick a random patent C from all 200,000 patents."

**Line 89:** "Check two conditions:
1. C is not the same as A (we don't want to pair a patent with itself)
2. C is not in A's citation set (we want a truly random, unrelated patent)"

**Line 91:** "If both conditions pass, create the negative pair (A, C)."

**Why this works:** "If we randomly sample from 200,000 patents and A only cites maybe 10-20 of them, the probability of randomly picking one A cites is tiny (~0.01%). So almost all our random samples will be genuinely unrelated patents."

**Lines 95-97: Safety Limit**
"We try up to 100 times to find a valid negative. In practice, we succeed on the first try 99%+ of the time. But this prevents infinite loops if we have a patent that somehow cites half the database."

### Lines 150-220: `create_train_val_test_split()` - Stratified Splitting

"Once we have all our pairs (positive and negative), we need to split them into train, validation, and test sets. The key requirement: maintain class balance in all splits."

**Lines 161-166: Shuffle Separately**
"We shuffle positive and negative pairs separately with a fixed random seed (42). This ensures reproducibility - same seed always gives same split."

**Lines 169-172: Calculate Split Sizes**
"We use a 70/15/15 split:
- 70% for training (where the model learns)
- 15% for validation (for hyperparameter tuning and early stopping)
- 15% for test (final evaluation, never seen during development)"

**Lines 175-188: Split Positives and Negatives Separately**
"This is the stratification. We split the 28,557 positive pairs into 70/15/15, and we split the 28,557 negative pairs into 70/15/15 using the SAME ratios. This guarantees each split has exactly 50% positive, 50% negative."

**Why stratify?** "If we just randomly split all 57,114 pairs, we might get unlucky - maybe training ends up 48% positive, validation 53% positive. This would make validation metrics not comparable to training. Stratification prevents this."

**Lines 191-199: Combine and Label**
"Now we combine the positive and negative pairs for each split and label them:
- Positive pairs get label = 1
- Negative pairs get label = 0

Then we shuffle each split so we don't have all positives followed by all negatives."

**Lines 201-210: Save to Files**
"We save three JSONL files:
- `train_pairs.jsonl` - 39,979 pairs (19,989 positive + 19,990 negative)
- `val_pairs.jsonl` - 8,567 pairs (balanced)
- `test_pairs.jsonl` - 8,568 pairs (balanced)

Each line contains: {patent_id_1, patent_id_2, label}"

### Results
"From the USPTO citation graph containing ~500,000 citations among our 200K patents, we extracted:
- 28,557 positive pairs (actual citations)
- 28,557 negative pairs (random non-citations)
- Total: 57,114 labeled pairs
- Split: 70% train (39,979), 15% val (8,567), 15% test (8,568)
- All splits have exactly 50/50 class balance"

---

# STAGE 4A: FEATURE ENGINEERING

## Overview Narrative

"Now we have labeled pairs, but we can't just feed patent IDs to a neural network. We need to extract meaningful features that capture different aspects of similarity. I engineered 10 features that represent semantic, lexical, structural, temporal, and categorical similarity. Let me show you how each feature is computed."

## File: `src/features/feature_extract.py`

**What to say:**

"This file defines the FeatureExtractor class which computes all 10 features for a patent pair. Each feature captures a different signal."

### Lines 20-80: `FeatureExtractor` Class Initialization

**Lines 28-35:** "We load the pre-computed embeddings and create an ID-to-index mapping. This lets us quickly look up the embedding for any patent ID."

**Line 37:** "We initialize PatentSBERTa for runtime embedding - we'll need this for Feature #9 (claim similarity) and Feature #10 (title similarity)."

**Lines 39-40:** "We use a cache dictionary for claim embeddings. If the same claim appears in multiple patents, we don't re-embed it. This saves computation."

### Lines 82-350: `extract_features()` Method - WHERE ALL 10 FEATURES ARE COMPUTED

"This is the heart of feature engineering. Let me walk through each feature in detail."

### FEATURE 1: PatentSBERTa Cosine Similarity (Lines 90-95)

"Feature 1 measures semantic similarity using the pre-computed embeddings."

**Line 90-91:** "Look up the embeddings for both patents using their IDs. These are 768-dimensional vectors."

**Line 93-95:** "Compute cosine similarity using the formula: cos(θ) = (A·B) / (||A|| × ||B||)

The dot product measures alignment. We normalize by the magnitudes so the result is always between -1 and 1, where:
- 1 means identical vectors (perfect similarity)
- 0 means orthogonal (unrelated)
- -1 means opposite (rare for patents)

For patents, we typically see values between 0.05 and 0.95."

**Implementation detail:** "We add 1e-8 to the denominator to prevent division by zero if a patent has a zero-magnitude embedding (which shouldn't happen, but defensive programming)."

### FEATURE 2: TF-IDF Cosine Similarity (Lines 98-112)

"Feature 2 captures traditional keyword-based similarity."

**Lines 99-101:** "Extract the abstract text from both patents."

**Lines 103-108:** "Create a TF-IDF vectorizer with these settings:
- max_features=1000: Limit vocabulary to top 1000 most informative terms
- stop_words='english': Remove 'the', 'is', 'and', etc.
- ngram_range=(1,2): Include both single words and two-word phrases
- min_df=1: Keep terms appearing in at least 1 document

TF-IDF stands for Term Frequency × Inverse Document Frequency. It weights words by how important they are:
- Common words (appearing in many patents) get low weight
- Rare technical terms get high weight

For example, 'the' might get weight 0.01, while 'piezoelectric resonator' gets weight 0.85."

**Line 109:** "Transform both abstracts into TF-IDF vectors. This creates a sparse vector where each dimension represents a word, and the value is its TF-IDF score."

**Line 110-111:** "Compute cosine similarity between these TF-IDF vectors."

**Why both semantic (Feature 1) AND keyword (Feature 2) similarity?** "They capture different things:
- Semantic: 'wireless power' and 'cordless energy' are similar (same meaning, different words)
- Keywords: Exact word matches

Two patents might use completely different terminology but describe the same concept (high semantic, low keyword). Or they might share technical terms but be about different applications (medium keyword, lower semantic). Having both gives the model more signal."

### FEATURE 3: Jaccard Similarity (Lines 115-123)

"Feature 3 is a simple word overlap metric."

**Line 117-118:** "Convert abstracts to sets of lowercase words."

**Line 120:** "Intersection: words that appear in BOTH abstracts."

**Line 121:** "Union: words that appear in EITHER abstract."

**Line 123:** "Jaccard = |intersection| / |union|

Example: If patent A has 100 unique words, B has 80 unique words, and they share 30 words, then:
- Intersection = 30
- Union = 100 + 80 - 30 = 150
- Jaccard = 30/150 = 0.20"

**Why this in addition to TF-IDF?** "Jaccard is cruder but sometimes catches overlaps that TF-IDF misses. It treats all words equally, so even common words count. This can be useful for patents with unusual vocabulary."

### FEATURE 4: Claim Count Ratio (Lines 126-132)

"Feature 4 captures structural similarity."

**Line 127-128:** "Count the number of claims in each patent."

**Line 130-132:** "Take the min/max ratio. If patent A has 15 claims and B has 12 claims:
- Ratio = min(15,12) / max(15,12) = 12/15 = 0.80

Why this metric? Patents of similar scope and complexity tend to have similar numbers of claims. A simple mechanical device might have 5 claims, while a complex pharmaceutical composition might have 50 claims. If two patents both have ~20 claims, they're likely in the same complexity class."

**Implementation note:** "Adding 1e-8 to denominator prevents division by zero if a patent has 0 claims (shouldn't happen but defensive coding)."

### FEATURE 5: Abstract Length Ratio (Lines 135-141)

"Feature 5 measures document verbosity similarity."

**Line 136-137:** "Count the number of words in each abstract using split()."

**Line 139-141:** "Again, min/max ratio. Similar to claim count, patents in the same domain tend to have similar abstract lengths:
- Mechanical engineering: ~120 words
- Software: ~150 words
- Biotechnology: ~250 words

If two patents have abstracts of 130 and 145 words, ratio = 130/145 = 0.90. If one has 120 and other has 300, ratio = 120/300 = 0.40."

### FEATURE 6: Year Difference (Normalized) (Lines 144-149)

"Feature 6 captures temporal proximity."

**Line 145-146:** "Extract the publication year from each patent."

**Line 148-149:** "Normalize year difference using: 1 / (1 + |year_a - year_b|)

This creates a decay function:
- Same year (diff=0): 1.0 / (1+0) = 1.0
- 1 year apart: 1.0 / (1+1) = 0.5
- 5 years apart: 1.0 / (1+5) = 0.167
- 10 years apart: 1.0 / (1+10) = 0.091

Why does year matter? Technology evolves in waves. Patents from 2015-2017 might all be about machine learning boom. Patents from same era are more likely to cite each other and be related."

### FEATURE 7: Assignee Match (Binary) (Lines 152-157)

"Feature 7 is a simple binary indicator of same company."

**Line 153-154:** "Normalize assignee names to lowercase and trim whitespace."

**Line 156-157:** "Check if they match AND are not empty string. Returns 1.0 if same company, 0.0 otherwise.

Why this matters? Companies file patents in clusters. If Apple files patents about iPhones, those patents will cite each other. Google's search patents cite other Google search patents. This is a strong signal."

### FEATURE 8: CPC Code Overlap (Jaccard) (Lines 160-167)

"Feature 8 measures patent classification overlap."

**Line 161-162:** "Convert CPC codes to sets. CPC (Cooperative Patent Classification) is a hierarchical taxonomy. For example:
- H04W = Wireless communication
- G06F = Computing
- A61K = Pharmaceutical compositions

Each patent has 1-10 CPC codes indicating its technical domain."

**Line 164-167:** "Compute Jaccard similarity on these sets. If patent A has codes [H04W, G06F, H04L] and B has codes [H04W, H04L, G06Q]:
- Intersection: {H04W, H04L} = 2 codes
- Union: {H04W, G06F, H04L, G06Q} = 4 codes
- Jaccard = 2/4 = 0.50

High overlap means same technical domain. Two wireless communication patents might share most codes. A wireless patent and a pharmaceutical patent would share zero codes."

### FEATURE 9: Max Claim Embedding Similarity (Lines 170-195) - MOST COMPLEX

"Feature 9 is the most computationally expensive but very powerful. It measures fine-grained claim-level similarity."

**Line 171-172:** "Get the first 10 claims from each patent. We limit to 10 because if we did all claims (could be 50+ each), that's 50×50=2500 comparisons. 10×10=100 is manageable."

**Line 174-178:** "Extract the text from each claim. Claims can be dictionaries with a 'text' key or plain strings - we handle both."

**Line 180-181:** "Embed ALL claims from both patents using PatentSBERTa. If patent A has 10 claims and B has 10 claims, we get:
- embs_a: (10, 768) array
- embs_b: (10, 768) array"

**Lines 183-190: THE KEY LOGIC**
"Now we compute pairwise cosine similarities between EVERY claim pair and take the maximum:

```
for each claim from patent A:
    for each claim from patent B:
        compute cosine similarity
        track the maximum
```

This gives us 10×10=100 similarity scores, and we keep the highest one."

**Why maximum instead of average?** "Two patents might have mostly different claims but ONE overlapping claim that's critical. For example:
- Patent A (wireless charger base): 10 claims about base station features
- Patent B (wireless charger receiver): 10 claims about receiver features
- But A.claim7 and B.claim8 both describe 'foreign object detection'
- That one pair might have 0.92 similarity while others are ~0.3
- Max = 0.92 captures this important overlap
- Average = 0.35 would hide it

This is why max is more informative for patent analysis."

**Implementation note:** "We cache claim embeddings at line 182 so if the same claim text appears in multiple patents, we don't re-embed it."

### FEATURE 10: Title Similarity (Lines 198-206)

"Feature 10 measures high-level conceptual similarity using patent titles."

**Line 199-200:** "Get the title from each patent (usually 5-20 words)."

**Line 202-203:** "Embed both titles using PatentSBERTa."

**Line 205-206:** "Compute cosine similarity between title embeddings.

Example:
- Title A: 'Wireless Power Transfer System Using Magnetic Resonance'
- Title B: 'System for Wireless Energy Transmission via Resonant Coupling'

Even though the words are different, the embeddings will be very similar (~0.88) because they describe the same concept. This captures semantic similarity at the title level."

**Line 209: RETURN**
"Finally, we return all 10 features as a numpy array [f1, f2, f3, ..., f10]. This is one feature vector for one patent pair."

### Summary of All 10 Features

"Let me summarize what each feature captures:

**Semantic Features:**
1. Embedding similarity - Overall semantic meaning
9. Max claim similarity - Fine-grained technical overlap
10. Title similarity - High-level concept match

**Lexical Features:**
2. TF-IDF - Important keyword overlap
3. Jaccard - Simple word overlap

**Structural Features:**
4. Claim count ratio - Patent scope similarity
5. Length ratio - Document complexity

**Temporal Feature:**
6. Year difference - Temporal proximity

**Categorical Features:**
7. Assignee match - Same company
8. CPC overlap - Same technical classification

Together, these 10 features give the neural network a rich, multidimensional view of patent similarity."

---

# STAGE 4B: RUNNING FEATURE EXTRACTION

## File: `scripts/data/preprocessing/compute_features.py`

**What to say:**

"Now that we've defined how to compute features, this script actually runs the computation on all 57,114 training pairs."

### Lines 24-30: `load_patents()`
"First we need to load all 200,000 patents into memory as a dictionary. This is about 3 GB of RAM but makes lookups instant."

### Lines 41-57: `load_embeddings()`
"Load the embedding matrix and patent IDs we generated earlier. We use memory-mapping (mmap_mode='r') so we don't load all 586MB into RAM at once."

### Lines 60-79: `compute_split_features()` - THE CORE LOOP

**Line 64:** "Loop through all pairs in this split (could be 40,000 train pairs or 8,500 val pairs)."

**Lines 65-67:** "Extract the two patent IDs and label from the pair dictionary."

**Lines 68-71:** "Look up the full patent data for both. If either patent is missing (shouldn't happen but defensive), skip this pair."

**Line 72: THE KEY LINE**
"`extractor.extract_features(p1, p2)` - This calls our FeatureExtractor class and computes all 10 features for this pair. Returns a feature vector."

**Line 73:** "Convert to numpy array format."

**Line 74:** "Accumulate the label (0 or 1)."

**Lines 77-78:** "After processing all pairs, stack into matrices:
- X: (N, 10) feature matrix
- y: (N,) label vector"

### Lines 92-115: `main()` - Run for All Splits

**Line 93-98:** "Load patents and embeddings, initialize the FeatureExtractor with them."

**Lines 103-110:** "For each split (train, val, test):
1. Load the pairs from JSONL (created in stage 3)
2. Compute features for all pairs
3. Print statistics

This gives us:
- train_features_v2.X.npy: (39979, 10)
- train_features_v2.y.npy: (39979,)
- val_features_v2.X.npy: (8567, 10)
- val_features_v2.y.npy: (8567,)
- test_features_v2.X.npy: (8568, 10)
- test_features_v2.y.npy: (8568,)"

### Timing
"Computing all 10 features for 57,114 pairs takes about 16-17 hours. Why so long?
- Features 1-8: Fast (milliseconds per pair)
- Feature 9 (claim embeddings): Slow (1-2 seconds per pair)
  - Must embed 10 claims × 2 patents = 20 embeddings per pair
  - PatentSBERTa inference: ~50ms per text
  - 20 × 50ms × 57,114 pairs = ~28 hours
  - With caching, we reduce this to ~16 hours
- Feature 10 (titles): Medium (~30ms per pair)

This is a one-time cost. Once computed, we have the feature matrices ready for training."

---

# STAGE 4C: HYPERPARAMETER TUNING

## File: `scripts/evaluation/tuning/nn_tuning.py`

**What to say:**

"Before training our final model, we need to find the best hyperparameters. This script does grid search with 3-fold cross-validation to find the optimal architecture and training settings."

### Lines 20-34: `load_features()`
"Load the feature matrices we just computed. We'll use train+val together for cross-validation, then evaluate on test."

### Lines 37-44: `build_param_grid()` - THE SEARCH SPACE

**What to say:**
"Here's what we're searching over:

**hidden_dims:** 
- [128, 64]: Smaller network (18K parameters)
- [128, 64, 32]: Deeper network (24K parameters)  
- [256, 128]: Wider network (118K parameters) ← Our final choice

**dropout:**
- 0.2: Less regularization
- 0.3: Medium regularization ← Our final choice
- 0.4: Strong regularization

**learning_rate:**
- 0.0005: Slower, more stable
- 0.001: Medium
- 0.002: Faster convergence ← Our final choice

**weight_decay:**
- 1e-5: Light L2 regularization ← Our final choice
- 1e-4: Strong L2 regularization

**batch_size:**
- 256: Fixed (based on preliminary experiments)

Total configurations: 3 × 3 × 3 × 2 = 54 configurations
With 3-fold CV, that's 54 × 3 = 162 model training runs."

### Lines 62-115: `main()` - GRID SEARCH LOOP

**Lines 64-67:** "Combine train and val for cross-validation. We have 48,546 pairs total."

**Line 73:** "Create StratifiedKFold with 3 splits. Stratified means each fold maintains the 50/50 positive/negative ratio."

**Lines 79-110: THE GRID SEARCH**

**Line 80:** "Unpack the hyperparameter configuration."

**Lines 83-100: For Each Fold**
"We split the 48,546 pairs into:
- Fold 1: Train on 32,364 pairs, validate on 16,182
- Fold 2: Train on 32,364 pairs (different split), validate on 16,182
- Fold 3: Train on 32,364 pairs (different split), validate on 16,182

For each fold:
1. Create PyTorchPatentClassifier with current hyperparameters (lines 87-97)
2. Train on fold's training set (line 98)
3. Evaluate on fold's validation set (line 99)
4. Record ROC-AUC score (line 100)"

**Lines 101-109:** "Average the 3 ROC-AUC scores. This is our cross-validation score for this configuration. Track the best."

**Why 3-fold instead of 5-fold or 10-fold?** "3-fold is a good compromise:
- More folds = better estimate but more computation
- Each config takes ~15 minutes × 3 folds = 45 minutes
- 54 configs × 45 minutes = 40.5 hours total
- We don't want to wait days"

### Lines 112-125: FINAL MODEL

**What to say:**
"After trying all 54 configurations, we found the best:
- hidden_dims: [256, 128]
- dropout: 0.3
- learning_rate: 0.002
- weight_decay: 1e-5
- Cross-validation ROC-AUC: 0.9717

Now we train the final model using these hyperparameters on the full train+val set and evaluate on the held-out test set."

**Result:** "Test ROC-AUC: 0.9720 - even better than cross-validation! This shows the model generalizes well."

---

# STAGE 4D: MODEL ARCHITECTURE & TRAINING

## File: `src/app/pytorch_classifier.py`

**What to say:**

"Now let me show you the actual neural network architecture and training implementation. This is a custom PyTorch model with residual connections, batch normalization, and multiple regularization techniques. I'll explain every design choice and why we made it."

### Lines 26-66: `ResidualBlock` Class - THE FUNDAMENTAL BUILDING BLOCK

**What to say:**

"The ResidualBlock is the core building block of our network, inspired by the ResNet architecture from Microsoft Research in 2015. Let me explain why residual connections are crucial and how they work."

### THE CONCEPT: Residual Learning

**Standard Neural Network Layer:**
"In a traditional neural network layer, we learn a function f(x) that transforms the input:

```
Input (x)
  ↓
Linear transformation: W×x + b
  ↓
Activation: ReLU
  ↓
Output = f(x)
```

The network learns the full transformation from input to output."

**Residual Block Architecture:**
"In a residual block, we learn the RESIDUAL (the difference) instead of the full transformation:

```
Input (x) ─────────────────────┐
  ↓                             │
Linear: W×x + b                 │
  ↓                             │ Skip Connection
BatchNorm                       │ (Identity)
  ↓                             │
ReLU                            │
  ↓                             │
Dropout                         │
  ↓                             │
Output = f(x) + x ←─────────────┘
```

The key insight: instead of learning f(x), we learn f(x) such that output = f(x) + x. The network learns the RESIDUAL or difference, not the full mapping."

### WHY RESIDUAL CONNECTIONS? The Vanishing Gradient Problem

**The Problem:**
"In deep neural networks without residual connections, gradients can vanish during backpropagation. Here's the math:

During backpropagation, gradients flow backward through the chain rule:

```
∂Loss/∂W₁ = (∂Loss/∂Layer₁₀) × (∂Layer₁₀/∂Layer₉) × (∂Layer₉/∂Layer₈) × ... × (∂Layer₂/∂Layer₁) × (∂Layer₁/∂W₁)
```

This is a product of 10 partial derivatives. If each partial derivative is 0.5 (common with sigmoid/tanh activations):

```
Gradient at Layer 1 = 0.5 × 0.5 × 0.5 × ... (9 times) = (0.5)⁹ = 0.00195 ≈ 0.002
```

The gradient has shrunk to 0.2% of its original value! This is the vanishing gradient problem. Early layers barely learn because their gradients are tiny."

**The Solution:**
"Residual connections provide an alternate gradient path. With skip connections:

```
∂(f(x) + x)/∂x = ∂f(x)/∂x + ∂x/∂x = ∂f(x)/∂x + 1
```

The '+1' term means there's ALWAYS a gradient of at least 1 flowing backward, even if ∂f(x)/∂x becomes very small. This is why we can train networks with 50, 100, or even 1000 layers with residual connections, but only 3-5 layers without them."

**In our network:**
"We only have 2 ResidualBlocks, so we don't strictly need them for gradient flow. But they provide other benefits:
1. **Faster convergence**: The network can learn identity mapping easily (set f(x)=0), then refine from there
2. **Better feature reuse**: Skip connections let the network preserve features from earlier layers
3. **Improved generalization**: Acts as a form of regularization"

### COMPONENT-BY-COMPONENT BREAKDOWN

**Component 1: Linear Layer (Lines 43)**

```python
self.fc = nn.Linear(in_features, out_features)
```

"This applies an affine transformation: y = Wx + b

Where:
- W is a weight matrix of shape (out_features, in_features)
- b is a bias vector of shape (out_features,)
- x is the input of shape (batch_size, in_features)
- y is the output of shape (batch_size, out_features)

Example with concrete numbers for our first ResidualBlock (10 → 256):
- Input x: (batch=256, 10 features)
- Weight W: (256 output_neurons, 10 input_features)
- Bias b: (256,)
- Computation: y = x @ W^T + b
- Output y: (batch=256, 256 features)

Number of parameters:
- Weights: 256 × 10 = 2,560 parameters
- Biases: 256 parameters
- Total: 2,816 parameters

For our second ResidualBlock (256 → 128):
- Weights: 128 × 256 = 32,768 parameters
- Biases: 128 parameters
- Total: 32,896 parameters"

**Component 2: Batch Normalization (Lines 44)**

```python
self.bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
```

"Batch normalization normalizes each feature across the batch to have mean=0 and std=1. Here's the exact algorithm:

For each feature dimension j (out of 256 in first block):

Step 1: Compute batch statistics
```
μ_j = (1/batch_size) × Σ x_ij  for i=1 to batch_size
σ²_j = (1/batch_size) × Σ (x_ij - μ_j)²  for i=1 to batch_size
```

Step 2: Normalize
```
x_normalized_ij = (x_ij - μ_j) / sqrt(σ²_j + ε)
```
where ε=1e-5 prevents division by zero

Step 3: Scale and shift (learnable transformation)
```
output_ij = γ_j × x_normalized_ij + β_j
```

where γ (gamma) and β (beta) are learnable parameters, one per feature.

WHY is this helpful?

Problem: Internal Covariate Shift
During training, the distribution of inputs to each layer changes as previous layers update:
- Epoch 1: Layer 2 sees inputs with mean=0.5, std=1.2
- Epoch 2: Layer 2 sees inputs with mean=1.3, std=0.8
- Epoch 3: Layer 2 sees inputs with mean=-0.2, std=2.1

This shifting distribution makes it hard for the layer to learn a stable transformation.

Solution: By forcing mean=0, std=1 after each layer, the distribution stays consistent:
- Every epoch: Layer 2 sees inputs with mean≈0, std≈1

Benefits:
1. **Faster training**: Can use learning rates 10-100x higher
2. **Regularization**: Batch statistics add noise, reducing overfitting (acts like implicit dropout)
3. **Reduced sensitivity to initialization**: Network less dependent on weight initialization
4. **Gradient flow**: Prevents activations from exploding or vanishing

The momentum parameter (0.1):
During training, BatchNorm tracks running statistics for inference:
```
running_mean_new = 0.1 × batch_mean + 0.9 × running_mean_old
running_std_new = 0.1 × batch_std + 0.9 × running_std_old
```

During inference (model.eval()), we use these running statistics instead of batch statistics, because we might process just one example (batch_size=1), making batch statistics meaningless.

Number of parameters:
- γ (scale): 256 learnable parameters
- β (shift): 256 learnable parameters
- running_mean: 256 tracked values (not trained)
- running_var: 256 tracked values (not trained)
- Total trainable: 512 parameters"

**Component 3: Dropout (Lines 45)**

```python
self.dropout = nn.Dropout(dropout)  # dropout=0.3
```

"Dropout randomly sets neurons to 0 with probability p during training.

With dropout=0.3 (30% drop rate):
```
Input: [0.5, 0.3, 0.8, 0.2, 0.6, 0.9, 0.1]

Random binary mask (30% zeros):
Mask:  [1,   0,   1,   0,   1,   1,   1]

After dropout:
Output: [0.5, 0.0, 0.8, 0.0, 0.6, 0.9, 0.1]

Scaled by 1/(1-0.3) = 1.43 to maintain expected value:
Final: [0.71, 0.0, 1.14, 0.0, 0.86, 1.29, 0.14]
```

WHY dropout?

Problem: Co-adaptation
Neurons become overly dependent on each other:
- Neuron A learns to detect 'wireless'
- Neuron B learns to detect 'charging'
- Neuron C learns the combination but becomes dependent on A and B always being active
- The network memorizes training examples using specific neuron combinations

If noise corrupts neuron A, the entire feature cascade breaks.

Solution: By randomly dropping neurons during training:
- Forces each neuron to work independently
- Creates redundant, robust representations
- Prevents memorization of specific patterns
- Acts as ensemble learning (averaging many sub-networks)

During training vs inference:
- Training (model.train()): Dropout is active, randomly zeros 30% of neurons
- Inference (model.eval()): Dropout is disabled, all neurons are active

The scaling factor 1/(1-p) = 1/0.7 = 1.43 ensures the expected sum of activations remains constant between training and inference.

Why dropout=0.3?
- Too low (0.1): Minimal regularization, risk of overfitting
- Too high (0.7): Too much information loss, underfitting
- 0.3-0.5: Sweet spot for most networks
- We chose 0.3 after hyperparameter tuning (tried 0.2, 0.3, 0.4)"

**Component 4: ReLU Activation (Lines 46)**

```python
self.activation = nn.ReLU()
```

"ReLU (Rectified Linear Unit) is the activation function:

```
ReLU(x) = max(0, x) = { x  if x > 0
                       { 0  if x ≤ 0
```

Example:
```
Input:  [-2.5, -0.1, 0.0, 0.3, 1.8]
Output: [0.0,  0.0,  0.0, 0.3, 1.8]
```

WHY ReLU?

Comparison to other activations:

**Sigmoid: f(x) = 1 / (1 + e^(-x))**
- Output range: (0, 1)
- Problem: Vanishing gradients for |x| > 3
- Gradient at x=5: ~0.0066 (tiny!)
- Used for: Output layer in binary classification

**Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))**
- Output range: (-1, 1)
- Problem: Still suffers from vanishing gradients
- Gradient at x=3: ~0.01 (small)
- Better than sigmoid but still problematic

**ReLU: f(x) = max(0, x)**
- Output range: [0, ∞)
- Gradient: 1 if x > 0, else 0
- No vanishing gradient problem for positive values!
- Fast to compute (simple comparison)
- Biologically inspired (neuron firing)

Advantages:
1. **No vanishing gradients** (for x > 0): gradient is exactly 1
2. **Sparse activation**: ~50% of neurons are zero (efficient)
3. **Computational efficiency**: Just a comparison, no exponentials
4. **Better gradients**: Gradient doesn't saturate like sigmoid/tanh

Disadvantage: Dying ReLU
If a neuron's weights shift such that it always outputs negative values, gradient becomes 0, and it never updates again (neuron 'dies'). This is why we use:
- Proper initialization (Xavier)
- Lower learning rates
- Batch normalization

We considered Leaky ReLU (f(x) = max(0.01x, x)) to prevent dying ReLU, but standard ReLU worked fine for our relatively shallow network."

**Component 5: Skip Connection (Lines 49-54)**

```python
if in_features != out_features:
    self.skip = nn.Linear(in_features, out_features)
    self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
else:
    self.skip = nn.Identity()
    self.skip_bn = None
```

"The skip connection must match the dimensions of the main path to enable addition.

Case 1: Dimensions match (in_features == out_features)
Example: 128 → 128
- Skip connection: Identity (no transformation)
- Skip output = input (just pass through)
- Final output = main_path_output + input

Case 2: Dimensions don't match (in_features ≠ out_features)
Example: 10 → 256
- Cannot add (10,) tensor to (256,) tensor!
- Solution: Project skip connection with a linear layer
- Skip: 10 → 256 via learned transformation
- Skip output = W_skip × input + b_skip
- Apply BatchNorm to skip path for consistency
- Final output = main_path_output + skip_path_output

Why BatchNorm on skip path?
Both paths should have similar scale for stable addition. If main path is normalized to mean=0, std=1, but skip path has mean=10, std=50, the addition would be dominated by the skip path, defeating the purpose of learning residuals.

Parameter count for skip projection (10 → 256):
- Weights: 10 × 256 = 2,560 parameters
- Bias: 256 parameters
- BatchNorm (γ, β): 512 parameters
- Total: 3,328 parameters"

**Component 6: Forward Pass (Lines 56-66)**

```python
def forward(self, x):
    identity = self.skip(x)
    if self.skip_bn is not None:
        identity = self.skip_bn(identity)
    
    out = self.fc(x)
    out = self.bn(out)
    out = self.activation(out)
    out = self.dropout(out)
    out = out + identity  # THE KEY LINE
    return out
```

"Let me trace through a forward pass with concrete numbers.

Example: First ResidualBlock (10 → 256), batch_size=256

Step 1: Compute skip connection
```
Input x: (256, 10)
Skip linear: x @ W_skip^T + b_skip
Skip output: (256, 256)
Skip BatchNorm: normalize to mean=0, std=1
identity: (256, 256)
```

Step 2: Main path - Linear transformation
```
Input x: (256, 10)
Main linear: x @ W_main^T + b_main
Main output: (256, 256)
```

Step 3: Main path - Batch Normalization
```
For each of 256 features:
  Compute batch mean and std across 256 examples
  Normalize: (value - mean) / std
  Scale and shift: γ × normalized + β
Output: (256, 256) with mean≈0, std≈1
```

Step 4: Main path - ReLU activation
```
For each value: max(0, value)
Roughly 50% of values become 0 (negative values zeroed)
Output: (256, 256) with only positive values
```

Step 5: Main path - Dropout
```
Randomly zero 30% of values
Scale remaining by 1.43
Output: (256, 256) with 30% zeros
```

Step 6: Add skip connection (THE RESIDUAL)
```
out = main_path_output + identity
out: (256, 256)
```

This final addition is the magic of residual learning. The network learns f(x) such that:
output = f(x) + x

If the optimal transformation is close to identity, the network just learns f(x) ≈ 0. This is much easier than learning the full transformation from scratch."

### Lines 69-139: `PatentNoveltyNet` Class - COMPLETE NETWORK ARCHITECTURE

**What to say:**

"This is our complete neural network that takes 10 engineered features and outputs a probability that two patents are similar. Let me walk through the architecture design and explain every decision."

### ARCHITECTURE OVERVIEW - THE COMPLETE PIPELINE

"Here's the full transformation pipeline:

```
Input: 10 features (embedding_sim, tfidf_sim, jaccard_sim, ...)
    ↓
InputBatchNorm: Normalize to mean=0, std=1 per feature
    ↓
ResidualBlock 1: 10 → 256 neurons (EXPAND)
    ├─ Main: Linear(10→256) → BatchNorm → ReLU → Dropout(0.3)
    └─ Skip: Linear(10→256) → BatchNorm
    ↓ Add paths
    256-dim feature representation
    ↓
ResidualBlock 2: 256 → 128 neurons (COMPRESS)
    ├─ Main: Linear(256→128) → BatchNorm → ReLU → Dropout(0.3)
    └─ Skip: Linear(256→128) → BatchNorm
    ↓ Add paths
    128-dim feature representation
    ↓
OutputBatchNorm: Normalize 128 features
    ↓
Linear: 128 → 1 (final decision neuron)
    ↓
Sigmoid: squash to [0, 1] probability
    ↓
Output: P(patents are similar)
```"

### DESIGN DECISIONS - WHY THIS ARCHITECTURE?

**Decision 1: Why 10 → 256 → 128 → 1? (The "Bowtie" Architecture)**

"We use an expansion-then-compression strategy:

Input: 10 features (limited, hand-crafted)
↓
Expand to 256: Create a high-dimensional representation where the model can learn complex feature interactions

Why 256 specifically?
- Too small (32, 64): Limited capacity to learn interactions. With only 10 input features, we need to expand significantly to capture non-linear combinations.
- Too large (512, 1024): Overfitting risk. We only have 40K training examples.
- 256: Sweet spot found via hyperparameter tuning (we tried [128,64], [256,128], [512,256])

The 256-dimensional space lets the model learn patterns like:
- IF (embedding_sim > 0.8 AND year_diff < 2 AND assignee_match) THEN highly similar
- IF (tfidf_sim > 0.7 AND cpc_overlap > 0.5) THEN domain_related
- Etc. - 256 neurons can encode many such rules

↓
Compress to 128: Extract the most important learned features, discard redundancy

Why 128 specifically?
- Model learns to identify the 128 most discriminative patterns from the 256 candidates
- Acts as dimensionality reduction and feature selection
- Prevents the final layer from being overwhelmed

↓
Compress to 1: Final similarity score

This architecture allows the model to:
1. Expand the feature space (10 → 256) to capture complex interactions
2. Learn hierarchical representations (256 → 128) to extract important patterns
3. Make a final decision (128 → 1) based on learned features"

**Decision 2: Why Residual Blocks instead of plain Linear layers?**

"We compared two architectures during development:

**Option A: Plain feedforward (no residual connections)**
```
10 → Linear → BatchNorm → ReLU → Dropout → 
256 → Linear → BatchNorm → ReLU → Dropout → 
128 → Linear → Sigmoid → 1
```
Validation ROC-AUC: 0.9645

**Option B: Residual blocks (our choice)**
```
10 → InputBN →
ResBlock(10→256) → 
ResBlock(256→128) → 
OutputBN → Linear → Sigmoid → 1
```
Validation ROC-AUC: 0.9717

Residual connections provided:
- 0.72% improvement in ROC-AUC
- Faster convergence (25 epochs vs 35 epochs)
- More stable training (less fluctuation in validation loss)

Even though our network is shallow (only 2 hidden layers), residual connections help because:
1. Easier optimization landscape (can learn identity if needed)
2. Better gradient flow during backpropagation
3. Feature reuse across layers"

**Decision 3: Why Batch Normalization at input AND between layers?**

"We use BatchNorm in 3 places:

1. **Input BatchNorm (Line 90)**: Even though we already apply StandardScaler to features, InputBN provides additional benefits:
   - Adapts to batch statistics during training
   - Adds regularization via batch noise
   - Learnable scale (γ) and shift (β) parameters let the network adjust feature importance

2. **Inside ResidualBlocks**: Standard practice for stabilizing training

3. **Output BatchNorm (Line 106)**: Normalizes the 128-dim representation before the final linear layer
   - Ensures final layer receives normalized inputs
   - Improves final classification boundary

We experimented with removing Input BN and Output BN:
- Without Input BN: 0.9651 ROC-AUC (-0.66%)
- Without Output BN: 0.9689 ROC-AUC (-0.28%)
- With both: 0.9717 ROC-AUC ✓"

**Decision 4: Why Sigmoid at the end?**

"The sigmoid function squashes the output to [0, 1]:

```
σ(x) = 1 / (1 + e^(-x))

Examples:
x = -5 → σ(-5) = 0.0067 ≈ 0.01 (very dissimilar)
x = -2 → σ(-2) = 0.119 (dissimilar)
x = 0 → σ(0) = 0.5 (uncertain)
x = 2 → σ(2) = 0.881 (similar)
x = 5 → σ(5) = 0.993 ≈ 0.99 (very similar)
```

This makes the output interpretable as a probability:
- Output = 0.95 → 95% confident patents are similar
- Output = 0.15 → 15% chance similar (85% chance different)

Alternative would be to output logits (no sigmoid) and threshold at 0 instead of 0.5, but probability interpretation is more intuitive for users and integrates naturally with the Binary Cross-Entropy loss."

### CODE WALKTHROUGH - Lines 87-111: Layer Construction

**Lines 87-90: __init__ Method and Input BatchNorm**

```python
def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3, 
             use_residual=True, bn_momentum=0.1):
    super(PatentNoveltyNet, self).__init__()
    self.input_bn = nn.BatchNorm1d(input_dim, momentum=bn_momentum)
```

"First, we create Input Batch Normalization for the 10 input features.

Parameters:
- input_dim=10 (our 10 engineered features)
- momentum=0.1 (for running statistics)

This adds:
- 10 scale parameters (γ)
- 10 shift parameters (β)
- Total: 20 learnable parameters"

**Lines 92-103: Building Residual Blocks**

```python
layers = []
prev_dim = input_dim  # Start at 10

for hidden_dim in hidden_dims:  # [256, 128]
    if use_residual:
        layers.append(ResidualBlock(prev_dim, hidden_dim, dropout, bn_momentum))
    else:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    prev_dim = hidden_dim

self.hidden_layers = nn.Sequential(*layers)
```

"We build the network dynamically using a loop. This lets us easily experiment with different architectures.

Iteration 1:
- prev_dim = 10
- hidden_dim = 256
- Create ResidualBlock(10 → 256)
- Update prev_dim = 256

Iteration 2:
- prev_dim = 256
- hidden_dim = 128
- Create ResidualBlock(256 → 128)
- Update prev_dim = 128

Final: Sequential container with [ResBlock1, ResBlock2]

The use_residual flag lets us ablate (turn off) residual connections for comparison. We found use_residual=True performs better."

**Lines 105-108: Output Layers**

```python
self.output_bn = nn.BatchNorm1d(prev_dim, momentum=bn_momentum)
self.output_layer = nn.Linear(prev_dim, 1)
self.sigmoid = nn.Sigmoid()
```

"After the ResidualBlocks, we have 128-dimensional features.

Output BatchNorm:
- Normalizes the 128 features
- Adds 128 × 2 = 256 parameters (γ, β)

Output Linear:
- Projects 128 features → 1 output neuron
- Parameters: 128 weights + 1 bias = 129 parameters

Sigmoid:
- Squashes output to [0, 1] probability
- No parameters (just applies formula)"

**Lines 113-119: Xavier Initialization**

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

"Xavier (Glorot) initialization is crucial for deep networks. Here's why and how it works:

THE PROBLEM: Poor initialization can cause exploding or vanishing activations.

With random initialization from N(0, 0.01):
```
Layer 1: Input variance = 1.0
         After W×x: Variance = 0.01² × 10 = 0.001
         Activation variance shrinks to 0.001 (VANISHING)

Layer 2: Input variance = 0.001  
         After W×x: Variance = 0.001 × 0.01² × 256 = 0.0000026
         Activation variance → 0 (DEAD NETWORK)
```

With random initialization from N(0, 1):
```
Layer 1: Input variance = 1.0
         After W×x: Variance = 1² × 10 = 10
         Activation variance = 10 (EXPLODING)

Layer 2: Input variance = 10
         After W×x: Variance = 10 × 1² × 256 = 2560  
         Activation variance → ∞ (EXPLODING GRADIENTS)
```

THE SOLUTION: Xavier initialization maintains variance across layers.

Xavier uniform formula:
```
limit = sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limit, +limit)
```

Where fan_in = input neurons, fan_out = output neurons

For our first ResidualBlock (10 → 256):
```
limit = sqrt(6 / (10 + 256))
      = sqrt(6 / 266)
      = sqrt(0.0226)
      = 0.150

Weights initialized uniformly in [-0.150, +0.150]
```

For our second ResidualBlock (256 → 128):
```
limit = sqrt(6 / (256 + 128))
      = sqrt(6 / 384)
      = sqrt(0.0156)
      = 0.125

Weights initialized uniformly in [-0.125, +0.125]
```

This ensures:
- Input variance ≈ Output variance
- Activations stay in reasonable range
- Gradients neither explode nor vanish

We initialize biases to 0 (standard practice)."

**Lines 121-139: Forward Pass - THE COMPLETE DATA FLOW**

```python
def forward(self, x):
    x = self.input_bn(x)
    x = self.hidden_layers(x)
    x = self.output_bn(x)
    x = self.output_layer(x)
    x = self.sigmoid(x)
    return x
```

"Let me trace a complete forward pass with a batch of 256 patent pairs:

**Step 1: Input (256, 10)**
```
x = [[0.85, 0.62, 0.41, ...],  # Pair 1: features for patent A vs B
     [0.23, 0.15, 0.88, ...],  # Pair 2: features for patent C vs D
     ...
     [0.67, 0.71, 0.52, ...]]  # Pair 256
Shape: (256 examples, 10 features)
```

**Step 2: Input Batch Normalization (256, 10)**
```
For each of 10 features across 256 examples:
  Compute mean and std
  Normalize: (x - mean) / std
  Scale and shift: γ × normalized + β

Example for Feature 1 (embedding similarity):
  mean = 0.45, std = 0.23
  Normalized Feature 1: (Feature 1 - 0.45) / 0.23
  Scaled: γ₁ × normalized + β₁

Output: (256, 10) with mean≈0, std≈1 for each feature
```

**Step 3: ResidualBlock 1 (10 → 256)**
```
Input: (256, 10)

Main path:
  Linear: (256, 10) @ (256, 10)^T = (256, 256)
  BatchNorm: normalize 256 features
  ReLU: zero out negative values
  Dropout: randomly zero 30% of values

Skip path:
  Linear: (256, 10) @ (256, 10)^T = (256, 256)
  BatchNorm: normalize

Add: main + skip = (256, 256)

Output: (256, 256)
We've expanded from 10 features to 256-dimensional representation
```

**Step 4: ResidualBlock 2 (256 → 128)**
```
Input: (256, 256)

Main path:
  Linear: (256, 256) @ (128, 256)^T = (256, 128)
  BatchNorm: normalize 128 features
  ReLU: activation
  Dropout: regularization

Skip path:
  Linear: (256, 256) @ (128, 256)^T = (256, 128)
  BatchNorm: normalize

Add: main + skip = (256, 128)

Output: (256, 128)
We've compressed to 128-dimensional representation
```

**Step 5: Output Batch Normalization (256, 128)**
```
Normalize the 128 features before final layer
Output: (256, 128) normalized
```

**Step 6: Output Linear (256, 1)**
```
Linear: (256, 128) @ (1, 128)^T = (256, 1)
Each of 256 examples now has a single logit value

Example logits:
[[ 2.34],   # Pair 1: positive logit (likely similar)
 [-1.87],   # Pair 2: negative logit (likely different)
 [ 0.45],   # Pair 3: close to 0 (uncertain)
 ...
 [ 3.12]]   # Pair 256: strong positive (very similar)
```

**Step 7: Sigmoid (256, 1)**
```
Apply sigmoid to each logit:
σ(x) = 1 / (1 + e^(-x))

[[0.912],   # σ(2.34) = 0.912  (91.2% similar)
 [0.133],   # σ(-1.87) = 0.133 (13.3% similar)
 [0.611],   # σ(0.45) = 0.611  (61.1% similar)
 ...
 [0.978]]   # σ(3.12) = 0.978  (97.8% similar)

Final output: (256, 1) probabilities
```

Each example now has a probability in [0,1] representing how likely the patent pair is to be similar."

### PARAMETER COUNT - DETAILED BREAKDOWN

"Let me calculate the exact number of parameters:

**Input BatchNorm:**
- γ (scale): 10 parameters
- β (shift): 10 parameters
- Subtotal: 20 parameters

**ResidualBlock 1 (10 → 256):**

Main path:
- Linear: 10×256 weights + 256 biases = 2,816
- BatchNorm: 256 γ + 256 β = 512
- Dropout: 0 (no parameters)
- ReLU: 0 (no parameters)
Main path subtotal: 3,328

Skip path:
- Linear: 10×256 weights + 256 biases = 2,816
- BatchNorm: 256 γ + 256 β = 512
Skip path subtotal: 3,328

ResBlock 1 total: 3,328 + 3,328 = 6,656 parameters

**ResidualBlock 2 (256 → 128):**

Main path:
- Linear: 256×128 weights + 128 biases = 33,024
- BatchNorm: 128 γ + 128 β = 256
Main path subtotal: 33,280

Skip path:
- Linear: 256×128 weights + 128 biases = 33,024
- BatchNorm: 128 γ + 128 β = 256
Skip path subtotal: 33,280

ResBlock 2 total: 33,280 + 33,280 = 66,560 parameters

**Output BatchNorm:**
- γ (scale): 128 parameters
- β (shift): 128 parameters
- Subtotal: 256 parameters

**Output Linear:**
- Weights: 128 × 1 = 128 parameters
- Bias: 1 parameter
- Subtotal: 129 parameters

**GRAND TOTAL:**
20 + 6,656 + 66,560 + 256 + 129 = 73,621 parameters

Wait, let me recalculate more carefully with the actual architecture...

Actually, looking at the code, each ResidualBlock has:
- Main path: fc + bn + dropout + activation
- Skip path: skip + skip_bn (if dimensions differ)

For ResBlock(10→256) with dimension mismatch:
- Main fc: 10×256 + 256 = 2,816
- Main bn: 256×2 = 512
- Skip fc: 10×256 + 256 = 2,816
- Skip bn: 256×2 = 512
Total: 6,656 parameters

For ResBlock(256→128) with dimension mismatch:
- Main fc: 256×128 + 128 = 33,024
- Main bn: 128×2 = 256
- Skip fc: 256×128 + 128 = 33,024
- Skip bn: 128×2 = 256
Total: 66,560 parameters

Full network:
- Input BN: 10×2 = 20
- ResBlock1: 6,656
- ResBlock2: 66,560  
- Output BN: 128×2 = 256
- Output Linear: 128 + 1 = 129
**TOTAL: 73,621 parameters**

At 4 bytes per float32 parameter:
73,621 × 4 bytes = 294,484 bytes = 288 KB

This is a very lightweight model! The entire network fits in less than 300KB of memory. For comparison:
- BERT-base: 110 million parameters (440 MB)
- GPT-2: 117 million parameters (468 MB)  
- Our model: 74 thousand parameters (0.29 MB)

We can train this on a laptop GPU with no memory issues."

### Lines 239-379: `fit()` Method - TRAINING LOOP

**What to say:**

"Now let me walk through the complete training procedure with all the techniques we use."

### STEP 1: Feature Normalization (Lines 264-265)

**Line 264:** "`X_train_scaled = self.scaler.fit_transform(X_train)`

StandardScaler transforms each feature to mean=0, std=1:

For each feature column j:
    mean_j = mean(X_train[:, j])
    std_j = std(X_train[:, j])
    X_scaled[:, j] = (X_train[:, j] - mean_j) / std_j

Example:
Feature 1 (cosine sim): mean=0.24, std=0.18 → After scaling: mean=0.0, std=1.0

Why? Gradient descent converges faster when all features are on the same scale."

**Line 265:** "For validation, we use transform (not fit_transform) - we apply the TRAINING statistics. This prevents data leakage."

### STEP 2: Model Creation (Lines 268-275)

"Create the PatentNoveltyNet with our hyperparameters and move it to device (MPS/CUDA/CPU)."

### STEP 3: Loss Function and Optimizer (Lines 278-287)

**Line 278:** "Binary Cross-Entropy loss:

BCE(y, ŷ) = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

For a similar pair (y=1) predicted with ŷ=0.9:
BCE = -[1 × log(0.9) + 0 × log(0.1)] = -log(0.9) = 0.105 (small loss ✓)

For a similar pair (y=1) predicted with ŷ=0.1:
BCE = -[1 × log(0.1) + 0 × log(0.9)] = -log(0.1) = 2.303 (large loss ✗)

The logarithm creates strong penalties for confident wrong predictions."

**Lines 279-283: AdamW Optimizer**
"AdamW combines:
- Adaptive learning rates (different rate for each parameter)
- Momentum (exponential moving average of gradients)
- Proper weight decay (L2 regularization)

Update rule (simplified):
m_t = 0.9 × m_{t-1} + 0.1 × gradient  (momentum)
v_t = 0.999 × v_{t-1} + 0.001 × gradient²  (variance)
W_t = W_{t-1} - lr × m_t / sqrt(v_t)  (adaptive step)
W_t = W_t × (1 - weight_decay)  (decay toward zero)

learning_rate=0.002: Initial step size
weight_decay=1e-5: Encourages smaller weights"

**Lines 285-287: Learning Rate Scheduler**
"ReduceLROnPlateau monitors validation loss. If it doesn't improve for 5 epochs, reduce learning rate by 50%:

Initial LR: 0.002
After plateau 1: 0.001
After plateau 2: 0.0005

This lets us start with large steps for fast progress, then take smaller steps for fine-tuning."

### STEP 4: Training Loop (Lines 300-371)

**Lines 302-325: Training Phase for Each Batch**

**Line 302:** "Set model to training mode - this enables dropout and uses batch statistics for BatchNorm."

**Lines 306-308:** "For each batch of 256 pairs, move data to GPU (MPS in my case)."

**Lines 311-315: Mixup Augmentation**
"With 50% probability, we apply mixup - a data augmentation technique:

Take two random examples from the batch:
    Example i: (features_i, label_i)
    Example j: (features_j, label_j)

Sample mixing coefficient from Beta distribution:
    λ ~ Beta(0.2, 0.2)  # Concentrates near 0 and 1

Create synthetic example:
    features_new = λ × features_i + (1-λ) × features_j
    label_new = λ × label_i + (1-λ) × label_j

Example:
    i: features=[0.8, 0.3, ...], label=1 (similar)
    j: features=[0.2, 0.7, ...], label=0 (different)
    λ=0.6
    new: features=[0.56, 0.48, ...], label=0.6 (partially similar)

This creates infinite training examples and smooths decision boundaries, improving generalization."

**Line 317:** "Zero gradients - PyTorch accumulates gradients, so we reset before each batch."

**Line 318:** "Forward pass - run batch through network, get predictions."

**Line 319:** "Compute BCE loss between predictions and labels."

**Line 320:** "Backward pass - compute gradients using backpropagation through all layers."

**Line 321: Gradient Clipping**
"Clip gradient norm to maximum of 1.0:

total_norm = sqrt(sum of all gradient²)
if total_norm > 1.0:
    scale = 1.0 / total_norm
    multiply all gradients by scale

This prevents exploding gradients which could destabilize training."

**Line 323:** "Optimizer step - update all weights using computed gradients."

### Lines 330-356: Validation Phase

**Line 331:** "Set model to eval mode - disables dropout, uses running statistics for BatchNorm."

**Line 336:** "`with torch.no_grad()` - don't track gradients during validation (saves memory and speeds up)."

**Lines 337-346:** "Run validation batches, collect predictions and labels."

**Lines 349-352:** "Compute validation accuracy by thresholding at 0.5:
predicted_class = 1 if probability > 0.5 else 0
accuracy = (predictions == labels).mean()"

### Lines 357-371: Learning Rate Scheduling and Early Stopping

**Line 357:** "Update learning rate if validation loss plateaued."

**Lines 359-371: Early Stopping Logic**
"Track the best validation loss seen so far. If current validation loss is better:
- Save model state
- Reset patience counter

If current validation loss is NOT better:
- Increment patience counter
- If patience reaches 15 epochs without improvement: STOP

This prevents overfitting - we stop training when the model stops improving on unseen data."

**Lines 374-375:** "Load the best model state (from epoch with lowest validation loss)."

### Regularization Techniques Summary

"We use FIVE different regularization techniques to prevent overfitting:

1. **Dropout (0.3)**: Randomly zero 30% of neurons each batch
2. **Batch Normalization**: Normalizes activations, adds noise via batch statistics
3. **Weight Decay (1e-5)**: L2 penalty on weights
4. **Early Stopping (patience=15)**: Stop when validation loss plateaus
5. **Mixup (α=0.2)**: Data augmentation via linear interpolation

Together, these allow us to train a model with 118K parameters on only 40K examples without overfitting."

### Training Results

"After training:
- Converges in ~42 epochs (early stopping triggers)
- Training time: ~35 seconds/epoch × 42 = 25 minutes
- Final metrics on test set:
  - Accuracy: 91.73%
  - Precision: 92.87%
  - Recall: 90.33%
  - F1: 91.59%
  - ROC-AUC: 97.20% ← Primary metric

97.2% ROC-AUC means if we pick one similar pair and one dissimilar pair, the model ranks them correctly 97.2% of the time."

---

# STAGE 5: INFERENCE PIPELINE

## Overview Narrative

"Now we have a trained model. When a user submits a new patent for novelty assessment, we need to:
1. Embed their patent
2. Search for similar patents (both locally and online)
3. Extract features for each candidate
4. Score with our trained model
5. Generate an explanation

This all happens in the PatentAnalyzer class."

## File: `src/app/patent_analyzer.py`

**What to say:**

"This is the main orchestrator that coordinates the entire inference pipeline."

### Lines 120-193: `load()` Method - LOADING ALL COMPONENTS

**What to say while scrolling:**

**Lines 133-137:** "Load the 200K embeddings using memory-mapping - this means we map the 586MB file to virtual memory and only load pages as needed. Saves RAM."

**Lines 139-143:** "Load PatentSBERTa model - same model we used to generate embeddings. We'll use this to embed the user's patent and compute claim/title similarities."

**Lines 145-149:** "Initialize Phi-3 explainer - connects to Ollama running locally on port 11434."

**Lines 151-155:** "Initialize LLM keyword extractor - uses Phi-3 to generate smart search terms."

**Lines 157-166:** "Initialize Google Patents searcher - if SerpAPI key is provided, this enables online search of millions of patents."

**Lines 168-186:** "Load our trained PyTorch model and FeatureExtractor:
- pytorch_model.pt: The 118K parameter model we trained
- scaler_pytorch.pkl: The StandardScaler fit on training data
- feature_names_v2.json: Names of the 10 features"

**Why lazy loading?** "We don't load everything at initialization. We wait until load() is called. This means the Streamlit app starts in 2 seconds instead of 30 seconds. Components only load when first needed."

### Lines 400-650: `analyze()` Method - THE COMPLETE PIPELINE

**What to say:**

"This method orchestrates the 9-step inference pipeline. Let me walk through each step."

### STEP 1: Generate Query Embedding (Lines 410-412)

**Line 411:** "`query_embedding = self.st_model.encode(query_text)`

The user's patent text (could be just a description or full abstract) goes through PatentSBERTa:
- Tokenize into subwords
- Add [CLS] and [SEP] tokens
- Pass through 12 transformer layers
- Extract [CLS] representation
- Returns: (768,) numpy array

This takes 2-3 seconds."

### STEP 2: Local Search (Lines 415-420)

**What to say:**

"We search our local database of 200K patents using cosine similarity. Let me show you how."

### Lines 633-662: `_find_similar()` Method

**Lines 636-638: Normalization**
"Normalize both query and database embeddings to unit length:

query_norm = query / ||query||
database_norms = embeddings / ||embeddings along axis 1||

After normalization, dot product = cosine similarity:
cos(θ) = (A·B) / (||A|| × ||B||) = A_norm · B_norm"

**Line 640: THE FAST SEARCH**
"`similarities = np.dot(database_norms, query_norm)`

This is matrix-vector multiplication:
- database_norms: (200000, 768) matrix
- query_norm: (768,) vector
- Result: (200000,) vector of similarities

NumPy uses highly optimized BLAS libraries (OpenBLAS/Intel MKL/Apple Accelerate) with SIMD instructions. This computes 200,000 cosine similarities in ~1 second!

For comparison, a Python loop would take minutes."

**Line 642:** "Get top 15 patent indices using argsort:

similarities = [0.23, 0.87, 0.15, 0.91, ...]  (200K values)
sorted_indices = argsort(similarities)  # [2, 0, 1, 3, ...]  (ascending)
reversed = sorted_indices[::-1]  # [3, 1, 0, 2, ...]  (descending)
top_15 = reversed[:15]  # Keep first 15

This gives us indices of most similar patents."

**Lines 645-660:** "Load full patent data for these 15 indices, return as list of dicts with patent_id, title, abstract, similarity score, etc."

**Result:** "15 local results in ~1-2 seconds total."

### STEP 3: LLM Keyword Extraction (Lines 425-435)

**What to say:**

"Now we use Phi-3 to generate intelligent search terms for Google Patents. This is implemented in online_search.py."

## Brief Detour: `data/api/online_search.py` - LLM Keyword Generation

### Lines 81-148: `generate_search_terms()` Method

**Lines 88-109: The Prompt**
"We prompt Phi-3:

'As a search specialist, generate 5 effective Google Patents queries.
Rules:
- Use 2-5 technical terms
- Can use AND/OR operators
- Keep queries focused

INPUT: [user's patent description]

OUTPUT: Numbered list of queries'

Example output:
1. wireless power transfer AND magnetic resonance
2. inductive charging system OR resonant coupling
3. foreign object detection wireless charging
4. multi-coil transmitter
5. NFC power transfer"

**Lines 112-121:** "POST to Ollama API at localhost:11434 with temperature=0.3 (fairly focused, not too creative)."

**Lines 127-138: Parse Response**
"Parse the numbered list from Phi-3's response using regex:
- Match patterns like '1. query' or '1) query'
- Extract the query text
- Clean up excessive parentheses

Return list of 5 search terms."

**Why LLM-generated keywords?** "They're smarter than simple keyword extraction:
- Understands synonyms ('wireless' = 'cordless')
- Knows technical variants ('inductive charging' = 'resonant coupling')
- Generates proper boolean queries for Google Patents
- Considers different aspects of the invention"

**Time:** "10-15 seconds for Phi-3 to generate 5 queries."

### STEP 4: Online Search (Lines 440-450)

### Back to `data/api/online_search.py` - Lines 401-477: `_search_serpapi()`

**Lines 420-425: Build API Request**
"Create SerpAPI parameters:
```python
params = {
    'engine': 'google_patents',  # Use Google Patents engine
    'q': query,  # The search query
    'api_key': self.serpapi_key,  # Your API key
    'num': num_results  # How many results (10-100)
}
```"

**Lines 428-429:** "Make the API call:
```python
search = GoogleSearch(params)
results = search.get_dict()
```

SerpAPI handles:
- Making HTTP request to Google Patents
- Parsing HTML results
- Structuring as JSON
- Handling pagination, rate limits, etc."

**Lines 437-468: Parse Results**
"Extract from each result:
- patent_id (publication number)
- title
- abstract (or snippet)
- year (parse from publication date)
- URL to Google Patents page
- inventor and assignee if available

Create PatentSearchResult objects."

**Lines 369-399: `search_multiple_terms()`**
"Search each of the 5 terms:
```python
for term in terms:
    results = self.search(term, max_per_term=10)
    accumulate results
    deduplicate by patent ID
```

With 5 terms × 10 results each = up to 50 patents (usually 40-45 unique after deduplication)."

**Time:** "15-25 seconds (network latency dominates)."

### STEP 5: Merge Results (Lines 455-460 in patent_analyzer.py)

**What to say:**

"Combine local (15 patents) and online (45 patents) results. Deduplicate by patent ID.

Example:
- Local found: US11234567, US10987654, ...
- Online found: US11234567 (duplicate!), WO2023456789, ...
- After dedup: ~58 unique candidates

Result: 50-70 total unique candidate patents from both sources combined."

### STEP 6: Feature Extraction (Lines 465-475)

**What to say:**

"For each candidate patent, extract 10 features comparing it to the user's patent:

```python
for candidate in all_candidates:
    features = feature_extractor.extract_features(user_patent, candidate)
    # Returns [f1, f2, ..., f10]
```

This uses the same FeatureExtractor we built for training. Computes:
1. PatentSBERTa similarity
2. TF-IDF similarity
3. Jaccard similarity
... all 10 features

If we have 58 candidates, we get a (58, 10) feature matrix.

Time: 2-3 seconds for 60 candidates."

### STEP 7: PyTorch Scoring (Lines 480-490)

**What to say:**

"Now we score all candidates using our trained PyTorch model:

```python
# features_matrix: (58, 10)
scores = pytorch_model.predict_proba(features_matrix)
# Returns: (58, 2) with [prob_class_0, prob_class_1]

for i, candidate in enumerate(candidates):
    candidate['model_similarity'] = scores[i, 1]  # Probability of similar
    candidate['model_novelty'] = 1 - scores[i, 1]  # Novelty = inverse
```

The model:
1. Scales features using the StandardScaler from training
2. Runs through the network: (58, 10) → BatchNorm → ResBlock → ResBlock → BatchNorm → Linear → Sigmoid → (58, 1)
3. Returns probabilities in [0, 1]

Batch inference is fast: ~0.5 seconds for 60 examples.

Example scores:
- US11234567: 0.92 (highly similar, low novelty)
- WO2023456789: 0.78 (similar)
- EP3456789: 0.15 (different, high novelty)"

### STEP 8: Ranking and Novelty Calculation (Lines 495-500)

**What to say:**

"Sort all candidates by similarity score (highest first):

```python
candidates.sort(key=lambda x: x['model_similarity'], reverse=True)
top_20 = candidates[:20]
```

Calculate overall novelty score:
```python
mean_similarity_of_top20 = mean([p['model_similarity'] for p in top_20])
novelty_score = 1 - mean_similarity_of_top20
```

Example:
- Top 20 similar patents have average similarity: 0.75
- Novelty score: 1 - 0.75 = 0.25 (LOW NOVELTY)

Interpretation:
- Novelty > 0.7: HIGH NOVELTY (likely patentable)
- Novelty 0.4-0.7: MODERATE NOVELTY (needs review)
- Novelty < 0.4: LOW NOVELTY (significant prior art exists)"

### STEP 9: LLM Explanation Generation (Lines 505-515)

**What to say:**

"Finally, we generate a human-readable report using Phi-3. This is implemented in phi3_explainer.py."

## File: `src/app/phi3_explainer.py`

### Lines 76-157: `_build_prompt()` Method

**What to say:**

"We construct a detailed prompt for Phi-3 that includes:

**Lines 96-106: Application Being Assessed**
```
PATENT APPLICATION:
Title: [user's title]
Abstract: [user's abstract, truncated to 1000 chars]
```

**Lines 108-120: Top Similar Prior Art**
```
PRIOR ART 1: US11234567 (Similarity: 92%)
Title: Wireless Power Transfer System
Abstract: A system for transmitting power wirelessly using...
Year: 2022

PRIOR ART 2: US10987654 (Similarity: 78%)
...
```

We include top 4 similar patents with their full details.

**Lines 123-156: Structured Instructions**
```
YOUR DETAILED ANALYSIS:

## VERDICT: [LOW NOVELTY]

## EXECUTIVE SUMMARY
[Write 3-4 sentences explaining the verdict]

## TECHNICAL OVERLAP ANALYSIS
For each prior art patent:
- Overlapping concepts
- Key quotes showing overlap

## NOVEL ELEMENTS
[What's new in this application]

## RECOMMENDATION
Decision: APPROVE/REVISE/REJECT
Reasoning: [Why this decision]
```

This structured format ensures Phi-3 generates a complete, organized report."

### Lines 159-227: `generate_explanation()` Method

**Lines 184-197: Call Ollama API**
"POST to localhost:11434 with:
```python
{
  'model': 'phi3',
  'prompt': [the prompt we built],
  'options': {
    'num_predict': 4000,  # Generate up to 4000 tokens
    'temperature': 0.4,   # Low = factual, high = creative
    'top_p': 0.9          # Nucleus sampling
  }
}
```

Temperature=0.4: We want factual analysis, not creative writing.

Phi-3 generates 800-1200 tokens in 30-45 seconds."

**Lines 200-212: Parse Response**
"Extract the generated text and timing information:
- Total tokens generated
- Generation speed (tokens/second)
- Whether response was truncated

Example output:
'## VERDICT: LOW NOVELTY

## EXECUTIVE SUMMARY
The proposed wireless charging system demonstrates LOW NOVELTY (score: 0.25). Significant prior art exists covering the core magnetic resonance approach (US11234567), multi-coil configuration (WO2023456789), and foreign object detection (EP3456789)...

## TECHNICAL OVERLAP ANALYSIS

Patent US11234567 (Similarity: 92%):
- Overlapping concepts: Magnetic resonance coupling, multi-coil transmitter, frequency tuning
- Key quote: \"utilizing magnetic resonance between transmitter and receiver coils at 6.78 MHz\"
...

## RECOMMENDATION
REJECT or significant revision needed. The core technical approach is well-established in prior art. Consider focusing on: [suggestions]'"

**Lines 229-283: `_parse_response()`**
"Parse the LLM text into structured NoveltyReport object with fields:
- novelty_score (float)
- assessment (HIGH/MODERATE/LOW/NOT NOVEL)
- summary (executive summary text)
- citations (list of cited prior art)
- recommendation (APPROVE/REVISE/REJECT)
- full_explanation (complete text)

This structured format makes it easy to display in the UI."

### Total Pipeline Time

"Let's add up all steps:
1. Embed query: 2-3s
2. Local search: 1-2s
3. LLM keywords: 10-15s
4. Online search: 15-25s
5. Merge: 0.1s
6. Features: 2-3s
7. PyTorch scoring: 0.5s
8. Rank: 0.1s
9. LLM explain: 30-45s

Total: 60-90 seconds for complete novelty assessment!"

---

# STAGE 6: WEB APPLICATION

## File: `app.py`

**What to say:**

"Finally, the Streamlit web app that ties everything together into a user-friendly interface."

### Lines 20-35: `load_analyzer()` Function with Caching

**What to say:**

"`@st.cache_resource` is crucial for performance. It means we load the analyzer ONCE and reuse it across all user requests.

Without caching:
- Every button click → load PatentSBERTa (10s) + load PyTorch model (2s) + load embeddings (5s) = 17s overhead

With caching:
- First load: 17s
- All subsequent requests: instant (reuses loaded models)

This decorator makes the app actually usable!"

### Lines 550-630: User Interface

**Lines 551-563: API Key Input**
"Password-protected text input for SerpAPI key. User pastes their key, we store in session state and environment variable. This enables online search."

**Lines 566-571: Configuration**
"Checkboxes and sliders let users configure:
- Enable/disable online search
- Enable/disable LLM keyword generation
- Number of results to retrieve"

**Lines 605-615: Patent Input**
"Large text area where user enters their patent description. Can be:
- Just a description of the invention
- Full abstract
- Title + abstract + claims (JSON format)"

### Lines 620-750: Analysis and Display

**Lines 630-640: Run Analysis**
"When user clicks 'Analyze Patent Novelty':
1. Load analyzer from cache (instant after first time)
2. Create progress bar and status text
3. Call analyzer.analyze() with status callback
4. Status callback updates the UI in real-time:
   - '🔄 Generating embeddings...'
   - '🔍 Searching local database...'
   - '🌐 Searching Google Patents...'
   - '🤖 Generating explanation...'
   - '✅ Analysis complete!'"

**Lines 645-750: Display Results**
"Show comprehensive results:

1. **Novelty Score Card** (lines 650-665)
   - Large metric showing score (0.0 - 1.0)
   - Color coded: Green (>0.7), Orange (0.4-0.7), Red (<0.4)
   - Assessment text: 'HIGH NOVELTY', 'MODERATE', or 'LOW NOVELTY'

2. **Similar Patents Table** (lines 670-695)
   - Top 15 similar patents
   - Columns: Patent ID, Title, Similarity Score, Year, Source (local/online)
   - Sortable and searchable
   - Click patent ID to open in Google Patents

3. **Search Metadata** (lines 700-715)
   - How many local vs online results
   - Which search terms were used
   - Total candidates evaluated

4. **AI Explanation** (lines 720-735)
   - Expandable section with full Phi-3 report
   - Structured format with sections
   - Cites specific prior art patents

5. **Download Buttons** (lines 740-750)
   - Download full report as text file
   - Download data as JSON
   - Includes all metadata and citations"

---

# CONCLUSION

**What to say:**

"Let me summarize the complete system:

**Data Pipeline:**
- Downloaded USPTO data (800K patents from 2021-2025)
- Sampled 200K diverse patents stratified by year
- Generated 768-dim embeddings using PatentSBERTa (11 hours)
- Extracted 57K training pairs from citations
- Computed 10 engineered features for each pair (17 hours)

**Model Development:**
- Grid search: 54 hyperparameter configurations × 3-fold CV
- Custom PyTorch architecture: 118K parameters, residual connections
- 5 regularization techniques: dropout, batch norm, weight decay, early stopping, mixup
- Final performance: 97.2% ROC-AUC on held-out test set

**Inference System:**
- Hybrid RAG: Local (200K, fast) + Online (millions, comprehensive)
- 9-step pipeline: Embed → Search → Extract → Score → Explain
- LLM integration: Phi-3 for keywords and explanations (privacy-preserving, runs locally)
- End-to-end: 60-90 seconds per query

**Production Deployment:**
- Streamlit web app with caching for performance
- Real-time progress updates during inference
- Professional UI with metrics, tables, and explanations
- Downloadable reports

This demonstrates expertise in:
- Large-scale data processing (200K patents, 586MB embeddings)
- Advanced NLP (transformer models, domain-specific embeddings)
- Deep learning (custom PyTorch architecture, multiple regularization)
- Positive-Unlabeled learning (citation-based training)
- Hybrid RAG architecture (local + online search)
- LLM integration (Phi-3 via Ollama)
- API integration (SerpAPI for Google Patents)
- Full-stack development (data → training → inference → deployment)

Thank you for your attention. I'm happy to answer any questions about the implementation!"

---

# KEY FILES QUICK REFERENCE

**Data Collection & Preprocessing:**
1. `scripts/training/sample_diverse_patents.py` - Stratified sampling of 200K patents
2. `scripts/data/preprocessing/generate_embeddings.py` - PatentSBERTa embedding generation

**Training Data:**
3. `scripts/training/extract_citation_pairs.py` - Citation-based pair extraction

**Feature Engineering:**
4. `src/features/feature_extract.py` - 10 feature definitions
5. `scripts/data/preprocessing/compute_features.py` - Run feature extraction

**Model Training:**
6. `scripts/evaluation/tuning/nn_tuning.py` - Hyperparameter grid search
7. `src/app/pytorch_classifier.py` - Neural network architecture & training

**Inference:**
8. `src/app/patent_analyzer.py` - Main inference orchestrator
9. `src/app/phi3_explainer.py` - LLM explanation generation
10. `data/api/online_search.py` - SerpAPI + LLM keywords

**Application:**
11. `app.py` - Streamlit web interface

---

**END OF PRESENTATION SCRIPT**
