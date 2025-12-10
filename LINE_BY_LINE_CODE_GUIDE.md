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

"Now that we have our 200,000 patents, we need to convert them into numerical representations that machine learning models can work with. This is where PatentSBERTa comes in. PatentSBERTa is a BERT model that was fine-tuned by AI-Growth-Lab on 1.2 million patent abstracts. Unlike general BERT which was trained on Wikipedia, PatentSBERTa understands patent-specific terminology like 'prior art,' 'embodiments,' and technical jargon."

## File: `scripts/data/preprocessing/generate_embeddings.py`

**What to say:**

"This script converts all 200,000 patents into 768-dimensional embedding vectors. It took about 11 hours to run on my Apple M1 machine with GPU acceleration."

### Lines 1-20: Imports
"We're using the `sentence-transformers` library which provides easy access to BERT-based models. PatentSBERTa is hosted on HuggingFace's model hub."

### Lines 32-56: `get_patent_text()` Function
"This function extracts the best text representation from a patent for embedding."

**Lines 34-35:** "We prefer abstracts because they're concise but comprehensive - usually 100-300 words that summarize the entire invention."

**Lines 37-38:** "If there's no abstract, we fall back to the summary section."

**Lines 39-47:** "If there's no summary either, we use the first claim. Notice the defensive programming here - claims can be dictionaries with a 'text' key or just plain strings. We handle both cases."

**Why truncate to 500 characters at line 35?** "PatentSBERTa is based on BERT which has a maximum sequence length of 512 tokens. A token is roughly 0.75 words, so 512 tokens is about 384 words or roughly 2000 characters. We use 500 characters to be conservative and ensure we never hit that limit."

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

"Now let me show you the actual neural network architecture and training implementation. This is a custom PyTorch model with residual connections, batch normalization, and multiple regularization techniques."

### Lines 26-66: `ResidualBlock` Class - BUILDING BLOCK

**What to say:**

"The ResidualBlock is inspired by ResNet. Let me explain the concept and implementation."

**Theory:**
"In deep neural networks, we can encounter the vanishing gradient problem. As gradients backpropagate through many layers, they get multiplied by many numbers less than 1, causing them to shrink exponentially:

gradient_at_layer_1 = gradient_at_output × (∂layer10/∂layer9) × ... × (∂layer2/∂layer1)

If each partial derivative is 0.5, then (0.5)^9 = 0.002 - the gradient vanishes!

Residual connections solve this by creating an alternate gradient path. Instead of learning f(x), we learn f(x) + x. The '+x' creates a shortcut for gradients to flow backward."

**Implementation:**

**Lines 43-46: Main Transformation Path**
"The main path does: Linear → BatchNorm → ReLU → Dropout

- Linear: Learnable transformation (Wx + b)
- BatchNorm: Normalizes to mean=0, std=1 for stable training
- ReLU: Non-linear activation max(0, x)
- Dropout: Randomly zeros 30% of neurons to prevent overfitting"

**Lines 49-54: Skip Connection**
"If input and output dimensions are different (e.g., 10→256), we need to project the skip connection to match dimensions. Otherwise, we use identity (just pass through unchanged).

Example:
- Input: (batch, 10)
- Main path transforms: (batch, 10) → (batch, 256)
- Skip path projects: (batch, 10) → (batch, 256)
- Now we can add them!"

**Lines 56-66: Forward Pass**
"The forward method:
1. Compute skip connection (with or without projection)
2. Apply main transformation
3. Add skip to output: `out = transformation(x) + skip(x)`

This '+skip(x)' ensures gradients can flow backward easily, enabling deeper networks."

### Lines 69-139: `PatentNoveltyNet` Class - FULL ARCHITECTURE

**What to say:**

"This is our complete neural network. Let me show you the architecture."

**Architecture Overview:**
"The network transforms 10 input features through these layers:

Input (10 features)
    ↓
InputBatchNorm (normalize inputs)
    ↓
ResidualBlock (10 → 256) with BatchNorm, ReLU, Dropout
    ↓
ResidualBlock (256 → 128) with BatchNorm, ReLU, Dropout
    ↓
OutputBatchNorm (normalize before final layer)
    ↓
Linear (128 → 1) 
    ↓
Sigmoid (squash to [0,1] probability)
    ↓
Output: probability that patents are similar"

**Lines 87-111: Layer Construction**

**Line 90:** "Input batch normalization - normalizes the 10 features to have mean=0, std=1. This helps training stability even though we already StandardScaler the features."

**Lines 92-103:** "Build the hidden layers. We use a loop to construct ResidualBlocks for each layer size:
- First block: 10 → 256 (expands feature space to learn interactions)
- Second block: 256 → 128 (compresses to extract important patterns)"

**Lines 107-108:** "Output layer: 128 → 1. The final sigmoid activation squashes the output to [0, 1], making it interpretable as a probability."

**Lines 113-119: Xavier Initialization**
"We initialize weights using Xavier uniform initialization. For each linear layer:

limit = sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limit, +limit)

For our 10→256 layer:
limit = sqrt(6 / (10+256)) = sqrt(0.0226) = 0.15
Weights initialized in [-0.15, +0.15]

This initialization ensures variance is maintained across layers, preventing exploding/vanishing activations."

**Lines 121-139: Forward Pass**

"During forward propagation:
1. Input (batch, 10) → InputBatchNorm → (batch, 10) normalized
2. → ResidualBlock 1 → (batch, 256)
3. → ResidualBlock 2 → (batch, 128)
4. → OutputBatchNorm → (batch, 128) normalized
5. → Linear → (batch, 1) logits
6. → Sigmoid → (batch, 1) probabilities

For a batch of 256 examples, each step processes all 256 in parallel using matrix operations."

**Parameter Count:**
"Let me calculate total parameters:
- InputBN: 20 (γ, β for 10 features)
- ResBlock 10→256: 10×256 + 256 bias + 256×2 (BN) + skip projection 10×256 + 256 + 512 (BN) ≈ 5,888
- ResBlock 256→128: 256×128 + 128 + 256 (BN) + skip projection 256×128 + 128 + 256 (BN) ≈ 66,048
- OutputBN: 256 (γ, β for 128 features)
- OutputLinear: 128×1 + 1 = 129
Total: ~118,421 parameters

At 4 bytes per float32, that's ~473 KB - a very lightweight model!"

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
