# CS 372 Final Project - Self Assessment

## Selected Rubric Items with Evidence

Total Points: 121

---

## 1. Completed project individually without a partner (10 pts)

**Evidence:**
- Solo project with no partner
- All design decisions, implementation, and evaluation conducted by me

**Files:**
- `ATTRIBUTION.md` - Documents individual effort for specific tasks
- `README.md` - Individual contributions section

---

## 2. Built retrieval-augmented generation (RAG) system with document retrieval and generation components (10 pts)

**Evidence:**
- Hybrid RAG architecture combining local and online retrieval with LLM generation
- Local retrieval: PatentSBERTa embeddings for 200K patents using cosine similarity
- Online retrieval: SerpAPI integration for Google Patents search
- Generation: Phi-3 LLM generates novelty explanations from retrieved documents
- Complete pipeline: Query → Embedding → Retrieval (local + online) → Ranking → Explanation generation

**Files:**
- `src/app/patent_analyzer.py` (lines 75-432) - RAG orchestration and retrieval logic
- `src/app/phi3_explainer.py` (lines 26-199) - LLM-based generation component
- `data/api/online_search.py` - SerpAPI integration for online retrieval
- `data/embeddings/patent_embeddings.npy` - Local retrieval database (200K embeddings)
- `app.py` - Production RAG deployment

---

## 3. Deployed model as functional web application with user interface (10 pts)

**Evidence:**
- Fully functional Streamlit web application for patent novelty assessment
- Interactive multi-tab interface (Novelty Assessment, Prior Art Search)
- Real-time analysis with status updates
- Report downloads (TXT, JSON formats)
- Production deployment with error handling and user feedback

**Files:**
- `app.py` (804 lines) - Complete Streamlit application
- `SETUP.md` - Deployment and setup instructions
- `README.md` - Usage guide and quick start

---

## 4. Built multi-stage ML pipeline connecting outputs of one model to inputs of another (7 pts)

**Evidence:**
- Stage 1: PatentSBERTa transformer generates embeddings from patent text
- Stage 2: Feature extraction combines embeddings with metadata to create 10 engineered features
- Stage 3: PyTorch neural network produces novelty scores from features
- Stage 4: Phi-3 LLM generates explanations conditioned on novelty scores
- Complete data flow: Raw text → Embeddings → Features → Classification scores → Text explanations

**Files:**
- `src/app/patent_analyzer.py` (lines 294-500) - Multi-stage pipeline orchestration
- `src/features/feature_extract.py` - Feature engineering from embeddings
- `src/app/pytorch_classifier.py` - PyTorch classification model
- `src/app/phi3_explainer.py` - Conditional text generation based on scores
- `scripts/data/preprocessing/generate_embeddings.py` - Stage 1 implementation

---

## 5. Implemented agentic system where model outputs trigger automated actions or tool calls (7 pts)

**Evidence:**
- LLM keyword extraction output triggers SerpAPI search calls
- Novelty score thresholds trigger different recommendation logic
- Search terms automatically trigger multi-query online patent searches
- Model confidence scores determine adaptive retrieval strategies (top-k selection)
- System makes autonomous decisions about API calls and tool usage based on model outputs

**Files:**
- `src/app/patent_analyzer.py` (lines 320-407) - LLM outputs trigger SerpAPI tool calls
- `data/api/online_search.py` - Automated API integration
- `src/app/phi3_explainer.py` (lines 76-157) - Score-based prompt adaptation
- `app.py` (lines 420-443) - Score-based recommendation logic

---

## 6. Used or fine-tuned a transformer language model (7 pts)

**Evidence:**
- PatentSBERTa: Transformer-based BERT model for patent text embeddings (200K patents embedded)
- Phi-3: Transformer LLM for explanation generation via Ollama
- Both models integrated into production pipeline with domain-specific usage

**Files:**
- `src/app/patent_analyzer.py` (lines 134-159) - PatentSBERTa loading and inference
- `src/app/phi3_explainer.py` - Phi-3 transformer for text generation
- `scripts/data/preprocessing/generate_embeddings.py` (lines 59-86) - Batch PatentSBERTa inference
- `src/features/claim_embeddings.py` - Transformer embeddings for patent claims
- `data/embeddings/patent_embeddings.npy` - 200K transformer-generated embeddings

---

## 7. Conducted systematic hyperparameter tuning using validation data or cross-validation (5 pts)

**Evidence:**
- 54 hyperparameter configurations tested
- 3-fold stratified cross-validation for all configurations
- Grid search over: hidden dimensions, dropout rates, learning rates, weight decay, batch sizes
- 5.4 hours of systematic tuning with full result logging
- Best configuration selected based on mean CV ROC-AUC score

**Files:**
- `scripts/evaluation/tuning/nn_tuning.py` - Hyperparameter tuning script with CV
- `results/hyperparameter_tuning/pytorch/pytorch_hyperparameter_results.json` - All 54 configurations with CV scores
- `src/app/pytorch_classifier.py` - Production model using tuned hyperparameters

---

## 8. Applied regularization techniques to prevent overfitting (at least two: L2 penalty, dropout, early stopping) (5 pts)

**Evidence:**
- L2 Regularization: weight_decay=1e-05 in AdamW optimizer
- Dropout: dropout=0.3 applied in all hidden layers via ResidualBlock
- Early Stopping: patience=15 epochs with validation loss monitoring
- All three techniques used in production PyTorch model

**Files:**
- `src/app/pytorch_classifier.py`:
  - Line 154, 174, 282: L2 weight decay parameter
  - Lines 41-45, 64, 102: Dropout layer implementations
  - Lines 159-160, 296-298, 346-356: Early stopping implementation
- `results/hyperparameter_tuning/pytorch/pytorch_hyperparameter_results.json` - Tuning results showing regularization effectiveness

---

## 9. Applied feature engineering (created embeddings and derived features) (5 pts)

**Evidence:**
- 10 engineered features combining embeddings, text statistics, and domain knowledge:
  1. Cosine document similarity (from PatentSBERTa embeddings)
  2. Max claim cosine similarity
  3. Embedding difference mean
  4. Embedding difference standard deviation
  5. Year difference (normalized)
  6. Title Jaccard similarity
  7. Abstract length ratio
  8. Claim count ratio
  9. Shared rare terms ratio
  10. Claim similarity (TF-IDF based)
- Features extracted for all 59,114 training pairs

**Files:**
- `src/features/feature_extract.py` (160 lines) - Complete feature extraction pipeline
- `src/features/claim_embeddings.py` - Claim-level embedding features
- `scripts/data/preprocessing/compute_features.py` - Batch feature computation
- `data/features/feature_names_v2.json` - Feature definitions
- `data/features/train_features_v2.X.npy` - Computed feature matrices (59K samples)

---

## 10. Conducted ablation study demonstrating impact of at least two design choices with quantitative comparison (5 pts)

**Evidence:**
- 5 ablation experiments removing different feature groups
- Quantitative comparison showing performance impact:
  - All Features: ROC-AUC = 0.9688
  - Without embedding: ROC-AUC = 0.9321 (3.67% drop)
  - Without text similarity: ROC-AUC = 0.9317 (3.71% drop)
  - Without metadata: ROC-AUC = 0.9233 (4.55% drop)
  - Without claim features: ROC-AUC = 0.9337 (3.51% drop)
- Results demonstrate embeddings and metadata are critical features

**Files:**
- `results/analysis/ablation_study/ablation_results.json` - Full quantitative results for all experiments
- `results/plots/ablation/ablation_study_comparison.png` - Visual comparison
- `scripts/evaluation/run_ablation_study.py` - Ablation experiment implementation
- `scripts/evaluation/plots/plot_ablation.py` - Results visualization
- `README.md` (lines 96-113) - Ablation study results table

---

## 11. Compared multiple model architectures or approaches quantitatively (5 pts)

**Evidence:**
- 6 models compared with quantitative metrics:
  1. Random Guessing: 50.3% ROC-AUC (baseline)
  2. Majority Class: 50.0% ROC-AUC (baseline)
  3. Logistic Regression: 96.7% ROC-AUC
  4. Cosine Similarity Heuristic: 91.7% ROC-AUC
  5. MLP Classifier: 94.5% ROC-AUC
  6. PyTorch Neural Net: 97.2% ROC-AUC (production model)
- PyTorch model demonstrates 2.7% improvement over MLP baseline

**Files:**
- `results/plots/baseline/baseline_results.json` - Quantitative comparison results
- `results/plots/baseline/baseline_comparison.png` - Visual comparison
- `scripts/evaluation/plots/plot_baseline_comparison.py` - Comparison implementation
- `README.md` (lines 86-94) - Baseline comparison table

---

## 12. Defined and trained a custom neural network architecture using PyTorch (5 pts)

**Evidence:**
- Custom PatentNoveltyNet architecture with residual connections
- Custom ResidualBlock module with skip connections and batch normalization
- Architecture features: input batch norm, residual blocks, output batch norm, Xavier initialization
- Not a pretrained model - designed specifically for patent novelty assessment
- Trained from scratch on 59K patent pairs

**Files:**
- `src/app/pytorch_classifier.py`:
  - Lines 26-66: Custom ResidualBlock class
  - Lines 69-139: Custom PatentNoveltyNet architecture
  - Lines 142-502: Complete PyTorchPatentClassifier training implementation
- `models/pytorch_model.pt` - Trained model weights
- `results/plots/pytorch/training_curve.png` - Training curves

---

## 13. Implemented preprocessing pipeline handling data quality issues (5 pts)

**Evidence:**
- Handles missing abstract/title/claim data with fallbacks
- Processes inconsistent patent formats across multiple years (2021-2025)
- Normalizes patent IDs and dates from multiple sources
- Filters empty or malformed patents
- Handles class imbalance through balanced pair sampling
- Standardizes features using StandardScaler

**Files:**
- `scripts/data/preprocessing/compute_features.py` (lines 24-112) - Preprocessing with missing data handling
- `scripts/data/preprocessing/generate_embeddings.py` (lines 32-56) - Text extraction with fallbacks
- `scripts/training/extract_citation_pairs.py` (lines 85-202) - Balanced pair generation
- `src/app/pytorch_classifier.py` (lines 263-265) - StandardScaler for feature normalization
- `data/processed/processing_stats.json` - Processing statistics

---

## 14. Used sentence embeddings for semantic similarity or retrieval (5 pts)

**Evidence:**
- PatentSBERTa sentence embeddings for all 200K patents in database
- Cosine similarity for semantic retrieval of similar patents
- Real-time query embedding for similarity search
- Embedding-based features (cosine similarity, embedding difference statistics)
- Production retrieval returns top-k most similar patents based on embedding cosine distance

**Files:**
- `scripts/data/preprocessing/generate_embeddings.py` - Embedding generation for 200K patents
- `data/embeddings/patent_embeddings.npy` - 200K pre-computed embeddings
- `data/embeddings/patent_ids.json` - Embedding to patent ID mapping
- `src/app/patent_analyzer.py` (lines 341-342) - Cosine similarity retrieval
- `src/features/feature_extract.py` (lines 118-125, 147-157) - Embedding-based features

---

## 15. Implemented production-grade deployment with multiple considerations (10 pts)

**Evidence:**
- **Caching:** `@st.cache_resource` decorator for analyzer loading to prevent redundant model loading (performance optimization)
- **Error Handling:** Comprehensive try-except blocks throughout app with graceful error messages to users (reliability)
- **Logging:** Extensive print statements and error tracking for debugging and monitoring (147+ log statements across app)
- **Status Updates:** Real-time status callbacks during analysis to inform users of progress (user experience)
- **Session State Management:** Proper state handling to preserve results across Streamlit reruns (stateful operations)
- All five production considerations implemented in deployed web application

**Justification:**
The rubric states this item is for "production-grade deployment (evidence of at least two considerations such as rate limiting, caching, monitoring, error handling, logging)." This project qualifies because:
- Localhost deployment is still production-grade if it demonstrates production practices
- The rubric focuses on implementation quality, not hosting infrastructure
- Five production considerations implemented (only needed two)
- System is functional, reliable, and handles real user interactions
- `SETUP.md` provides complete deployment instructions for running locally

**Files:**
- `app.py`:
  - Line 28: `@st.cache_resource` for model caching
  - Lines 638-685: Error handling for novelty analysis with user-friendly error messages
  - Lines 786-790: Error handling for search functionality
  - Lines 611-618, 743-751: Status callback implementation for real-time updates
  - Lines 547-586: Configuration management with session state
- `src/app/patent_analyzer.py`:
  - Lines 134-259: Error handling in load() method with status callbacks
  - Lines 260-292: Error handling in analyze() with fallback logic
- `src/app/pytorch_classifier.py`:
  - Lines 22-23: Logging configuration
  - Lines 294-364: Status logging during training
- `SETUP.md` - Complete deployment instructions

---

## 16. Processed and successfully trained on exceptionally large dataset (>100K samples for NLP) (10 pts)

**Evidence:**
- 200,000 patents processed from PatentsView (2021-2025)
- Successfully generated PatentSBERTa embeddings for all 200K patents
- 57,114 training pairs (patent pairs) created from citation data
- Trained PyTorch neural network on 39,979 training samples (48,546 samples used in 3-fold CV during hyperparameter tuning)
- All processing completed with batch processing and memory-efficient pipelines

**Files:**
- `data/sampled/patents_sampled.jsonl` - 200,000 patents (verified via wc -l)
- `data/sampled/sampling_metadata.json` - Documents 200K total with stratified year distribution
- `data/embeddings/patent_embeddings.npy` - 200K embeddings successfully generated
- `data/training/dataset_stats.json` - 57,114 training pairs (39,979 train + 8,567 val + 8,568 test)
- `scripts/data/preprocessing/generate_embeddings.py` - Batch processing of 200K patents
- `scripts/training/extract_citation_pairs.py` - Large-scale citation extraction from 100M+ records
- `results/hyperparameter_tuning/pytorch/pytorch_hyperparameter_results.json` - Training on full dataset
- `ATTRIBUTION.md` (line 14) - Documents 200K patent corpus

---

## 17. Collected or constructed original dataset through substantial engineering effort with documented methodology (10 pts)

**Evidence:**
- 200,000 patents collected from PatentsView API (2021-2025) across multiple years
- 57,114 training pairs extracted from 100M+ citation records (10GB TSV file)
- Citation-based positive pair generation (patents that cite each other are similar)
- Random negative pair sampling with de-duplication to prevent data leakage
- Multi-stage preprocessing pipeline handling inconsistent patent data formats across years
- Stratified train/validation/test splits (70% / 15% / 15%) with balanced class distribution
- Documented methodology in scripts and metadata files

**Justification:**
The rubric states this item is for datasets collected through "substantial engineering effort (e.g., API integration, web scraping, manual annotation/labeling, custom curation) with documented methodology." This project qualifies because:
- API integration with PatentsView to download 100GB+ of raw patent data
- Custom curation: Processing 5 years of data, filtering, sampling 200K patents
- Engineering effort: Handling inconsistent formats, missing data, extracting citations from 100M+ records
- Documented: All processing steps documented in scripts with metadata files tracking statistics

**Files:**
- `scripts/training/extract_citation_pairs.py` (224 lines) - Citation extraction from 10GB TSV file scanning 100M+ rows
- `scripts/data/preprocessing/generate_embeddings.py` - Batch embedding generation for 200K patents
- `scripts/data/preprocessing/compute_features.py` - Feature computation pipeline
- `scripts/training/sample_diverse_patents.py` - Stratified sampling across years
- `data/sampled/patents_sampled.jsonl` - 200,000 curated patents
- `data/training/train_pairs.jsonl`, `val_pairs.jsonl`, `test_pairs.jsonl` - 57,114 pairs with splits
- `data/citations/g_us_patent_citation.tsv` - 10GB raw citation data
- `data/training/dataset_stats.json` - Dataset statistics and split ratios
- `data/sampled/sampling_metadata.json` - Sampling methodology documentation

---

## Summary

Total Points: 121

This exceeds the 70-point requirement by 51 points, with all evidence based on production implementation and clearly documented in the codebase.

