# CS 372 Final Project - Self Assessment

## Selected Rubric Items with Evidence

**Total Items:** 15  
**Total Points:** 99

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
- Complete pipeline: Query -> Embedding -> Retrieval (local + online) -> Ranking -> Explanation generation

**Files:**
- `src/app/patent_analyzer.py` (lines 75-432) - RAG orchestration and retrieval logic
- `src/app/phi3_explainer.py` (lines 26-199) - LLM-based generation component
- `data/api/online_search.py` - SerpAPI integration for online retrieval
- `data/embeddings/patent_embeddings.npy` - Local retrieval database (200K embeddings, 586MB, available via Duke Box)
- `app.py` - RAG deployment in Streamlit
- `notebooks/pipeline.ipynb` - Complete RAG pipeline demonstration with execution outputs

---

## 3. Deployed model as functional web application with user interface (10 pts)

**Evidence:**
- Fully functional Streamlit web application for patent novelty assessment
- Interactive multi-tab interface (Novelty Assessment, Prior Art Search)
- Real-time analysis with status updates
- Report downloads (TXT, JSON formats)
- Deployment with error handling and user feedback

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
- Complete data flow: Raw text -> Embeddings -> Features -> Classification scores -> Text explanations

**Files:**
- `src/app/patent_analyzer.py` (lines 294-500) - Multi-stage pipeline orchestration connecting all 4 stages
- `src/embeddings/patent_sberta.py` (292 lines) - Stage 1: PatentSBERTa transformer embeddings
- `src/features/feature_extract.py` (160 lines) - Stage 2: Feature engineering from embeddings
- `src/app/pytorch_classifier.py` (502 lines) - Stage 3: PyTorch neural network classification
- `src/app/phi3_explainer.py` (319 lines) - Stage 4: Conditional text generation based on scores
- `scripts/data/preprocessing/generate_embeddings.py` - Batch processing for Stage 1 (200K patents)
- `notebooks/pipeline.ipynb` - Complete end-to-end demonstration with execution outputs

---

## 5. Implemented agentic system where model outputs trigger automated actions or tool calls (7 pts)

**Evidence:**
- LLM keyword extraction output triggers SerpAPI search calls
- Novelty score thresholds trigger different recommendation logic
- Search terms automatically trigger multi-query online patent searches
- Model confidence scores determine adaptive retrieval strategies (top-k selection)
- System makes autonomous decisions about API calls and tool usage based on model outputs

**Files:**
- `src/app/patent_analyzer.py` (lines 320-407) - LLM keyword extraction output triggers SerpAPI tool calls
- `data/api/online_search.py` (239 lines) - Automated Google Patents API integration
- `src/app/phi3_explainer.py` (lines 76-157) - Score-based prompt adaptation and tool usage decisions
- `app.py` (lines 420-443) - Score-based recommendation logic (novelty thresholds)
- `notebooks/pipeline.ipynb` - Demonstrates agentic behavior with SerpAPI tool calls triggered by LLM

---

## 6. Used or fine-tuned a transformer language model (7 pts)

**Evidence:**
- PatentSBERTa: Transformer-based BERT model for patent text embeddings (200K patents embedded)
- Phi-3: Transformer LLM for explanation generation via Ollama
- Both models integrated into pipeline with domain-specific usage

**Files:**
- `src/app/patent_analyzer.py` (lines 134-159) - PatentSBERTa loading and inference
- `src/app/phi3_explainer.py` - Phi-3 transformer for text generation
- `scripts/data/preprocessing/generate_embeddings.py` (lines 59-86) - Batch PatentSBERTa inference
- `src/features/claim_embeddings.py` - Transformer embeddings for patent claims
- `data/embeddings/patent_embeddings.npy` - 200K transformer-generated embeddings (586MB, available via Duke Box)
- `notebooks/pipeline.ipynb` - Demonstrates both transformers with execution outputs

---

## 7. Conducted systematic hyperparameter tuning using validation data or cross-validation (5 pts)

**Evidence:**
- 54 hyperparameter configurations tested
- 3-fold stratified cross-validation for all configurations
- Grid search over: hidden dimensions, dropout rates, learning rates, weight decay, batch sizes
- 5.4 hours of systematic tuning with full result logging
- Best configuration selected based on mean CV ROC-AUC score

**Files:**
- `scripts/evaluation/tuning/nn_tuning.py` (423 lines) - Hyperparameter tuning script with 3-fold CV
- `results/hyperparameter_tuning/pytorch_tuning.json` - All 54 configurations with CV scores and statistics
- `src/app/pytorch_classifier.py` - Final model using best hyperparameters: hidden_dims=[256, 128], dropout=0.3, lr=0.002
- `notebooks/pytorch_classifier.ipynb` - Training notebook showing hyperparameter selection process

---

## 8. Applied regularization techniques to prevent overfitting (at least two: L2 penalty, dropout, early stopping) (5 pts)

**Evidence:**
- L2 Regularization: weight_decay=1e-05 in AdamW optimizer
- Dropout: dropout=0.3 applied in all hidden layers via ResidualBlock
- Early Stopping: patience=15 epochs with validation loss monitoring
- All three techniques used in final PyTorch model

**Files:**
- `src/app/pytorch_classifier.py`:
  - Lines 282-283: L2 weight decay in AdamW optimizer (`weight_decay=1e-05`)
  - Lines 44-45, 64, 102: Dropout layer implementations (`nn.Dropout(dropout)`)
  - Lines 296-298, 346-356: Early stopping with patience=15 epochs
- `results/hyperparameter_tuning/pytorch_tuning.json` - Shows impact of dropout rates (0.1, 0.2, 0.3) on CV performance
- `results/plots/pytorch/training_curve.png` - Visualizes early stopping behavior

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
- `src/features/feature_extract.py` (160 lines) - Complete feature extraction pipeline with all 10 features
- `src/features/claim_embeddings.py` - Claim-level embedding features (max claim similarity)
- `scripts/data/preprocessing/compute_features.py` - Batch feature computation for 57K pairs
- `data/features/feature_names_v2.json` - Feature name definitions (10 features after ablation)
- `data/features/train_features_v2.X.npy` - Training feature matrix (39,979 × 10)
- `data/features/val_features_v2.X.npy` - Validation feature matrix (8,567 × 10)
- `data/features/test_features_v2.X.npy` - Test feature matrix (8,568 × 10)

---


## 10. Compared multiple model architectures or approaches quantitatively (5 pts)

**Evidence:**
- 6 models compared with quantitative metrics:
  1. Random Guessing: 50.3% ROC-AUC (baseline)
  2. Majority Class: 50.0% ROC-AUC (baseline)
  3. Logistic Regression: 96.7% ROC-AUC
  4. Cosine Similarity Heuristic: 91.7% ROC-AUC
  5. MLP Classifier: 94.5% ROC-AUC
  6. PyTorch Neural Net: 97.0% ROC-AUC (final model)
- PyTorch model demonstrates 2.5% improvement over MLP baseline

**Files:**
- `results/metrics/baseline_comparison.json` - Quantitative comparison results with all 6 model metrics
- `results/plots/baseline/baseline_comparison.png` - Visual bar chart comparing all models
- `scripts/plots/plot_baseline_comparison.py` - Comparison implementation and visualization
- `results/metrics/mlp_metrics.json` - MLP baseline: 94.5% ROC-AUC
- `results/metrics/pytorch_metrics.json` - PyTorch final: 97.0% ROC-AUC
- `README.md` (Model Comparison section) - Baseline comparison table with all metrics

---

## 11. Defined and trained a custom neural network architecture using PyTorch (5 pts)

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
- `models/pytorch_nn/pytorch_model.pt` - Trained model weights (~2 MB, available via Duke Box)
- `models/pytorch_nn/scaler_pytorch.pkl` - Feature scaler (~20 KB, available via Duke Box)
- `results/plots/pytorch/training_curve.png` - Training curves visualization
- `notebooks/pytorch_classifier.ipynb` - Training notebook with execution outputs

---

## 12. Implemented preprocessing pipeline handling data quality issues (5 pts)

**Evidence:**
- **Missing Data:** Fallback logic for missing abstracts/titles/claims (abstract -> summary -> claims -> empty string)
- **Class Imbalance:** Balanced pair generation (28,557 positive + 28,557 negative pairs, ratio=0.5)
- **Text Extraction:** Handles inconsistent patent formats across 5 years (2021-2025)
- **Normalization:** StandardScaler for feature standardization (fit on train, transform on val/test)
- **Filtering:** Removes empty or malformed patents during embedding generation
- **Impact:** Enables training on clean, balanced dataset with 91.20% accuracy

**Files:**
- `scripts/data/preprocessing/generate_embeddings.py` (lines 32-56) - Text extraction with missing data fallbacks
- `scripts/training/extract_citation_pairs.py` (lines 64-113) - Balanced negative pair generation for class imbalance
- `src/app/pytorch_classifier.py` (lines 263-265) - StandardScaler implementation
- `data/training/dataset_stats.json` - Shows balanced pairs: `"positive_ratio": 0.5`
- `data/processed/processing_stats.json` - Processing statistics and data quality metrics

---

## 13. Used sentence embeddings for semantic similarity or retrieval (5 pts)

**Evidence:**
- PatentSBERTa sentence embeddings for all 200K patents in database
- Cosine similarity for semantic retrieval of similar patents
- Real-time query embedding for similarity search
- Embedding-based features (cosine similarity, embedding difference statistics)
- Retrieval system returns top-k most similar patents based on embedding cosine distance

**Files:**
- `scripts/data/preprocessing/generate_embeddings.py` - Embedding generation for 200K patents
- `data/embeddings/patent_embeddings.npy` - 200K pre-computed embeddings (586MB, available via Duke Box)
- `data/embeddings/patent_ids.json` - Embedding to patent ID mapping (2.3MB, available via Duke Box)
- `data/embeddings/embedding_metadata.json` - Documents embedding generation (11.1 hours processing time)
- `src/app/patent_analyzer.py` (lines 633-662) - Cosine similarity retrieval implementation
- `src/features/feature_extract.py` (lines 118-125, 147-157) - Embedding-based feature extraction

---

## 14. Used at least three distinct and appropriate evaluation metrics for your task (3 pts)

**Evidence:**
- Seven evaluation metrics used to assess model performance:
  1. **Accuracy** - Overall correctness (91.20% on test set)
  2. **Precision** - Positive prediction reliability (93.98%)
  3. **Recall** - True positive detection rate (87.87%)
  4. **F1 Score** - Harmonic mean of precision/recall (90.82%)
  5. **ROC-AUC** - Ranking quality (97.01%)
  6. **Recall@K** - Retrieval effectiveness (R@10: 99.9%)
  7. **MRR (Mean Reciprocal Rank)** - Average rank of first relevant result (0.9996)
- Metrics appropriate for binary classification and information retrieval tasks
- Results documented across multiple JSON files

**Files:**
- `results/metrics/pytorch_metrics.json` - Contains accuracy, precision, recall, F1, ROC-AUC
- `results/metrics/mlp_metrics.json` - Baseline model metrics for comparison
- `results/metrics/recall_mrr.json` - Retrieval metrics (Recall@K, MRR)
- `results/metrics/additional_metrics.json` - Brier score and log loss
- `notebooks/pytorch_classifier.ipynb` - Displays all metrics in execution outputs

---

## 15. Processed and successfully trained on exceptionally large dataset (>100K samples for NLP) (10 pts)

**Evidence:**
- 200,000 patents processed from PatentsView (2021-2025)
- Successfully generated PatentSBERTa embeddings for all 200K patents
- 57,114 training pairs (patent pairs) created from citation data
- Trained PyTorch neural network on 39,979 training samples (48,546 samples used in 3-fold CV during hyperparameter tuning)
- All processing completed with batch processing and memory-efficient pipelines

**Files and Verification:**
- `data/sampled/patents_sampled.jsonl` - **200,000 patents** (verified: `wc -l` returns exactly 200000 lines, file size 3.8GB, available via Duke Box - see `SETUP.md`)
- `data/sampled/sampling_metadata.json` - Documents `"total_sampled": 200000` with stratified year distribution (40K per year × 5 years)
- `data/embeddings/patent_embeddings.npy` - **Shape: (200000, 768)** verified via `np.load().shape`, file size 586MB (available via Duke Box - see `SETUP.md`)
- `data/embeddings/patent_ids.json` - **200,000 unique patent IDs** (verified: `len(json.load()) = 200000`, file size 2.3MB, available via Duke Box)
- `data/embeddings/embedding_metadata.json` - Documents `"num_patents": 200000` with 11.1 hours processing time (40,110 seconds)
- `data/training/dataset_stats.json` - 57,114 training pairs (39,979 train + 8,567 val + 8,568 test) with balanced positive/negative ratio
- `models/pytorch_nn/pytorch_model.pt` - Trained on 39,979 samples (~2 MB, available via Duke Box)
- `models/pytorch_nn/scaler_pytorch.pkl` - Feature scaler (~20 KB, available via Duke Box)
- `notebooks/pipeline.ipynb` (line 115) - Live execution output: "Loaded 200,000 embeddings"
- `scripts/data/preprocessing/generate_embeddings.py` - Batch processing implementation for 200K patents
- `scripts/training/extract_citation_pairs.py` - Large-scale citation extraction from 100M+ citation records
- `ATTRIBUTION.md` (line 14) - Documents 200K patent corpus source
- `SETUP.md` (lines 83-232) - Duke Box download instructions for all large data/model files
- `README.md` - Documents hybrid RAG architecture using 200K patent database

**Verification Commands:**
```bash
# Patent count verification
$ wc -l data/sampled/patents_sampled.jsonl
200000 data/sampled/patents_sampled.jsonl

# Embeddings shape verification
$ python -c "import numpy as np; print(np.load('data/embeddings/patent_embeddings.npy', mmap_mode='r').shape)"
(200000, 768)

# Patent IDs count verification
$ python -c "import json; print(len(json.load(open('data/embeddings/patent_ids.json'))))"
200000
```