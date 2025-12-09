# Attribution

This document provides attribution for all datasets, models, libraries, external services, and AI assistance used in the Patent Novelty Assessment System.

---

## Dataset Attribution

### PatentsView Public Data (2021-2025)
- **Source:** United States Patent and Trademark Office (USPTO) via PatentsView  
- **URL:** https://patentsview.org/download/data-download-tables  
- **License:** Public domain (U.S. government works)  
- **Files:** Patent abstracts (`g_brf_sum_text_*.tsv.zip`), claims (`g_claims_*.tsv.zip`), and citations (`g_us_patent_citation.tsv`)
- **Usage:** Raw source data for 200,000 sampled patents (2021-2025) used for training, embeddings, and retrieval

### Processed Patent Corpus
- **Derived from:** PatentsView data (above)
- **Files:** `data/sampled/patents_sampled.jsonl`, `data/embeddings/patent_embeddings.npy`, `data/embeddings/patent_ids.json`
- **Usage:** Core patent database for semantic retrieval, feature extraction, and training

---

## Pre-trained Models

### PatentSBERTa
- **Source:** AI-Growth-Lab/PatentSBERTa
- **URL:** https://huggingface.co/AI-Growth-Lab/PatentSBERTa
- **License:** Apache License 2.0
- **Paper:** Aristodemou & Tietze (2018), "The state-of-the-art on Intellectual Property Analytics" (https://arxiv.org/abs/2103.11933)
- **Usage:** Generating semantic embeddings for patent text

### Phi-3 Mini (via Ollama)
- **Source:** Microsoft Research
- **URL:** https://ollama.ai/library/phi3
- **License:** MIT License
- **Paper:** Abdin et al. (2024), "Phi-3 Technical Report" (https://arxiv.org/abs/2404.14219)
- **Usage:** LLM keyword extraction for online search and novelty explanation generation

---

## Custom Trained Models

### PyTorch Neural Network Classifier
- **Files:** `models/pytorch_nn/pytorch_model.pt`, `scaler_pytorch.pkl`
- **Implementation:** `src/app/pytorch_classifier.py`
- **Training:** Trained on 10 engineered features with 3-fold cross-validation

### MLP Baseline Classifier
- **Files:** `models/mlp/mlp_model.pkl`, `scaler.pkl`
- **Usage:** Baseline comparison model

---

## External APIs and Services

### SerpAPI (Google Patents Search)
- **URL:** https://serpapi.com/google-patents-api
- **Package:** `google-search-results`
- **Usage:** Online patent search for hybrid retrieval (complements local 200K corpus)

### Ollama
- **URL:** https://ollama.ai/
- **Usage:** Local LLM runtime for Phi-3 (keyword extraction and explanations)

---

## Libraries and Frameworks

### Core Machine Learning
- **PyTorch** (https://pytorch.org/) - Neural network framework
- **scikit-learn** (https://scikit-learn.org/) - ML utilities, StandardScaler, GridSearchCV
- **sentence-transformers** (https://www.sbert.net/) - PatentSBERTa wrapper
- **transformers** (https://huggingface.co/transformers) - Pre-trained model loading
- **skorch** (https://skorch.readthedocs.io/) - PyTorch-scikit-learn integration

### Data and NLP
- **NumPy** (https://numpy.org/) - Array operations and embeddings
- **pandas** (https://pandas.pydata.org/) - Data loading and processing
- **nltk** (https://www.nltk.org/) - Text preprocessing
- **FAISS** (https://github.com/facebookresearch/faiss) - Vector similarity search
- **rank-bm25** (https://github.com/dorianbrown/rank_bm25) - BM25 ranking

### Application
- **Streamlit** (https://streamlit.io/) - Web application framework (`app.py`)
- **plotly** (https://plotly.com/) - Interactive visualizations
- **matplotlib** (https://matplotlib.org/) - Static plots
- **seaborn** (https://seaborn.pydata.org/) - Statistical visualizations

### Utilities
- **requests** (https://requests.readthedocs.io/) - HTTP client
- **tqdm** (https://tqdm.github.io/) - Progress bars
- **orjson** (https://github.com/ijl/orjson) - Fast JSON parsing
- **python-dotenv** (https://github.com/theskumar/python-dotenv) - Environment variables
- **Jupyter** (https://jupyter.org/) - Notebook environment

---

## AI-Generated and AI-Assisted Code

The overall system architecture, hybrid retrieval design, feature engineering strategy, model selection, and training methodology were my own design decisions. AI tools assisted with implementation details, code refactoring, debugging, and handling edge cases.

### Key Areas of AI Assistance

**Application and Core Pipeline:**
- `app.py` - Streamlit UI components and session state management
- `src/app/patent_analyzer.py` - Core pipeline orchestration and result formatting
- `src/app/phi3_explainer.py` - Ollama API integration and prompt refinement
- `src/app/pytorch_classifier.py` - Neural network implementation details

**Data Preprocessing:**
- `scripts/data/preprocessing/compute_features.py` - Feature extraction with error handling for inconsistent patent data
- `scripts/data/preprocessing/generate_embeddings.py` - Batch processing and memory management for embeddings
- `src/data/loader.py`, `src/data/preprocessor.py` - Handling malformed entries and missing fields

**Training and Evaluation:**
- `scripts/evaluation/tuning/nn_tuning.py` and `nn_tuning.ipynb` - Hyperparameter tuning setup and formatting
- `scripts/evaluation/plots/` - Visualization scripts for ablation, baselines, precision-recall, and metrics
- `scripts/training/` - Training pair generation and citation extraction

**Online Search:**
- `data/api/online_search.py` - SerpAPI integration, error handling, and keyword extraction

**Notebooks:**
- `notebooks/pipeline.ipynb`, `notebooks/pytorch_classifier.ipynb` - Configuration issues, cell organization, formatting

**Key Learning:** Through this project, I learned that patent data processing is extremely challenging due to inconsistent formats across years, missing fields, and varying document structures. AI assistance was invaluable for iterating through edge cases and implementing robust fallback strategies. All strategic decisions (architecture, features, metrics, model design) were my own

---

## Acknowledgments

- **CS 372 (Introduction to Machine Learning) course staff and faculty** at Duke University for project guidance, feedback, and instruction throughout the semester.
- **United States Patent and Trademark Office (USPTO)** for providing free public access to patent data via PatentsView.
- **Hugging Face** for hosting PatentSBERTa and providing the transformers library ecosystem.
- **Microsoft Research** for open-sourcing the Phi-3 language model family.
- **SerpAPI** for providing Google Patents search API with free tier access.
- **Ollama** for creating an excellent local LLM runtime with optimized inference.
- **Meta AI Research** for developing and open-sourcing FAISS for efficient vector similarity search.
- **Open-source maintainers** of PyTorch, scikit-learn, sentence-transformers, Streamlit, NumPy, pandas, and all other libraries used in this project.
- **AI-Growth-Lab** for training and releasing PatentSBERTa as an open-source patent embedding model.
- **PatentsView** team for curating and maintaining high-quality structured patent datasets.

---

*Last Updated: December 2025*
