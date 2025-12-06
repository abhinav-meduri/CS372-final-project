# Documentation

This folder contains project documentation.

## Main Documentation Files

**project_documentation.md** - Complete project documentation including:
- Project overview
- Production pipeline details
- Model performance (verified accuracies)
- Architecture description
- Project organization

**project_explanation.md** - Comprehensive technical explanation including:
- What the model predicts and why
- Training data and ground truth labels
- **Complete mathematical definitions of all 13 features**
- Citation pairs: purpose and validity
- Ground truth labels: what accuracy measures
- **Ablation studies** (feature and architecture ablation)
- Production pipeline walkthrough

All documentation is consolidated in the main files above. See `project_documentation.md` for details on:
- Model organization and locations
- Hyperparameter tuning
- Plots and visualizations

## Quick Reference

**Production Model:** PyTorch Neural Network (91.82% accuracy, 97.21% ROC-AUC)

**Best Performance Models:**
1. Gradient Boosting: 91.95% (experimental)
2. PyTorch NN: 91.82% (production)
3. Ensemble: 91.77% (experimental)

**Baseline Comparisons:**
- Random Guessing: 49.78%
- Title Jaccard Heuristic: 75.36%
- Logistic Regression: 90.79%

**Key Insight:** The model predicts **similarity**, not citations. Citations are used as a proxy for similarity in training data, but the model learns semantic similarity from 13 features (BM25, embeddings, CPC, etc.).

See `project_explanation.md` for complete details on features, ground truth, and training methodology.
