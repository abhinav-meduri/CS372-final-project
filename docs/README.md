# Documentation

This folder contains project documentation and demo images.

## Demo Images

- `demo_novelty_assessment.png` - Screenshot of the novelty assessment interface
- `demo_analysis_pipeline.png` - Screenshot of the analysis pipeline

## Quick Reference

**Production Model:** PyTorch Neural Network (91.73% accuracy, 97.20% ROC-AUC on test set)

**Baseline Comparisons:**
- Random Guessing: 49.78%
- Cosine Similarity Heuristic: 84.27%
- Logistic Regression: 90.93%

**Key Insight:** The model predicts **similarity**, not citations. Citations are used as a proxy for similarity in training data, but the model learns semantic similarity from 10 features (reduced from 13 via ablation study).

For complete project documentation, see the main [README.md](../README.md) and [SETUP.md](../SETUP.md).
