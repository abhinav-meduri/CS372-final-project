# Ensemble Model (Experimental)

This folder contains the ensemble model implementation and results that were evaluated but **not used in production**.

## Contents

- `ensemble.ipynb` - Jupyter notebook for training the ensemble model
- `models/` - Trained ensemble model files
- `results/` - Evaluation metrics and results

## Why Not Pursued Further?

The ensemble model (PyTorch + Gradient Boosting) was evaluated but did not provide significant improvement over the single PyTorch model. In fact, the ensemble performed slightly worse (91.58% vs 91.71% accuracy). The marginal difference in performance did not justify the added complexity of managing multiple models, increased inference time, and more complex codebase maintenance. The production system uses the PyTorch Neural Network directly, which provides the best balance of performance, simplicity, and maintainability.

## Model Performance

- **Ensemble (PyTorch + GB):** 91.58% accuracy, 0.9713 ROC-AUC, 0.9148 F1
- **Single PyTorch:** 91.71% accuracy, 0.9717 ROC-AUC, 0.9156 F1

The ensemble performed slightly worse than the single PyTorch model, so it was not used in production.

## Note

This is kept for reference and potential future experimentation, but is not part of the production pipeline.

