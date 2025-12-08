# Ensemble Models

This folder contains the trained ensemble model artifacts (not used in the final pipeline).

## Contents

**Ensemble Model** (`ensemble_model.pkl`)
   - Accuracy: 91.58% on test set
   - ROC-AUC: 0.9713
   - F1: 0.9148
   - Method: Stacking (Gradient Boosting + Logistic Regression)
   - Calibration: Platt scaling
   - Base models: PyTorch NN, Gradient Boosting

## Model Performance

The ensemble model combines predictions from:
1. **PyTorch Neural Network** (256-128 architecture with residual blocks)
2. **Gradient Boosting** (sklearn)

### Performance Comparison:

| Model | Test Accuracy | ROC-AUC | F1 | Status |
|-------|---------------|---------|----|--------|
| **PyTorch NN** | 91.71% | 0.9717 | 0.9156 | Used in Final Pipeline |
| **Ensemble (PyTorch + GB)** | 91.58% | 0.9713 | 0.9148 | Experimental |

## Why Not Pursued Further?

The ensemble model was evaluated but did not provide significant improvement over the single PyTorch model. In fact, the ensemble (91.58% accuracy) performed slightly worse than the standalone PyTorch model (91.71% accuracy). The marginal improvement in ROC-AUC (0.9713 vs 0.9717) was not sufficient to justify the added complexity of:

- Loading and managing multiple models
- Increased inference time
- More complex codebase maintenance
- Additional memory requirements

The production system uses the PyTorch Neural Network directly, which ultimatley proved to be better peforming and maintainable. 

## Model Files

- `ensemble_model.pkl` - Trained ensemble model
- `scaler.pkl` - Feature scaler used during training of the ensemble model.

## Evaluation Results

See `../results/ensemble_metrics.json` for detailed evaluation metrics.

