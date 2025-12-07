# Ensemble Models

This folder contains the **ensemble model** implementation and trained artifacts.

## Contents

### `models/ensemble/` - Trained Ensemble Model

The ensemble model trained and evaluated:

**Ensemble Model** (`models/ensemble/ensemble_model.pkl`)
   - Accuracy: 91.77% on test set
   - ROC-AUC: 0.9716
   - F1: 0.9160
   - Method: Stacking (Gradient Boosting + Logistic Regression)
   - Calibration: Platt scaling
   - Base models: MLP, PyTorch NN, Gradient Boosting

### `src/models/` - Ensemble Model Implementation

**`ensemble_model.py`** - CalibratedEnsemble class
   - Implements stacking ensemble with probability calibration
   - Combines predictions from multiple base models
   - Uses Platt scaling for probability calibration

## Model Performance

The ensemble model combines predictions from:
1. **MLP Classifier** (64-32 architecture)
2. **PyTorch Neural Network** (128-64-32 with residual blocks)
3. **Gradient Boosting** (sklearn)

### Performance Comparison:

| Model | Test Accuracy | ROC-AUC | F1 | Status |
|-------|---------------|---------|----|--------|
| **MLP Classifier** | 91.60% | 0.9709 | 0.9142 | Production |
| **PyTorch NN** | ~91.8% | ~0.97 | ~0.91 | Production |
| **Ensemble** | 91.77% | 0.9716 | 0.9160 | Available |

## How to Use the Ensemble Model

To integrate the ensemble model into production:

1. **Load the model**:
   ```python
   from models.ensemble.src.models.ensemble_model import CalibratedEnsemble
   ensemble = CalibratedEnsemble.load('models/ensemble/models/ensemble')
   ```

2. **Use in `src/app/patent_analyzer.py`**:
   - Load the ensemble model in `load()` method
   - Extract features for patent pairs
   - Use `ensemble.predict_proba()` for novelty scoring

## Model Files

- `models/ensemble/ensemble_model.pkl` - Trained ensemble model
- `src/models/ensemble_model.py` - Ensemble implementation class

## Evaluation Results

See `results/ablation_study/ablation_results.json` for:
- Ablation studies
- Feature importance analysis

## Future Work

These models could be integrated for:
- Better accuracy (87-92% vs 75-85%)
- More nuanced scoring
- Better handling of edge cases

But would require:
- Feature extraction pipeline
- Model loading overhead
- More complex inference code

