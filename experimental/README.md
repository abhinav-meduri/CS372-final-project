# Experimental Models and Scripts

This folder contains **trained models and training scripts that are NOT used in the production inference pipeline**.

## Contents

### `models/` - Trained Model Files

These models were trained and evaluated but are **not loaded** in production:

1. **MLP Classifier** (`mlp_model.pkl`, `scaler.pkl`)
   - Accuracy: 87.35% on test set
   - ROC-AUC: 94.48%
   - Architecture: (64, 32) hidden layers
   - Features: 13 original features

2. **PyTorch Neural Network** (experimental variants)
   - Accuracy: ~91.8% on test set
   - Architecture: Residual blocks, batch norm, dropout
   - Features: Various feature configurations tested
   - Parameters: 102,361 trainable

3. **Ensemble Model** (`ensemble/`)
   - Accuracy: ~92% on test set
   - Method: Stacking (Gradient Boosting + Logistic Regression)
   - Calibration: Platt scaling

### `scripts/` - Training and Evaluation Scripts

1. **`train_mlp.py`** - Train MLP classifier
2. **`train_improved_model.py`** - Train PyTorch model with enhanced features
3. **`train_nn_with_enhanced_features.py`** - Train PyTorch NN with 28 features
4. **`comprehensive_evaluation.py`** - Compare all models (baselines, MLP, PyTorch, ensemble)
5. **`hyperparameter_tuning.py`** - Grid search for MLP hyperparameters
6. **`retrain_with_hard_negatives.py`** - Data augmentation training

## Why Not Used in Production?

The **production pipeline** (`app.py` → `src/app/patent_analyzer.py`) uses:

- Simple similarity scoring: `novelty_score = 1 - max_similarity`
- No trained classifier: Just FAISS similarity search

**Not using**:
- MLP Classifier
- PyTorch Neural Network (Note: Actually now used in production)
- Ensemble Model

### Reasons:

1. **Simplicity**: Current approach is fast and straightforward
2. **Speed**: No model loading or feature extraction overhead
3. **Interpretability**: Direct relationship to embedding similarity

### Trade-offs:

| Approach | Accuracy | Speed | Complexity |
|----------|----------|-------|------------|
| **Current (Similarity)** | ~75-85% | Fast | Simple |
| **MLP Classifier** | 87.35% | Slower | Medium |
| **PyTorch NN** | 91.8% | Slower | Complex |

## How to Use These Models

If you want to integrate a trained model into production:

1. **Move model files back**:
   ```bash
   mv experimental/models/mlp_model.pkl models/
   mv experimental/models/scaler.pkl models/
   ```

2. **Modify `src/app/patent_analyzer.py`**:
   - Load the model in `load()` method
   - Extract features for patent pairs
   - Use `model.predict_proba()` instead of `1 - max_similarity`

3. **See**: `docs/PROJECT_ORGANIZATION.md` for detailed integration steps

## Model Performance Summary

| Model | Test Accuracy | ROC-AUC | Features | Status |
|-------|---------------|---------|----------|--------|
| **Current (Similarity)** | ~75-85%* | N/A | Embeddings only | Production |
| **MLP Classifier** | 87.35% | 94.48% | 13 features | Experimental |
| **Logistic Regression** | 90.79% | 96.76% | 13 features | Experimental |
| **PyTorch NN** | 91.8% | ~97% | 28 features | Production (now used) |
| **Ensemble** | ~92% | ~97% | 28 features | Experimental |

*Estimated based on similarity thresholds

## Evaluation Results

See `results/comprehensive_evaluation/comprehensive_evaluation.json` for:
- Baseline comparisons
- Ablation studies
- Inference time measurements
- Model comparison results

## Future Work

These models could be integrated for:
- Better accuracy (87-92% vs 75-85%)
- More nuanced scoring
- Better handling of edge cases

But would require:
- Feature extraction pipeline
- Model loading overhead
- More complex inference code

