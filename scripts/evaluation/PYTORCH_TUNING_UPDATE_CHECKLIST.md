# PyTorch Hyperparameter Tuning - Post-Completion Update Checklist

**Status:** ⏳ PyTorch tuning in progress (monitor with `tail -f logs/pytorch_tuning.log`)

**When tuning completes:** If PyTorch model performance improves, update the following:

## 📊 Plots to Regenerate

### 1. **Baseline Comparison Plot**
- **File:** `scripts/evaluation/plots/plot_baseline_comparison.py`
- **Output:** `results/plots/baseline/baseline_comparison.png`
- **Action:** Re-run script to update PyTorch metrics
- **Command:** `python scripts/evaluation/plots/plot_baseline_comparison.py`

### 2. **PyTorch Model Plots** (if retrained with new hyperparameters)
- **Files:** `notebooks/pytorch_classifier.ipynb` (regenerate plots)
- **Outputs:**
  - `results/plots/pytorch_nn/confusion_matrix.png`
  - `results/plots/pytorch_nn/roc_curve.png`
  - `results/plots/pytorch_nn/training_curve.png`
- **Action:** Re-run notebook cells that generate plots, or use existing plot generation code

### 3. **Ablation Study Plot** (if retrained)
- **File:** `scripts/evaluation/plots/plot_ablation.py`
- **Output:** `results/plots/ablation/ablation_study_comparison.png`
- **Action:** Re-run ablation study if model architecture changes, then regenerate plot
- **Command:** `python scripts/evaluation/plots/plot_ablation.py`

## 📈 Metrics Files to Update

### 1. **PyTorch Metrics JSON**
- **File:** `results/pytorch_nn/pytorch_metrics.json`
- **Action:** Automatically updated when model is retrained and evaluated
- **Check:** Verify test metrics reflect new performance

### 2. **Ensemble Metrics** (if PyTorch improves)
- **File:** `results/ensemble/ensemble_metrics.json`
- **Action:** Retrain ensemble model with updated PyTorch model
- **Note:** Ensemble combines MLP + PyTorch, so if PyTorch improves, ensemble should improve too

## 🔄 Model Files to Update

### 1. **PyTorch Model Checkpoint**
- **Location:** `models/pytorch_nn/`
- **Files:**
  - `model.pth` (model weights)
  - `training_results.json` (training metrics)
  - `training_history_pytorch.json` (training history)
- **Action:** Save new model after retraining with best hyperparameters

### 2. **Hyperparameter Results**
- **File:** `results/hyperparameter_tuning/pytorch/hyperparameter_results.json`
- **Action:** Already saved by tuning script
- **Check:** Review best hyperparameters found

## 📝 Documentation to Update

### 1. **README.md**
- **Section:** Model performance metrics
- **Action:** Update PyTorch model performance numbers if improved
- **Location:** Any section mentioning PyTorch accuracy/ROC-AUC

### 2. **Model Comparison Documentation**
- **Files:** Any docs comparing model performance
- **Action:** Update comparison tables/charts with new PyTorch metrics

## 🔍 Verification Steps

After updates, verify:

1. **Baseline comparison plot shows updated PyTorch metrics**
   ```bash
   python scripts/evaluation/plots/plot_baseline_comparison.py
   ```

2. **PyTorch metrics JSON has latest test performance**
   ```bash
   cat results/pytorch_nn/pytorch_metrics.json | grep -A 5 "test"
   ```

3. **All plots are regenerated with consistent styling**
   - Check color schemes match (neutral teal-blue)
   - Verify all value labels are correct

4. **Ensemble model retrained** (if PyTorch improved significantly)
   - Run `notebooks/ensemble_model.ipynb` to retrain with new PyTorch model

## 🎯 Decision Point

**If PyTorch tuning improves performance:**
- ✅ Update all plots and metrics
- ✅ Consider retraining ensemble
- ✅ Update documentation

**If PyTorch tuning doesn't improve (or degrades):**
- ⚠️ Keep current model
- ⚠️ Update baseline comparison plot with final tuning results
- ⚠️ Document that tuning was attempted but didn't improve performance

## 📋 Quick Update Script

After tuning completes, run:

```bash
# 1. Check tuning results
cat results/hyperparameter_tuning/pytorch/hyperparameter_results.json | grep -A 10 "best"

# 2. Retrain PyTorch with best hyperparameters (if better)
# Edit notebooks/pytorch_classifier.ipynb with best params, then run

# 3. Regenerate baseline comparison
python scripts/evaluation/plots/plot_baseline_comparison.py

# 4. Regenerate PyTorch plots (if retrained)
# Run plot generation cells in notebooks/pytorch_classifier.ipynb

# 5. Retrain ensemble (if PyTorch improved)
# Run notebooks/ensemble_model.ipynb
```

---

**Last Updated:** 2025-12-07  
**Tuning Status:** Monitor `logs/pytorch_tuning.log` for completion

