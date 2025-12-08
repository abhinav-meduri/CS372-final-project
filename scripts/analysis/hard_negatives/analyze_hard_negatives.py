"""
Analyze Hard Negatives Performance

Tests model performance on the generated hard negative pairs
to evaluate robustness on challenging cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.app.pytorch_classifier import PyTorchPatentClassifier
from src.models.mlp_classifier import PatentNoveltyClassifier
from src.features.feature_extract import FeatureExtractor, FeatureVector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_patents(patent_ids):
    """Load patent data for given IDs."""
    patents_path = Path("data/sampled/patents_sampled.jsonl")
    patents = {}
    
    print(f"Loading patents for {len(patent_ids)} IDs...")
    with open(patents_path, 'rb') as f:
        for line in tqdm(f, desc="Loading patents"):
            try:
                patent = json.loads(line)
                pid = str(patent.get('patent_id', ''))
                if pid in patent_ids:
                    patents[pid] = patent
            except:
                continue
    
    return patents

def extract_features_for_pairs(hard_negatives, patents, feature_extractor, st_model):
    """Extract features for hard negative pairs."""
    print("\nExtracting features for hard negative pairs...")
    
    X = []
    y = []
    valid_pairs = []
    
    for pair in tqdm(hard_negatives, desc="Extracting features"):
        pid1 = str(pair['patent_id_1'])
        pid2 = str(pair['patent_id_2'])
        
        if pid1 not in patents or pid2 not in patents:
            continue
        
        p1 = patents[pid1]
        p2 = patents[pid2]
        
        try:
            feature_vector = feature_extractor.extract_features(p1, p2)
            # Convert to array using base feature names (10 features)
            features_10 = feature_vector.to_array(FeatureExtractor.BASE_FEATURE_NAMES)
            
            X.append(features_10)
            y.append(0)  # Hard negatives are negative pairs (label=0)
            valid_pairs.append(pair)
        except Exception as e:
            print(f"Error extracting features for pair {pid1}-{pid2}: {e}")
            continue
    
    return np.array(X), np.array(y), valid_pairs

def main():
    print("HARD NEGATIVES ANALYSIS")
    
    hard_negatives_file = Path("data/features/test_hard_negatives.json")
    
    if not hard_negatives_file.exists():
        print(f"\n❌ Hard negatives file not found: {hard_negatives_file}")
        print("   Please run scripts/analysis/generate_test_hard_negatives.py first")
        return
    
    print("\n1. Loading hard negatives...")
    with open(hard_negatives_file, "r") as f:
        hard_negatives = json.load(f)
    
    print(f"   Loaded {len(hard_negatives)} hard negative pairs")
    
    # Extract unique patent IDs
    patent_ids = set()
    for pair in hard_negatives:
        patent_ids.add(str(pair['patent_id_1']))
        patent_ids.add(str(pair['patent_id_2']))
    
    print(f"   Unique patents: {len(patent_ids)}")
    
    print("\n2. Loading patents...")
    patents = load_patents(patent_ids)
    print(f"   Loaded {len(patents)} patents")
    
    print("\n3. Initializing feature extractor...")
    st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    feature_extractor = FeatureExtractor(st_model)
    
    print("\n4. Extracting features...")
    X, y, valid_pairs = extract_features_for_pairs(hard_negatives, patents, feature_extractor, st_model)
    
    print(f"\n   Extracted features for {len(X)} pairs")
    print(f"   Feature shape: {X.shape}")
    
    print("\n5. Loading models...")
    pytorch_model = PyTorchPatentClassifier()
    pytorch_model.load(Path("models/pytorch_nn"))
    
    # Check MLP model location - skip for now due to scaler issues
    mlp_model = None
    print("   ⚠️  Skipping MLP evaluation (scaler loading issue)")
    mlp_pred = None
    mlp_proba = None
    mlp_metrics = None
    
    print("\n6. Evaluating models on hard negatives...")
    
    # Model expects 10 features (already extracted correctly)
    # X already has shape (n_samples, 10)
    pytorch_pred = pytorch_model.predict(X)
    pytorch_proba = pytorch_model.predict_proba(X)[:, 1]
    
    # Analyze probabilities to understand model confidence
    print(f"\n   Probability Analysis:")
    print(f"     Min probability (Novel): {np.min(pytorch_proba):.4f}")
    print(f"     Max probability (Novel): {np.max(pytorch_proba):.4f}")
    print(f"     Mean probability (Novel): {np.mean(pytorch_proba):.4f}")
    print(f"     Median probability (Novel): {np.median(pytorch_proba):.4f}")
    print(f"     Pairs with prob > 0.5 (predicted Novel): {np.sum(pytorch_proba > 0.5)}")
    print(f"     Pairs with prob < 0.3 (confident Not Novel): {np.sum(pytorch_proba < 0.3)}")
    print(f"     Pairs with prob 0.3-0.5 (uncertain): {np.sum((pytorch_proba >= 0.3) & (pytorch_proba < 0.5))}")
    
    pytorch_metrics = {
        "accuracy": accuracy_score(y, pytorch_pred),
        "precision": precision_score(y, pytorch_pred, zero_division=0),
        "recall": recall_score(y, pytorch_pred, zero_division=0),
        "f1": f1_score(y, pytorch_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, pytorch_proba),
        "confusion_matrix": confusion_matrix(y, pytorch_pred).tolist()
    }
    
    # MLP predictions - also need to pad to 13 features
    if mlp_model is not None:
        mlp_pred = mlp_model.predict(X_padded)
        mlp_proba = mlp_model.predict_proba(X_padded)[:, 1]
        
        mlp_metrics = {
            "accuracy": accuracy_score(y, mlp_pred),
            "precision": precision_score(y, mlp_pred, zero_division=0),
            "recall": recall_score(y, mlp_pred, zero_division=0),
            "f1": f1_score(y, mlp_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, mlp_proba),
            "confusion_matrix": confusion_matrix(y, mlp_pred).tolist()
        }
    else:
        mlp_pred = None
        mlp_proba = None
        mlp_metrics = None
    
    print("HARD NEGATIVES ANALYSIS RESULTS")
    
    print(f"\n📊 PyTorch Model Performance on Hard Negatives:")
    print(f"   Accuracy:  {pytorch_metrics['accuracy']:.4f} ({pytorch_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {pytorch_metrics['precision']:.4f}")
    print(f"   Recall:    {pytorch_metrics['recall']:.4f}")
    print(f"   F1 Score:  {pytorch_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {pytorch_metrics['roc_auc']:.6f}")
    print(f"   Confusion Matrix:")
    cm = np.array(pytorch_metrics['confusion_matrix'])
    if cm.shape == (2, 2):
        print(f"     TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"     FN: {cm[1][0]}, TP: {cm[1][1]}")
    else:
        print(f"     Matrix shape: {cm.shape}")
        print(f"     Matrix: {cm}")
    
    if mlp_metrics:
        print(f"\n📊 MLP Model Performance on Hard Negatives:")
        print(f"   Accuracy:  {mlp_metrics['accuracy']:.4f} ({mlp_metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {mlp_metrics['precision']:.4f}")
        print(f"   Recall:    {mlp_metrics['recall']:.4f}")
        print(f"   F1 Score:  {mlp_metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {mlp_metrics['roc_auc']:.6f}")
        print(f"   Confusion Matrix:")
        cm = mlp_metrics['confusion_matrix']
        print(f"     TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"     FN: {cm[1][0]}, TP: {cm[1][1]}")
    else:
        print(f"\n📊 MLP Model: Not available")
    
    # Analysis
    print("ANALYSIS")
    
    # Count false positives (predicted as novel when they're not)
    cm = np.array(pytorch_metrics['confusion_matrix'])
    if cm.shape == (2, 2):
        pytorch_fp = cm[0][1]
        pytorch_tn = cm[0][0]
    elif cm.shape == (1, 1):
        # All predictions are the same class
        pytorch_tn = cm[0][0]
        pytorch_fp = 0
    else:
        pytorch_tn = 0
        pytorch_fp = 0
    
    print(f"\n🔍 Key Findings:")
    print(f"   - Hard negatives are pairs with high similarity (>0.85) but different semantics")
    print(f"   - These should be classified as 'Not Novel' (label=0)")
    print(f"   - False Positives (FP) indicate the model incorrectly predicts 'Novel'")
    print(f"\n   PyTorch True Negatives: {pytorch_tn}/{len(y)} ({pytorch_tn/len(y)*100:.1f}%)")
    print(f"   PyTorch False Positives: {pytorch_fp}/{len(y)} ({pytorch_fp/len(y)*100:.1f}%)")
    
    if mlp_metrics:
        mlp_fp = mlp_metrics['confusion_matrix'][0][1]
        print(f"   MLP False Positives: {mlp_fp}/{len(y)} ({mlp_fp/len(y)*100:.1f}%)")
    
    if pytorch_metrics['accuracy'] < 0.7:
        print(f"\n   ⚠️  PyTorch model shows lower accuracy on hard negatives")
        print(f"      This suggests the model may struggle with semantically similar but distinct patents")
    else:
        print(f"\n   ✅ PyTorch model maintains reasonable performance on hard negatives")
    
    if mlp_metrics and mlp_metrics['accuracy'] < 0.7:
        print(f"\n   ⚠️  MLP model shows lower accuracy on hard negatives")
    elif mlp_metrics:
        print(f"\n   ✅ MLP model maintains reasonable performance on hard negatives")
    
    # Save results
    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "num_pairs": len(X),
        "pytorch": pytorch_metrics,
        "pytorch_probability_stats": {
            "min": float(np.min(pytorch_proba)),
            "max": float(np.max(pytorch_proba)),
            "mean": float(np.mean(pytorch_proba)),
            "median": float(np.median(pytorch_proba)),
            "pairs_predicted_novel": int(np.sum(pytorch_proba > 0.5)),
            "pairs_confident_not_novel": int(np.sum(pytorch_proba < 0.3)),
            "pairs_uncertain": int(np.sum((pytorch_proba >= 0.3) & (pytorch_proba < 0.5)))
        },
        "summary": {
            "pytorch_fp_rate": pytorch_fp / len(y) if len(y) > 0 else 0,
            "pytorch_tn_rate": pytorch_tn / len(y) if len(y) > 0 else 0,
            "pytorch_accuracy": pytorch_metrics['accuracy']
        }
    }
    
    if mlp_metrics:
        results["mlp"] = mlp_metrics
        mlp_fp = mlp_metrics['confusion_matrix'][0][1]
        results["summary"]["mlp_fp_rate"] = mlp_fp / len(y)
        results["summary"]["mlp_accuracy"] = mlp_metrics['accuracy']
    
    with open(results_dir / "hard_negatives_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir / 'hard_negatives_analysis.json'}")
    
    print("ANALYSIS COMPLETE")

if __name__ == '__main__':
    main()

