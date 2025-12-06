"""
Compute all extended features and retrain the model.

This script:
1. Computes claim-level features (3 features)
2. Computes citation-based features (4 features)
3. Combines with base features (12 features)
4. Retrains MLP with all 19 features
5. Evaluates and saves results
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.claim_embeddings import ClaimEmbedder
from src.models.mlp_classifier import PatentNoveltyClassifier

# Set up logging
log_file = f'logs/full_feature_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CitationFeatureExtractor:
    """Extract citation-based features for patent pairs."""
    
    def __init__(self, citations_path: str = 'data/citations/filtered_citations.jsonl'):
        from collections import defaultdict
        self.patent_citations = defaultdict(set)
        self.patent_cited_by = defaultdict(set)
        self._load_citations(citations_path)
    
    def _load_citations(self, path: str):
        if not Path(path).exists():
            tsv_path = 'data/citations/g_us_patent_citation.tsv'
            if Path(tsv_path).exists():
                self._load_from_tsv(tsv_path)
            return
        
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                citing = str(data.get('citing_patent_id', ''))
                cited = str(data.get('cited_patent_id', ''))
                if citing and cited:
                    self.patent_citations[citing].add(cited)
                    self.patent_cited_by[cited].add(citing)
    
    def _load_from_tsv(self, path: str):
        with open(path, 'r') as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.patent_citations[parts[0]].add(parts[1])
                    self.patent_cited_by[parts[1]].add(parts[0])
    
    def extract_features(self, p1_id: str, p2_id: str) -> dict:
        cites1 = self.patent_citations.get(p1_id, set())
        cites2 = self.patent_citations.get(p2_id, set())
        cited_by_1 = self.patent_cited_by.get(p1_id, set())
        cited_by_2 = self.patent_cited_by.get(p2_id, set())
        
        # Citation overlap (Jaccard)
        if cites1 or cites2:
            intersection = len(cites1 & cites2)
            union = len(cites1 | cites2)
            citation_overlap = intersection / union if union > 0 else 0.0
        else:
            citation_overlap = 0.0
        
        # Shared citations (normalized)
        shared = len(cites1 & cites2)
        shared_norm = min(np.log1p(shared) / np.log1p(10), 1.0)
        
        # Co-citation score
        if cited_by_1 or cited_by_2:
            cociting = len(cited_by_1 & cited_by_2)
            norm = np.sqrt(len(cited_by_1) * len(cited_by_2)) if cited_by_1 and cited_by_2 else 1
            cocitation = cociting / norm if norm > 0 else 0.0
        else:
            cocitation = 0.0
        
        # Bibliographic coupling
        if cites1 and cites2:
            shared = len(cites1 & cites2)
            min_cites = min(len(cites1), len(cites2))
            biblio_coupling = shared / min_cites if min_cites > 0 else 0.0
        else:
            biblio_coupling = 0.0
        
        return {
            'citation_overlap': citation_overlap,
            'shared_citations_norm': shared_norm,
            'cocitation_score': cocitation,
            'bibliographic_coupling': biblio_coupling
        }


def load_patents(path: str = 'data/sampled/patents_sampled.jsonl'):
    patents = {}
    with open(path, 'r') as f:
        for line in f:
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    return patents


def load_pairs(split: str):
    path = f'data/training/{split}_pairs.jsonl'
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def compute_claim_features_batch(pairs, patents, embedder):
    """Compute claim features for all pairs."""
    features = []
    for pair in tqdm(pairs, desc="Claim features"):
        p1 = patents.get(str(pair['patent_id_1']))
        p2 = patents.get(str(pair['patent_id_2']))
        
        if not p1 or not p2:
            features.append([0.0, 0.0, 0.0])
            continue
        
        try:
            sims = embedder.compute_claim_similarity(p1, p2)
            features.append([
                sims['max_claim_similarity'],
                sims['mean_claim_similarity'],
                sims['independent_claim_similarity']
            ])
        except Exception:
            features.append([0.0, 0.0, 0.0])
    
    return np.array(features)


def compute_citation_features_batch(pairs, citation_extractor):
    """Compute citation features for all pairs."""
    features = []
    for pair in tqdm(pairs, desc="Citation features"):
        p1_id = str(pair['patent_id_1'])
        p2_id = str(pair['patent_id_2'])
        
        feats = citation_extractor.extract_features(p1_id, p2_id)
        features.append([
            feats['citation_overlap'],
            feats['shared_citations_norm'],
            feats['cocitation_score'],
            feats['bibliographic_coupling']
        ])
    
    return np.array(features)


def main():
    logger.info("=" * 60)
    logger.info("COMPUTING ALL FEATURES AND RETRAINING MODEL")
    logger.info("=" * 60)
    
    features_dir = Path('data/features')
    
    # Load resources
    logger.info("\n1. Loading resources...")
    patents = load_patents()
    logger.info(f"   Loaded {len(patents)} patents")
    
    # Initialize extractors
    logger.info("\n2. Initializing feature extractors...")
    claim_embedder = ClaimEmbedder(batch_size=32)
    citation_extractor = CitationFeatureExtractor()
    logger.info(f"   Citation data: {len(citation_extractor.patent_citations)} patents with citations")
    
    # Feature names
    base_features = [
        'bm25_doc_score', 'bm25_best_claim_score', 'cosine_doc_similarity',
        'cosine_max_claim_similarity', 'embedding_diff_mean', 'embedding_diff_std',
        'cpc_jaccard', 'year_diff', 'title_jaccard', 'abstract_length_ratio',
        'claim_count_ratio', 'shared_rare_terms_ratio'
    ]
    claim_features = ['max_claim_similarity', 'mean_claim_similarity', 'independent_claim_similarity']
    citation_features = ['citation_overlap', 'shared_citations_norm', 'cocitation_score', 'bibliographic_coupling']
    
    all_feature_names = base_features + claim_features + citation_features
    logger.info(f"\n   Total features: {len(all_feature_names)}")
    logger.info(f"   - Base: {len(base_features)}")
    logger.info(f"   - Claim: {len(claim_features)}")
    logger.info(f"   - Citation: {len(citation_features)}")
    
    # Process each split
    all_X = {}
    all_y = {}
    
    for split in ['train', 'val', 'test']:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing {split} split...")
        logger.info(f"{'=' * 40}")
        
        # Load base features
        X_base = np.load(features_dir / f'{split}_features.X.npy')
        y = np.load(features_dir / f'{split}_features.y.npy')
        logger.info(f"Base features: {X_base.shape}")
        
        # Load pairs
        pairs = load_pairs(split)
        logger.info(f"Pairs: {len(pairs)}")
        
        # Compute claim features
        logger.info("Computing claim features...")
        X_claim = compute_claim_features_batch(pairs, patents, claim_embedder)
        logger.info(f"Claim features: {X_claim.shape}")
        
        # Compute citation features
        logger.info("Computing citation features...")
        X_citation = compute_citation_features_batch(pairs, citation_extractor)
        logger.info(f"Citation features: {X_citation.shape}")
        
        # Combine all features
        X_combined = np.hstack([X_base, X_claim, X_citation])
        logger.info(f"Combined features: {X_combined.shape}")
        
        # Save
        np.save(features_dir / f'{split}_features_full.X.npy', X_combined)
        np.save(features_dir / f'{split}_features_full.y.npy', y)
        
        all_X[split] = X_combined
        all_y[split] = y
        
        # Statistics
        logger.info(f"\nFeature statistics for {split}:")
        for i, name in enumerate(claim_features):
            col = X_claim[:, i]
            logger.info(f"  {name}: mean={col.mean():.4f}, std={col.std():.4f}")
        for i, name in enumerate(citation_features):
            col = X_citation[:, i]
            nonzero = (col > 0).sum()
            logger.info(f"  {name}: mean={col.mean():.4f}, nonzero={nonzero} ({nonzero/len(col):.1%})")
    
    # Save feature names
    with open(features_dir / 'feature_names_full.json', 'w') as f:
        json.dump(all_feature_names, f, indent=2)
    
    # Train model
    logger.info(f"\n{'=' * 60}")
    logger.info("TRAINING MODEL WITH ALL FEATURES")
    logger.info(f"{'=' * 60}")
    
    clf = PatentNoveltyClassifier(
        hidden_layer_sizes=(128, 64, 32),  # Larger network for more features
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20
    )
    
    clf.fit(all_X['train'], all_y['train'], all_X['val'], all_y['val'], all_feature_names)
    
    # Evaluate
    logger.info(f"\n{'=' * 60}")
    logger.info("EVALUATION")
    logger.info(f"{'=' * 60}")
    
    results = {}
    for split in ['train', 'val', 'test']:
        metrics = clf.evaluate(all_X[split], all_y[split])
        results[split] = metrics
        logger.info(f"\n{split.upper()} Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  F1:        {metrics['f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
    
    # Feature importance
    logger.info("\nFeature Importance (Top 15):")
    importance = clf.get_feature_importance()
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * int(imp * 40)
        logger.info(f"  {name:35s} {imp:.4f} {bar}")
    
    # Evaluate on hard negatives
    logger.info(f"\n{'=' * 60}")
    logger.info("HARD NEGATIVE EVALUATION")
    logger.info(f"{'=' * 60}")
    
    hard_neg_path = Path('data/training/hard_negatives.jsonl')
    if hard_neg_path.exists():
        hard_negs = []
        with open(hard_neg_path, 'r') as f:
            for line in f:
                hard_negs.append(json.loads(line))
        
        logger.info(f"Computing features for {len(hard_negs)} hard negatives...")
        
        # Load base feature extractor components
        embeddings = np.load('data/embeddings/patent_embeddings.npy')
        with open('data/embeddings/patent_ids.json', 'r') as f:
            patent_ids = json.load(f)
        
        from src.features.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        extractor.set_embeddings(embeddings, patent_ids)
        
        hard_features = []
        for pair in tqdm(hard_negs, desc="Hard negative features"):
            p1 = patents.get(str(pair['patent_id_1']))
            p2 = patents.get(str(pair['patent_id_2']))
            
            if not p1 or not p2:
                hard_features.append(np.zeros(len(all_feature_names)))
                continue
            
            # Base features
            fv = extractor.extract_features(p1, p2, label=0)
            base_feats = fv.to_array(base_features)
            
            # Claim features
            try:
                sims = claim_embedder.compute_claim_similarity(p1, p2)
                claim_feats = [sims['max_claim_similarity'], sims['mean_claim_similarity'], sims['independent_claim_similarity']]
            except:
                claim_feats = [0.0, 0.0, 0.0]
            
            # Citation features
            cit_feats = citation_extractor.extract_features(str(pair['patent_id_1']), str(pair['patent_id_2']))
            cit_feats_arr = [cit_feats['citation_overlap'], cit_feats['shared_citations_norm'], 
                           cit_feats['cocitation_score'], cit_feats['bibliographic_coupling']]
            
            hard_features.append(np.concatenate([base_feats, claim_feats, cit_feats_arr]))
        
        X_hard = np.array(hard_features)
        y_hard_pred = clf.predict(X_hard)
        hard_accuracy = (y_hard_pred == 0).mean()
        
        logger.info(f"\nHard Negatives:")
        logger.info(f"  Correctly classified: {hard_accuracy:.1%}")
        logger.info(f"  False positive rate:  {1-hard_accuracy:.1%}")
        
        results['hard_negative_accuracy'] = float(hard_accuracy)
    
    # Save model and results
    clf.save('models')
    logger.info("\nModel saved to models/")
    
    # Save comprehensive results
    results['feature_importance'] = importance
    results['num_features'] = len(all_feature_names)
    results['feature_names'] = all_feature_names
    
    with open('results/metrics/all_metrics_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to results/metrics/all_metrics_full.json")
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total features: {len(all_feature_names)}")
    logger.info(f"  - Base features: {len(base_features)}")
    logger.info(f"  - Claim features: {len(claim_features)}")
    logger.info(f"  - Citation features: {len(citation_features)}")
    logger.info(f"\nTest Performance:")
    logger.info(f"  Accuracy: {results['test']['accuracy']:.1%}")
    logger.info(f"  ROC-AUC:  {results['test']['roc_auc']:.4f}")
    logger.info(f"  F1 Score: {results['test']['f1']:.4f}")
    if 'hard_negative_accuracy' in results:
        logger.info(f"  Hard Neg Accuracy: {results['hard_negative_accuracy']:.1%}")
    
    logger.info(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()

