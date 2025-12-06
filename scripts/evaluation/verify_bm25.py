"""
Verify that BM25 features are now being computed correctly (non-zero values).
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_extractor import FeatureExtractor
from src.embeddings.patent_sberta import PatentEmbedder
from src.retrieval.bm25_retriever import BM25Retriever


def verify_bm25():
    """Verify BM25 features are computed."""
    print("="*70)
    print("BM25 FEATURE VERIFICATION")
    print("="*70)
    
    # Load sample patents
    patents_path = project_root / 'data' / 'sampled' / 'patents_sampled.jsonl'
    if not patents_path.exists():
        print("✗ Patent data not found")
        return False
    
    print("\n1. Loading sample patents...")
    sample_patents = []
    with open(patents_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Get 5 sample patents
                break
            p = json.loads(line)
            sample_patents.append(p)
    
    print(f"   Loaded {len(sample_patents)} sample patents")
    
    # Load embeddings
    print("\n2. Loading embeddings...")
    embedder = PatentEmbedder()
    embeddings_path = project_root / 'data' / 'embeddings' / 'patent_embeddings.npy'
    patent_ids_path = project_root / 'data' / 'embeddings' / 'patent_ids.json'
    
    if not embeddings_path.exists():
        print("   ✗ Embeddings not found")
        return False
    
    embeddings, patent_ids = embedder.load_embeddings(embeddings_path)
    print(f"   Loaded {embeddings.shape[0]:,} embeddings")
    
    # Load patent IDs from JSON if not in embeddings file
    if patent_ids is None:
        if patent_ids_path.exists():
            with open(patent_ids_path, 'r') as f:
                patent_ids = json.load(f)
        else:
            # Generate from sample patents (for testing)
            patent_ids = [str(p['patent_id']) for p in sample_patents]
    
    # Load BM25 index
    print("\n3. Loading BM25 index...")
    bm25_retriever = BM25Retriever()
    try:
        bm25_retriever.load_index('bm25_index')
        print(f"   ✓ BM25 index loaded: {len(bm25_retriever.patent_ids):,} documents")
        has_bm25 = True
    except Exception as e:
        print(f"   ✗ BM25 index not available: {e}")
        return False
    
    # Initialize feature extractor
    print("\n4. Initializing feature extractor with BM25...")
    # For testing, we only need a small mapping
    patent_id_to_idx = {pid: i for i, pid in enumerate(patent_ids[:len(sample_patents)])}
    feature_extractor = FeatureExtractor(
        embeddings=embeddings,
        patent_id_to_idx=patent_id_to_idx,
        bm25_retriever=bm25_retriever
    )
    print("   ✓ Feature extractor initialized")
    
    # Test BM25 extraction on a pair
    print("\n5. Testing BM25 feature extraction...")
    if len(sample_patents) < 2:
        print("   ✗ Not enough sample patents")
        return False
    
    patent1 = sample_patents[0]
    patent2 = sample_patents[1]
    
    p1_id = str(patent1.get('patent_id', ''))
    p2_id = str(patent2.get('patent_id', ''))
    
    print(f"   Patent 1 ID: {p1_id}")
    print(f"   Patent 2 ID: {p2_id}")
    print(f"   Patent 2 in BM25 index: {p2_id in bm25_retriever.patent_ids}")
    
    fv = feature_extractor.extract_features(patent1, patent2)
    
    bm25_doc = fv.features.get('bm25_doc_score', 0)
    bm25_claim = fv.features.get('bm25_best_claim_score', 0)
    
    print(f"\n   Sample BM25 features:")
    print(f"     bm25_doc_score:       {bm25_doc:.6f}")
    print(f"     bm25_best_claim_score: {bm25_claim:.6f}")
    
    # Try with a patent that's definitely in the BM25 index
    if p2_id not in bm25_retriever.patent_ids:
        print(f"\n   Trying with patent from BM25 index...")
        test_p2_id = bm25_retriever.patent_ids[0]
        print(f"   Using patent ID from index: {test_p2_id}")
        
        # Find this patent in our sample
        test_patent2 = None
        for p in sample_patents:
            if str(p.get('patent_id', '')) == test_p2_id:
                test_patent2 = p
                break
        
        if test_patent2:
            fv2 = feature_extractor.extract_features(patent1, test_patent2)
            bm25_doc2 = fv2.features.get('bm25_doc_score', 0)
            bm25_claim2 = fv2.features.get('bm25_best_claim_score', 0)
            print(f"   With indexed patent:")
            print(f"     bm25_doc_score:       {bm25_doc2:.6f}")
            print(f"     bm25_best_claim_score: {bm25_claim2:.6f}")
            
            if bm25_doc2 > 0 or bm25_claim2 > 0:
                bm25_doc = bm25_doc2
                bm25_claim = bm25_claim2
    
    if bm25_doc > 0 or bm25_claim > 0:
        print("\n" + "="*70)
        print("✓ SUCCESS: BM25 features are now being computed!")
        print("="*70)
        print("\nBM25 features are no longer placeholders (0.0).")
        print("They are now computed using the actual BM25 index.")
        print("\nNOTE: The current model was trained with BM25 = 0.0,")
        print("      so for best results, the model should be retrained.")
        print("      However, the features are correctly implemented.")
        return True
    else:
        print("\n" + "="*70)
        print("✗ FAILED: BM25 features are still zero")
        print("="*70)
        return False


if __name__ == "__main__":
    success = verify_bm25()
    sys.exit(0 if success else 1)

