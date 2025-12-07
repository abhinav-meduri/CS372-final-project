"""
Generate PatentSBERTa Embeddings for Diverse Patent Sample

This script:
1. Loads the 200K diverse patent sample
2. Generates 768-dim embeddings using PatentSBERTa
3. Saves embeddings and patent IDs for later use
4. Builds FAISS index for fast similarity search

Usage:
    python scripts/generate_embeddings.py
"""

import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import time
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.patent_sberta import PatentEmbedder


def load_sampled_patents(sample_file: str = 'data/sampled/patents_sampled.jsonl'):
    """Load the diverse patent sample."""
    patents = []
    print(f"Loading patents from {sample_file}...")
    
    with open(sample_file, 'r') as f:
        for line in tqdm(f, desc="Loading"):
            patents.append(json.loads(line))
    
    print(f"Loaded {len(patents):,} patents")
    return patents


def get_patent_text(patent: dict) -> str:
    """
    Extract text for embedding from a patent.
    
    Uses abstract/summary only (512 tokens max for PatentSBERTa).
    Keeping it short for faster processing.
    """
    # Abstract or summary only - keep it short for speed
    if patent.get('abstract'):
        return patent['abstract'][:500]  # ~100 tokens
    elif patent.get('summary'):
        return patent['summary'][:500]
    
    # Fallback to first claim if no abstract
    claims = patent.get('claims', [])
    if claims:
        if isinstance(claims[0], dict):
            return claims[0].get('text', '')[:500]
        elif isinstance(claims[0], str):
            return claims[0][:500]
    
    return ""


def generate_embeddings(
    patents: list,
    batch_size: int = 64,
    output_dir: str = 'data/embeddings'
):
    """Generate embeddings for all patents."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoder
    encoder = PatentEmbedder(batch_size=batch_size)
    
    # Prepare texts
    print("\nPreparing texts...")
    patent_ids = []
    texts = []
    
    for patent in tqdm(patents, desc="Extracting text"):
        pid = str(patent['patent_id'])
        text = get_patent_text(patent)
        
        if text.strip():  # Only include patents with text
            patent_ids.append(pid)
            texts.append(text)
    
    print(f"Prepared {len(texts):,} patents with text")
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch_size={batch_size})...")
    start_time = time.time()
    
    embeddings = encoder.encode(
        texts,
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    rate = len(texts) / elapsed
    
    print(f"\nEmbedding generation complete!")
    print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Rate: {rate:.1f} patents/second")
    print(f"  Shape: {embeddings.shape}")
    
    # Save embeddings
    embeddings_file = output_path / 'patent_embeddings.npy'
    print(f"\nSaving embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)
    
    # Save patent IDs
    ids_file = output_path / 'patent_ids.json'
    print(f"Saving patent IDs to {ids_file}...")
    with open(ids_file, 'w') as f:
        json.dump(patent_ids, f)
    
    # Save metadata
    metadata = {
        'num_patents': len(patent_ids),
        'embedding_dim': embeddings.shape[1],
        'model': 'AI-Growth-Lab/PatentSBERTa',
        'generation_time_seconds': elapsed,
        'patents_per_second': rate
    }
    
    with open(output_path / 'embedding_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Memory info
    mem_mb = embeddings.nbytes / 1024 / 1024
    print(f"\nMemory usage: {mem_mb:.1f} MB")
    
    return embeddings, patent_ids


def build_faiss_index(
    embeddings: np.ndarray,
    output_dir: str = 'data/embeddings'
):
    """Build FAISS index for fast similarity search."""
    try:
        import faiss
    except ImportError:
        print("\nFAISS not installed. Skipping index creation.")
        print("Install with: pip install faiss-cpu")
        return None
    
    output_path = Path(output_dir)
    
    print("\nBuilding FAISS index...")
    start_time = time.time()
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    
    # Create index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
    index.add(embeddings_normalized)
    
    elapsed = time.time() - start_time
    print(f"Index built in {elapsed:.1f} seconds")
    print(f"Index contains {index.ntotal:,} vectors")
    
    # Save index
    index_file = output_path / 'faiss_index.bin'
    print(f"Saving FAISS index to {index_file}...")
    faiss.write_index(index, str(index_file))
    
    return index


def test_similarity_search(
    index,
    embeddings: np.ndarray,
    patent_ids: list,
    patents: list,
    num_queries: int = 3
):
    """Test the similarity search with a few examples."""
    import faiss
    
    print("\n" + "="*60)
    print("SIMILARITY SEARCH TEST")
    print("="*60)
    
    # Normalize for search
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    
    # Create patent lookup
    id_to_patent = {str(p['patent_id']): p for p in patents}
    
    # Test with random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(patent_ids), num_queries, replace=False)
    
    for qi in query_indices:
        query_id = patent_ids[qi]
        query_patent = id_to_patent.get(query_id, {})
        query_text = get_patent_text(query_patent)[:200]
        
        print(f"\nQuery: {query_id}")
        print(f"Text: {query_text}...")
        
        # Search
        query_vec = embeddings_normalized[qi:qi+1]
        scores, indices = index.search(query_vec, 6)  # Top 5 + self
        
        print(f"\nTop 5 similar patents:")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == qi:  # Skip self
                continue
            
            similar_id = patent_ids[idx]
            similar_patent = id_to_patent.get(similar_id, {})
            similar_text = get_patent_text(similar_patent)[:100]
            
            print(f"  {rank}. {similar_id} (similarity: {score:.3f})")
            print(f"     {similar_text}...")


def main():
    print("="*60)
    print("PATENT EMBEDDING GENERATION")
    print("="*60)
    
    # Load patents
    patents = load_sampled_patents()
    
    # Generate embeddings
    embeddings, patent_ids = generate_embeddings(patents, batch_size=64)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Test similarity search
    if index is not None:
        test_similarity_search(index, embeddings, patent_ids, patents)
    
    print("\n" + "="*60)
    print("[OK] EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - data/embeddings/patent_embeddings.npy ({embeddings.shape})")
    print(f"  - data/embeddings/patent_ids.json ({len(patent_ids):,} IDs)")
    print(f"  - data/embeddings/faiss_index.bin (similarity search)")
    print(f"  - data/embeddings/embedding_metadata.json")
    
    # Cleanup
    gc.collect()


if __name__ == "__main__":
    main()

