"""
Build BM25 index from processed patent data.

Usage:
    python scripts/build_bm25_index.py --sample 50000
    python scripts/build_bm25_index.py --years 2024 2025
    python scripts/build_bm25_index.py --all
"""

import argparse
import json
import sys
from pathlib import Path
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from tqdm import tqdm


def load_patents_from_files(
    processed_dir: str = 'data/processed',
    years: list = None,
    sample_per_year: int = None,
    total_sample: int = None
) -> list:
    """Load patents from JSONL files."""
    
    processed_path = Path(processed_dir)
    patent_files = sorted(processed_path.glob('patents_*.jsonl'))
    
    if years:
        patent_files = [f for f in patent_files 
                       if int(f.stem.split('_')[1]) in years]
    
    all_patents = []
    
    for pf in patent_files:
        year = int(pf.stem.split('_')[1])
        print(f"\nLoading {pf.name}...")
        
        count = 0
        with open(pf, 'r') as f:
            for line in tqdm(f, desc=f"Year {year}"):
                if sample_per_year and count >= sample_per_year:
                    break
                if total_sample and len(all_patents) >= total_sample:
                    break
                
                patent = json.loads(line)
                all_patents.append(patent)
                count += 1
        
        print(f"  Loaded {count} patents from {year}")
        
        if total_sample and len(all_patents) >= total_sample:
            break
        
        # Memory management
        gc.collect()
    
    return all_patents


def main():
    parser = argparse.ArgumentParser(description='Build BM25 index')
    parser.add_argument('--sample', type=int, default=None,
                       help='Total number of patents to sample')
    parser.add_argument('--sample-per-year', type=int, default=None,
                       help='Patents to sample per year')
    parser.add_argument('--years', nargs='+', type=int, default=None,
                       help='Specific years to include')
    parser.add_argument('--all', action='store_true',
                       help='Use all patents (warning: high memory)')
    parser.add_argument('--index-name', type=str, default='bm25_index',
                       help='Name for the saved index')
    
    args = parser.parse_args()
    
    # Default: sample 50k patents
    if not args.all and not args.sample and not args.sample_per_year:
        args.sample = 50000
    
    print("=== Building BM25 Index ===")
    print(f"Sample total: {args.sample}")
    print(f"Sample per year: {args.sample_per_year}")
    print(f"Years: {args.years or 'all'}")
    
    # Load patents
    patents = load_patents_from_files(
        years=args.years,
        sample_per_year=args.sample_per_year,
        total_sample=args.sample
    )
    
    print(f"\nTotal patents loaded: {len(patents)}")
    
    # Build index
    retriever = BM25Retriever()
    retriever.build_index(patents)
    
    # Save index
    retriever.save_index(args.index_name)
    
    # Print stats
    stats = retriever.get_statistics()
    print(f"\nIndex Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Test search
    print("\n=== Test Search ===")
    test_query = "machine learning neural network deep learning artificial intelligence"
    results = retriever.search(test_query, top_k=5)
    
    print(f"Query: '{test_query}'")
    print("\nTop 5 results:")
    for r in results:
        print(f"  {r.rank}. {r.patent_id} (score: {r.score:.2f})")
        if r.text_snippet:
            print(f"     {r.text_snippet[:100]}...")
    
    print("\n[OK] BM25 index built and saved successfully!")


if __name__ == "__main__":
    main()


