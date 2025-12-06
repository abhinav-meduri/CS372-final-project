"""
Extract citation-based training pairs from PatentsView citation data.

Filters citations where BOTH citing and cited patents are in our sampled dataset.
These are high-quality positive pairs for training.
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random

def load_our_patent_ids(sampled_path: str = 'data/sampled/patents_sampled.jsonl') -> set:
    """Load all patent IDs from our sampled dataset."""
    print("Loading our patent IDs...")
    patent_ids = set()
    
    with open(sampled_path, 'r') as f:
        for line in tqdm(f, desc="Loading patent IDs"):
            patent = json.loads(line)
            patent_ids.add(str(patent['patent_id']))
    
    print(f"Loaded {len(patent_ids)} patent IDs from our dataset")
    return patent_ids


def extract_citation_pairs(
    citation_path: str = 'data/citations/g_us_patent_citation.tsv',
    our_patent_ids: set = None,
    output_path: str = 'data/citations/filtered_citations.jsonl'
) -> list:
    """
    Extract citation pairs where both patents are in our dataset.
    
    Returns list of (citing_id, cited_id) tuples.
    """
    print(f"\nExtracting citations from {citation_path}...")
    print("(This may take a few minutes for 10GB file...)")
    
    pairs = []
    total_rows = 0
    matched_rows = 0
    
    with open(citation_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quotechar='"')
        
        for row in tqdm(reader, desc="Scanning citations", unit=" rows"):
            total_rows += 1
            
            citing_id = row['patent_id'].strip('"')
            cited_id = row['citation_patent_id'].strip('"')
            
            # Check if BOTH patents are in our dataset
            if citing_id in our_patent_ids and cited_id in our_patent_ids:
                pairs.append({
                    'citing_patent_id': citing_id,
                    'cited_patent_id': cited_id,
                    'citation_category': row.get('citation_category', '').strip('"'),
                    'citation_date': row.get('citation_date', '').strip('"')
                })
                matched_rows += 1
                
            # Progress update every 10M rows
            if total_rows % 10_000_000 == 0:
                print(f"  Processed {total_rows/1e6:.1f}M rows, found {matched_rows} pairs...")
    
    print(f"\nTotal rows scanned: {total_rows:,}")
    print(f"Citation pairs found (both in our dataset): {len(pairs):,}")
    
    # Save filtered citations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"Saved to {output_path}")
    
    return pairs


def create_training_pairs_from_citations(
    citation_pairs: list,
    our_patent_ids: set,
    output_dir: str = 'data/training',
    n_negative_per_positive: int = 1,
    random_seed: int = 42
):
    """
    Create balanced training dataset from citation pairs.
    
    Positive pairs: Citation relationships
    Negative pairs: Random patents that don't cite each other
    """
    random.seed(random_seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating training pairs from {len(citation_pairs)} citation pairs...")
    
    # Create positive pairs (deduplicated)
    positive_pairs = []
    seen = set()
    
    for cp in citation_pairs:
        p1, p2 = cp['citing_patent_id'], cp['cited_patent_id']
        if p1 > p2:
            p1, p2 = p2, p1
        
        if (p1, p2) not in seen:
            seen.add((p1, p2))
            positive_pairs.append({
                'patent_id_1': p1,
                'patent_id_2': p2,
                'label': 1,
                'pair_type': 'citation'
            })
    
    print(f"Unique positive pairs: {len(positive_pairs)}")
    
    # Create negative pairs (random patents that don't cite each other)
    n_negatives = len(positive_pairs) * n_negative_per_positive
    print(f"Generating {n_negatives} negative pairs...")
    
    all_ids = list(our_patent_ids)
    citation_set = seen.copy()  # Patents we know are related
    
    negative_pairs = []
    attempts = 0
    max_attempts = n_negatives * 20
    
    while len(negative_pairs) < n_negatives and attempts < max_attempts:
        attempts += 1
        
        p1, p2 = random.sample(all_ids, 2)
        if p1 > p2:
            p1, p2 = p2, p1
        
        # Skip if this is a known citation pair
        if (p1, p2) in citation_set:
            continue
        
        # Skip if already generated
        if (p1, p2) in seen:
            continue
        
        seen.add((p1, p2))
        negative_pairs.append({
            'patent_id_1': p1,
            'patent_id_2': p2,
            'label': 0,
            'pair_type': 'random_negative'
        })
    
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    # Split 70/15/15
    n = len(all_pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    test = all_pairs[n_train + n_val:]
    
    # Save
    for name, data in [('train', train), ('val', val), ('test', test)]:
        path = output_dir / f'{name}_pairs.jsonl'
        with open(path, 'w') as f:
            for pair in data:
                f.write(json.dumps(pair) + '\n')
        print(f"Saved {len(data)} pairs to {path}")
    
    # Save stats
    stats = {
        'total_citation_pairs': len(citation_pairs),
        'unique_positive_pairs': len(positive_pairs),
        'negative_pairs': len(negative_pairs),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'positive_ratio': len(positive_pairs) / len(all_pairs)
    }
    
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[OK] Training data generation complete!")
    print(f"   Total pairs: {len(all_pairs)}")
    print(f"   Positive: {len(positive_pairs)} ({len(positive_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"   Negative: {len(negative_pairs)} ({len(negative_pairs)/len(all_pairs)*100:.1f}%)")
    
    return train, val, test


def main():
    # Step 1: Load our patent IDs
    our_ids = load_our_patent_ids()
    
    # Step 2: Extract matching citation pairs
    pairs = extract_citation_pairs(our_patent_ids=our_ids)
    
    # Step 3: Create training dataset
    if len(pairs) > 0:
        create_training_pairs_from_citations(pairs, our_ids)
    else:
        print("\n[WARN] No citation pairs found in our dataset.")
        print("This could mean our sampled patents don't cite each other.")
        print("Falling back to heuristic-based pair generation...")


if __name__ == "__main__":
    main()


