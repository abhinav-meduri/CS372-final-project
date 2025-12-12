"""Extract citation pairs from PatentsView data for training."""

import json
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random

def load_our_patent_ids(sampled_path: str = 'data/sampled/patents_sampled.jsonl') -> set:
    """Load patent IDs from sampled dataset."""
    patent_ids = set()
    with open(sampled_path, 'r') as f:
        for line in tqdm(f, desc="Loading patent IDs"):
            patent = json.loads(line)
            patent_ids.add(str(patent['patent_id']))
    print(f"Loaded {len(patent_ids)} patent IDs")
    return patent_ids


def extract_citation_pairs(
    citation_path: str = 'data/citations/g_us_patent_citation.tsv',
    our_patent_ids: set = None,
    output_path: str = 'data/citations/filtered_citations.jsonl'
) -> list:
    """Extract citation pairs where both patents are in our dataset."""
    print(f"Extracting citations from {citation_path}...")
    
    pairs = []
    total_rows = 0
    matched_rows = 0
    
    with open(citation_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quotechar='"')
        
        for row in tqdm(reader, desc="Scanning citations", unit=" rows"):
            total_rows += 1
            
            citing_id = row['patent_id'].strip('"')
            cited_id = row['citation_patent_id'].strip('"')
            
            # Keep only citations between patents in our dataset
            if citing_id in our_patent_ids and cited_id in our_patent_ids:
                pairs.append({
                    'citing_patent_id': citing_id,
                    'cited_patent_id': cited_id,
                    'citation_category': row.get('citation_category', '').strip('"'),
                    'citation_date': row.get('citation_date', '').strip('"')
                })
                matched_rows += 1
    
    print(f"Total rows: {total_rows:,}, Citation pairs found: {len(pairs):,}")
    
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
    """Create positive (citation) and negative (random) training pairs."""
    random.seed(random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating training pairs from {len(citation_pairs)} citations...")
    
    # Create positive pairs from citations (label=1)
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
    
    print(f"Positive pairs: {len(positive_pairs)}")
    
    # Generate random negative pairs (label=0)
    n_negatives = len(positive_pairs) * n_negative_per_positive
    print(f"Generating {n_negatives} negative pairs...")
    
    all_ids = list(our_patent_ids)
    citation_set = seen.copy()
    negative_pairs = []
    attempts = 0
    max_attempts = n_negatives * 20
    
    while len(negative_pairs) < n_negatives and attempts < max_attempts:
        attempts += 1
        p1, p2 = random.sample(all_ids, 2)
        if p1 > p2:
            p1, p2 = p2, p1
        if (p1, p2) in citation_set or (p1, p2) in seen:
            continue
        seen.add((p1, p2))
        negative_pairs.append({
            'patent_id_1': p1,
            'patent_id_2': p2,
            'label': 0,
            'pair_type': 'random_negative'
        })
    
    print(f"Negative pairs: {len(negative_pairs)}")
    
    # Shuffle and split into train/val/test
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    test = all_pairs[n_train + n_val:]
    
    for name, data in [('train', train), ('val', val), ('test', test)]:
        path = output_dir / f'{name}_pairs.jsonl'
        with open(path, 'w') as f:
            for pair in data:
                f.write(json.dumps(pair) + '\n')
        print(f"Saved {len(data)} {name} pairs to {path}")
    
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
    
    print(f"Training data complete: {len(all_pairs)} pairs ({len(positive_pairs)/len(all_pairs)*100:.1f}% positive)")
    return train, val, test


def main():
    our_ids = load_our_patent_ids()
    pairs = extract_citation_pairs(our_patent_ids=our_ids)
    if len(pairs) > 0:
        create_training_pairs_from_citations(pairs, our_ids)
    else:
        print("No citation pairs found. Use heuristic-based generation instead.")


if __name__ == "__main__":
    main()


