"""
Diverse Patent Sampling Script

Creates a stratified random sample of patents ensuring:
1. Equal representation across all years (temporal diversity)
2. Random sampling within each year (avoids sequential bias)
3. Consistent reproducibility via random seed

Usage:
    python scripts/sample_diverse_patents.py --total 200000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def count_patents_in_file(filepath: Path) -> int:
    """Count lines in JSONL file."""
    count = 0
    with open(filepath, 'r') as f:
        for _ in f:
            count += 1
    return count


def load_all_patent_ids(filepath: Path) -> list:
    """Load all patent IDs from a file (memory efficient)."""
    ids_with_line = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            patent = json.loads(line)
            ids_with_line.append((patent['patent_id'], line_num))
    return ids_with_line


def sample_patents_from_file(
    filepath: Path,
    sample_size: int,
    random_seed: int
) -> list:
    """
    Randomly sample patents from a JSONL file.
    
    Returns list of patent dicts.
    """
    # First, get all patent IDs with their line numbers
    print(f"  Indexing {filepath.name}...")
    all_ids = load_all_patent_ids(filepath)
    
    # Random sample
    random.seed(random_seed)
    if len(all_ids) <= sample_size:
        sampled_indices = set(range(len(all_ids)))
        actual_sample_size = len(all_ids)
    else:
        sampled_items = random.sample(all_ids, sample_size)
        sampled_indices = {item[1] for item in sampled_items}
        actual_sample_size = sample_size
    
    # Read only the sampled lines
    print(f"  Sampling {actual_sample_size} patents...")
    sampled_patents = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num in sampled_indices:
                sampled_patents.append(json.loads(line))
    
    return sampled_patents


def create_diverse_sample(
    processed_dir: str = 'data/processed',
    output_dir: str = 'data/sampled',
    total_patents: int = 200000,
    random_seed: int = 42
):
    """
    Create a diverse sample of patents stratified by year.
    
    Args:
        processed_dir: Directory with patents_YYYY.jsonl files
        output_dir: Output directory for sampled data
        total_patents: Total patents to sample
        random_seed: Random seed for reproducibility
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all year files
    year_files = sorted(processed_path.glob('patents_*.jsonl'))
    year_files = [f for f in year_files if f.stem.split('_')[1].isdigit()]
    
    if not year_files:
        print("No patent files found!")
        return
    
    print(f"Found {len(year_files)} year files:")
    
    # Count patents per year
    year_counts = {}
    for yf in year_files:
        year = int(yf.stem.split('_')[1])
        count = count_patents_in_file(yf)
        year_counts[year] = count
        print(f"  {year}: {count:,} patents")
    
    total_available = sum(year_counts.values())
    print(f"\nTotal available: {total_available:,} patents")
    
    # Calculate samples per year (equal distribution)
    num_years = len(year_files)
    base_per_year = total_patents // num_years
    
    # Adjust if some years have fewer patents
    samples_per_year = {}
    remaining = total_patents
    
    for year in sorted(year_counts.keys()):
        available = year_counts[year]
        target = min(base_per_year, available, remaining)
        samples_per_year[year] = target
        remaining -= target
    
    # Distribute any remaining quota to years with capacity
    for year in sorted(year_counts.keys()):
        if remaining <= 0:
            break
        available = year_counts[year]
        current = samples_per_year[year]
        can_add = min(available - current, remaining)
        if can_add > 0:
            samples_per_year[year] += can_add
            remaining -= can_add
    
    print(f"\nSampling plan ({total_patents:,} total):")
    for year, count in sorted(samples_per_year.items()):
        pct = count / total_patents * 100
        print(f"  {year}: {count:,} patents ({pct:.1f}%)")
    
    # Perform sampling
    print("\n=== Sampling Patents ===")
    all_sampled = []
    
    for yf in year_files:
        year = int(yf.stem.split('_')[1])
        sample_size = samples_per_year[year]
        
        if sample_size == 0:
            continue
        
        print(f"\nYear {year}:")
        sampled = sample_patents_from_file(yf, sample_size, random_seed + year)
        all_sampled.extend(sampled)
        print(f"  [OK] Sampled {len(sampled):,} patents")
    
    # Shuffle the combined sample
    random.seed(random_seed)
    random.shuffle(all_sampled)
    
    # Save to output file
    output_file = output_path / 'patents_sampled.jsonl'
    print(f"\nSaving to {output_file}...")
    
    with open(output_file, 'w') as f:
        for patent in tqdm(all_sampled, desc="Writing"):
            f.write(json.dumps(patent) + '\n')
    
    # Save sampling metadata
    metadata = {
        'total_sampled': len(all_sampled),
        'samples_per_year': samples_per_year,
        'random_seed': random_seed,
        'source_files': [str(f) for f in year_files]
    }
    
    with open(output_path / 'sampling_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Diverse sampling complete!")
    print(f"   Total: {len(all_sampled):,} patents")
    print(f"   Output: {output_file}")
    print(f"   Metadata: {output_path / 'sampling_metadata.json'}")
    
    # Verify diversity
    print("\n=== Diversity Verification ===")
    year_distribution = {}
    for p in all_sampled:
        y = p.get('year', 'unknown')
        year_distribution[y] = year_distribution.get(y, 0) + 1
    
    for year, count in sorted(year_distribution.items()):
        pct = count / len(all_sampled) * 100
        print(f"  {year}: {count:,} ({pct:.1f}%)")
    
    return all_sampled


def main():
    parser = argparse.ArgumentParser(description='Create diverse patent sample')
    parser.add_argument('--total', type=int, default=200000,
                       help='Total patents to sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='data/sampled',
                       help='Output directory')
    
    args = parser.parse_args()
    
    create_diverse_sample(
        total_patents=args.total,
        random_seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


