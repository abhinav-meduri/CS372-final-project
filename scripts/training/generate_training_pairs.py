"""
Generate training pairs for MLP classifier.

Strategy (without CPC codes):
1. POSITIVE PAIRS: Patents from same year with similar claim structures
2. NEGATIVE PAIRS: Random pairs from different years with different claim counts

Later enhancement: Use embedding similarity once available.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass, asdict
from tqdm import tqdm
import os

@dataclass
class TrainingPair:
    """A labeled pair of patents for training."""
    patent_id_1: str
    patent_id_2: str
    label: int  # 1 = similar, 0 = not similar
    pair_type: str
    year_1: int = None
    year_2: int = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class TrainingPairGenerator:
    """Generate training pairs from sampled patents."""
    
    def __init__(
        self,
        input_path: str = 'data/sampled/patents_sampled.jsonl',
        output_dir: str = 'data/training',
        sample_size: int = None,  # Use all if None
        random_seed: int = 42
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(random_seed)
        
        self.patents = {}  # patent_id -> patent dict
        self.year_index = {}  # year -> list of patent_ids
        self.claim_count_index = {}  # claim_count_bucket -> list of patent_ids
        
    def load_patents(self, sample_size: int = None):
        """Load patents into memory with indexing."""
        print(f"Loading patents from {self.input_path}...")
        
        count = 0
        with open(self.input_path, 'r') as f:
            for line in tqdm(f, desc="Loading"):
                if sample_size and count >= sample_size:
                    break
                    
                patent = json.loads(line)
                pid = str(patent['patent_id'])
                year = patent.get('year', 2023)
                num_claims = patent.get('num_claims', 0)
                
                self.patents[pid] = {
                    'patent_id': pid,
                    'year': year,
                    'num_claims': num_claims,
                    'abstract_len': len(patent.get('abstract', '')),
                    'title': patent.get('title', '')[:100]
                }
                
                # Index by year
                if year not in self.year_index:
                    self.year_index[year] = []
                self.year_index[year].append(pid)
                
                # Index by claim count bucket (0-5, 6-10, 11-20, 21+)
                if num_claims <= 5:
                    bucket = 'small'
                elif num_claims <= 10:
                    bucket = 'medium'
                elif num_claims <= 20:
                    bucket = 'large'
                else:
                    bucket = 'xlarge'
                    
                if bucket not in self.claim_count_index:
                    self.claim_count_index[bucket] = []
                self.claim_count_index[bucket].append(pid)
                
                count += 1
        
        print(f"Loaded {len(self.patents)} patents")
        print(f"Years: {sorted(self.year_index.keys())}")
        print(f"Year distribution: {[(y, len(pids)) for y, pids in sorted(self.year_index.items())]}")
        print(f"Claim buckets: {[(b, len(pids)) for b, pids in self.claim_count_index.items()]}")
        
    def generate_same_year_pairs(
        self,
        n_pairs: int = 15000
    ) -> List[TrainingPair]:
        """
        Generate positive pairs from same year.
        Rationale: Patents filed in same year are more likely to be in related fields.
        """
        print(f"\nGenerating {n_pairs} same-year positive pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        years = list(self.year_index.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            # Pick a random year
            year = random.choice(years)
            year_patents = self.year_index[year]
            
            if len(year_patents) < 2:
                continue
            
            # Sample two patents from same year
            p1, p2 = random.sample(year_patents, 2)
            
            # Normalize order
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen:
                continue
            
            seen.add((p1, p2))
            pairs.append(TrainingPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=1,
                pair_type='same_year',
                year_1=year,
                year_2=year
            ))
        
        print(f"Generated {len(pairs)} same-year pairs")
        return pairs
    
    def generate_similar_claim_structure_pairs(
        self,
        n_pairs: int = 10000
    ) -> List[TrainingPair]:
        """
        Generate positive pairs from patents with similar claim counts.
        Rationale: Similar complexity patents may be in related domains.
        """
        print(f"\nGenerating {n_pairs} similar-claim-structure positive pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        buckets = list(self.claim_count_index.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            # Pick a random bucket
            bucket = random.choice(buckets)
            bucket_patents = self.claim_count_index[bucket]
            
            if len(bucket_patents) < 2:
                continue
            
            p1, p2 = random.sample(bucket_patents, 2)
            
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen:
                continue
            
            # Additional filter: same year for stronger signal
            y1 = self.patents[p1]['year']
            y2 = self.patents[p2]['year']
            
            if y1 != y2:
                continue
            
            seen.add((p1, p2))
            pairs.append(TrainingPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=1,
                pair_type='similar_structure',
                year_1=y1,
                year_2=y2
            ))
        
        print(f"Generated {len(pairs)} similar-structure pairs")
        return pairs
    
    def generate_random_negative_pairs(
        self,
        n_pairs: int = 15000
    ) -> List[TrainingPair]:
        """
        Generate negative pairs from random patents across different years.
        """
        print(f"\nGenerating {n_pairs} random cross-year negative pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        all_pids = list(self.patents.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            p1, p2 = random.sample(all_pids, 2)
            
            # Require different years for true negatives
            y1 = self.patents[p1]['year']
            y2 = self.patents[p2]['year']
            
            if y1 == y2:
                continue
            
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen:
                continue
            
            seen.add((p1, p2))
            pairs.append(TrainingPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=0,
                pair_type='random_cross_year',
                year_1=y1,
                year_2=y2
            ))
        
        print(f"Generated {len(pairs)} cross-year negative pairs")
        return pairs
    
    def generate_different_structure_pairs(
        self,
        n_pairs: int = 10000
    ) -> List[TrainingPair]:
        """
        Generate negative pairs from patents with very different claim structures.
        e.g., small vs xlarge claim counts
        """
        print(f"\nGenerating {n_pairs} different-structure negative pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        # Pair opposite buckets
        opposite_pairs = [('small', 'xlarge'), ('medium', 'large')]
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            b1, b2 = random.choice(opposite_pairs)
            
            if b1 not in self.claim_count_index or b2 not in self.claim_count_index:
                continue
            
            if len(self.claim_count_index[b1]) < 1 or len(self.claim_count_index[b2]) < 1:
                continue
            
            p1 = random.choice(self.claim_count_index[b1])
            p2 = random.choice(self.claim_count_index[b2])
            
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen:
                continue
            
            y1 = self.patents[p1]['year']
            y2 = self.patents[p2]['year']
            
            seen.add((p1, p2))
            pairs.append(TrainingPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=0,
                pair_type='different_structure',
                year_1=y1,
                year_2=y2
            ))
        
        print(f"Generated {len(pairs)} different-structure pairs")
        return pairs
    
    def generate_balanced_dataset(
        self,
        n_positive: int = 25000,
        n_negative: int = 25000
    ) -> Tuple[List[TrainingPair], List[TrainingPair], List[TrainingPair]]:
        """Generate balanced train/val/test splits."""
        print("\n" + "="*60)
        print("GENERATING BALANCED TRAINING DATASET")
        print("="*60)
        
        # Generate positive pairs
        same_year = self.generate_same_year_pairs(int(n_positive * 0.6))
        similar_struct = self.generate_similar_claim_structure_pairs(int(n_positive * 0.4))
        all_positives = same_year + similar_struct
        
        # Generate negative pairs  
        random_neg = self.generate_random_negative_pairs(int(n_negative * 0.6))
        struct_neg = self.generate_different_structure_pairs(int(n_negative * 0.4))
        all_negatives = random_neg + struct_neg
        
        # Combine and shuffle
        all_pairs = all_positives + all_negatives
        random.shuffle(all_pairs)
        
        # Split 70/15/15
        n_total = len(all_pairs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train = all_pairs[:n_train]
        val = all_pairs[n_train:n_train + n_val]
        test = all_pairs[n_train + n_val:]
        
        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairs: {n_total}")
        print(f"  Positive: {len(all_positives)} ({len(all_positives)/n_total*100:.1f}%)")
        print(f"  Negative: {len(all_negatives)} ({len(all_negatives)/n_total*100:.1f}%)")
        print(f"\nSplits:")
        print(f"  Train: {len(train)} pairs")
        print(f"  Val:   {len(val)} pairs")
        print(f"  Test:  {len(test)} pairs")
        
        return train, val, test
    
    def save_pairs(self, pairs: List[TrainingPair], filename: str):
        """Save pairs to JSONL."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + '\n')
        print(f"Saved {len(pairs)} pairs to {path}")
    
    def generate_and_save(
        self,
        n_positive: int = 25000,
        n_negative: int = 25000,
        sample_size: int = None
    ):
        """Full pipeline: load, generate, save."""
        self.load_patents(sample_size=sample_size)
        
        train, val, test = self.generate_balanced_dataset(n_positive, n_negative)
        
        self.save_pairs(train, 'train_pairs.jsonl')
        self.save_pairs(val, 'val_pairs.jsonl')
        self.save_pairs(test, 'test_pairs.jsonl')
        
        # Save metadata
        stats = {
            'n_positive_target': n_positive,
            'n_negative_target': n_negative,
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'total_patents': len(self.patents),
            'pair_types': {
                'same_year': sum(1 for p in train+val+test if p.pair_type == 'same_year'),
                'similar_structure': sum(1 for p in train+val+test if p.pair_type == 'similar_structure'),
                'random_cross_year': sum(1 for p in train+val+test if p.pair_type == 'random_cross_year'),
                'different_structure': sum(1 for p in train+val+test if p.pair_type == 'different_structure'),
            }
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n[OK] Training data generation complete!")
        print(f"   Output directory: {self.output_dir}")
        
        return train, val, test


if __name__ == "__main__":
    generator = TrainingPairGenerator()
    generator.generate_and_save(
        n_positive=25000,
        n_negative=25000,
        sample_size=50000  # Use 50K patents for pair generation (faster)
    )


