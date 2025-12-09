"""Generate training pairs using heuristics (fallback if citations unavailable)."""

import json
import random
from pathlib import Path
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass, asdict
from tqdm import tqdm
import os

@dataclass
class TrainingPair:
    patent_id_1: str
    patent_id_2: str
    label: int
    pair_type: str
    year_1: int = None
    year_2: int = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class TrainingPairGenerator:
    def __init__(
        self,
        input_path: str = 'data/sampled/patents_sampled.jsonl',
        output_dir: str = 'data/training',
        sample_size: int = None,
        random_seed: int = 42
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(random_seed)
        self.patents = {}
        self.year_index = {}
        self.claim_count_index = {}
        
    def load_patents(self, sample_size: int = None):
        count = 0
        with open(self.input_path, 'r') as f:
            for line in tqdm(f, desc="Loading patents"):
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
                
                if year not in self.year_index:
                    self.year_index[year] = []
                self.year_index[year].append(pid)
                
                bucket = 'small' if num_claims <= 5 else 'medium' if num_claims <= 10 else 'large' if num_claims <= 20 else 'xlarge'
                if bucket not in self.claim_count_index:
                    self.claim_count_index[bucket] = []
                self.claim_count_index[bucket].append(pid)
                count += 1
        
        print(f"Loaded {len(self.patents)} patents from {sorted(self.year_index.keys())}")
        
    def generate_same_year_pairs(self, n_pairs: int = 15000) -> List[TrainingPair]:
        print(f"Generating {n_pairs} same-year pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        years = list(self.year_index.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            year = random.choice(years)
            year_patents = self.year_index[year]
            if len(year_patents) < 2:
                continue
            p1, p2 = random.sample(year_patents, 2)
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
        return pairs
    
    def generate_similar_claim_structure_pairs(self, n_pairs: int = 10000) -> List[TrainingPair]:
        print(f"Generating {n_pairs} similar-structure pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        buckets = list(self.claim_count_index.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            bucket = random.choice(buckets)
            bucket_patents = self.claim_count_index[bucket]
            if len(bucket_patents) < 2:
                continue
            p1, p2 = random.sample(bucket_patents, 2)
            if p1 > p2:
                p1, p2 = p2, p1
            if (p1, p2) in seen:
                continue
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
        return pairs
    
    def generate_random_negative_pairs(self, n_pairs: int = 15000) -> List[TrainingPair]:
        print(f"Generating {n_pairs} random negative pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
        
        all_pids = list(self.patents.keys())
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            p1, p2 = random.sample(all_pids, 2)
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
        return pairs
    
    def generate_different_structure_pairs(self, n_pairs: int = 10000) -> List[TrainingPair]:
        print(f"Generating {n_pairs} different-structure pairs...")
        
        pairs = []
        seen: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 20
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
        return pairs
    
    def generate_balanced_dataset(self, n_positive: int = 25000, n_negative: int = 25000) -> Tuple[List[TrainingPair], List[TrainingPair], List[TrainingPair]]:
        same_year = self.generate_same_year_pairs(int(n_positive * 0.6))
        similar_struct = self.generate_similar_claim_structure_pairs(int(n_positive * 0.4))
        all_positives = same_year + similar_struct
        
        random_neg = self.generate_random_negative_pairs(int(n_negative * 0.6))
        struct_neg = self.generate_different_structure_pairs(int(n_negative * 0.4))
        all_negatives = random_neg + struct_neg
        
        all_pairs = all_positives + all_negatives
        random.shuffle(all_pairs)
        
        n_total = len(all_pairs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train = all_pairs[:n_train]
        val = all_pairs[n_train:n_train + n_val]
        test = all_pairs[n_train + n_val:]
        
        print(f"Generated {n_total} pairs: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test
    
    def save_pairs(self, pairs: List[TrainingPair], filename: str):
        path = self.output_dir / filename
        with open(path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + '\n')
    
    def generate_and_save(self, n_positive: int = 25000, n_negative: int = 25000, sample_size: int = None):
        self.load_patents(sample_size=sample_size)
        train, val, test = self.generate_balanced_dataset(n_positive, n_negative)
        
        self.save_pairs(train, 'train_pairs.jsonl')
        self.save_pairs(val, 'val_pairs.jsonl')
        self.save_pairs(test, 'test_pairs.jsonl')
        
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
        print(f"Saved to {self.output_dir}")
        return train, val, test


if __name__ == "__main__":
    generator = TrainingPairGenerator()
    generator.generate_and_save(n_positive=25000, n_negative=25000, sample_size=50000)


