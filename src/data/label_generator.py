"""
Labeled Data Generator for MLP Training

Generates positive and negative patent pairs for novelty classification.

Strategies:
1. POSITIVE PAIRS (similar/related patents):
   - Same CPC subclass + high embedding similarity
   - Citation pairs (if citation data available)
   - High PatentSBERTa cosine similarity (>0.85)

2. NEGATIVE PAIRS (unrelated patents):
   - Random cross-CPC pairs
   - Hard negatives: high BM25 but low embedding similarity
   - Temporal negatives: same field, 3+ years apart
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import os


@dataclass
class PatentPair:
    """A labeled pair of patents for training."""
    patent_id_1: str
    patent_id_2: str
    label: int  # 1 = similar/related, 0 = not related
    pair_type: str  # 'cpc_match', 'embedding_sim', 'random_neg', 'hard_neg', etc.
    similarity_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'patent_id_1': self.patent_id_1,
            'patent_id_2': self.patent_id_2,
            'label': self.label,
            'pair_type': self.pair_type,
            'similarity_score': self.similarity_score
        }


class LabeledDataGenerator:
    """Generate labeled patent pairs for MLP training."""
    
    def __init__(
        self,
        processed_dir: str = 'data/processed',
        embeddings_dir: str = 'data/embeddings',
        output_dir: str = 'data/training',
        random_seed: int = 42
    ):
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.patents = {}  # patent_id -> patent dict
        self.cpc_index = {}  # cpc_code -> list of patent_ids
        self.year_index = {}  # year -> list of patent_ids
        
    def load_patents(self, years: List[int] = None, sample_size: int = None):
        """
        Load patents into memory for pair generation.
        
        Args:
            years: List of years to load (default: all)
            sample_size: Max patents to load per year (for memory management)
        """
        print("Loading patents...")
        
        patent_files = sorted(self.processed_dir.glob('patents_*.jsonl'))
        
        for pf in patent_files:
            year = int(pf.stem.split('_')[1])
            if years and year not in years:
                continue
            
            print(f"  Loading {pf.name}...")
            count = 0
            
            with open(pf, 'r') as f:
                for line in tqdm(f, desc=f"Year {year}"):
                    if sample_size and count >= sample_size:
                        break
                    
                    patent = json.loads(line)
                    pid = str(patent['patent_id'])
                    
                    self.patents[pid] = patent
                    
                    # Index by year
                    if year not in self.year_index:
                        self.year_index[year] = []
                    self.year_index[year].append(pid)
                    
                    # Index by CPC (if available)
                    for cpc in patent.get('cpc', []):
                        if cpc not in self.cpc_index:
                            self.cpc_index[cpc] = []
                        self.cpc_index[cpc].append(pid)
                    
                    count += 1
        
        print(f"Loaded {len(self.patents)} patents")
        print(f"CPC codes indexed: {len(self.cpc_index)}")
        print(f"Years indexed: {list(self.year_index.keys())}")
    
    def generate_cpc_positive_pairs(
        self,
        n_pairs: int = 10000,
        min_cpc_size: int = 10
    ) -> List[PatentPair]:
        """
        Generate positive pairs from patents sharing CPC codes.
        
        Patents with same CPC subclass are likely related.
        """
        print(f"\nGenerating {n_pairs} CPC-based positive pairs...")
        
        pairs = []
        attempts = 0
        max_attempts = n_pairs * 10
        seen_pairs: Set[Tuple[str, str]] = set()
        
        # Filter CPC codes with enough patents
        valid_cpcs = [cpc for cpc, pids in self.cpc_index.items() 
                      if len(pids) >= min_cpc_size]
        
        if not valid_cpcs:
            print("Warning: No CPC codes with enough patents. Using random sampling.")
            return []
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            # Random CPC code
            cpc = random.choice(valid_cpcs)
            cpc_patents = self.cpc_index[cpc]
            
            if len(cpc_patents) < 2:
                continue
            
            # Sample two patents
            p1, p2 = random.sample(cpc_patents, 2)
            
            # Ensure consistent ordering
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen_pairs:
                continue
            
            seen_pairs.add((p1, p2))
            pairs.append(PatentPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=1,
                pair_type='cpc_match'
            ))
        
        print(f"Generated {len(pairs)} CPC positive pairs")
        return pairs
    
    def generate_random_negative_pairs(
        self,
        n_pairs: int = 10000
    ) -> List[PatentPair]:
        """
        Generate negative pairs from random patents (likely different fields).
        """
        print(f"\nGenerating {n_pairs} random negative pairs...")
        
        pairs = []
        all_pids = list(self.patents.keys())
        seen_pairs: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 10
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            p1, p2 = random.sample(all_pids, 2)
            
            # Ensure consistent ordering
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen_pairs:
                continue
            
            # Check they don't share CPC codes (true negatives)
            cpc1 = set(self.patents[p1].get('cpc', []))
            cpc2 = set(self.patents[p2].get('cpc', []))
            
            if cpc1 & cpc2:  # Shared CPC - skip
                continue
            
            seen_pairs.add((p1, p2))
            pairs.append(PatentPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=0,
                pair_type='random_neg'
            ))
        
        print(f"Generated {len(pairs)} random negative pairs")
        return pairs
    
    def generate_temporal_negative_pairs(
        self,
        n_pairs: int = 5000,
        min_year_gap: int = 3
    ) -> List[PatentPair]:
        """
        Generate negative pairs from patents far apart in time.
        
        Even same-field patents from 3+ years apart may have evolved significantly.
        """
        print(f"\nGenerating {n_pairs} temporal negative pairs (gap >= {min_year_gap} years)...")
        
        pairs = []
        years = sorted(self.year_index.keys())
        seen_pairs: Set[Tuple[str, str]] = set()
        attempts = 0
        max_attempts = n_pairs * 10
        
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            
            # Pick two years with sufficient gap
            y1, y2 = random.sample(years, 2)
            if abs(y1 - y2) < min_year_gap:
                continue
            
            p1 = random.choice(self.year_index[y1])
            p2 = random.choice(self.year_index[y2])
            
            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in seen_pairs:
                continue
            
            seen_pairs.add((p1, p2))
            pairs.append(PatentPair(
                patent_id_1=p1,
                patent_id_2=p2,
                label=0,
                pair_type='temporal_neg'
            ))
        
        print(f"Generated {len(pairs)} temporal negative pairs")
        return pairs
    
    def generate_high_similarity_pairs(
        self,
        embeddings: np.ndarray,
        patent_ids: List[str],
        n_pairs: int = 10000,
        min_similarity: float = 0.85
    ) -> List[PatentPair]:
        """
        Generate positive pairs based on embedding similarity.
        
        Requires pre-computed embeddings.
        """
        print(f"\nGenerating high-similarity pairs (cosine >= {min_similarity})...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        pairs = []
        n = len(patent_ids)
        
        # Compute similarities in batches to avoid memory issues
        batch_size = 1000
        
        for i in tqdm(range(0, n, batch_size)):
            batch_end = min(i + batch_size, n)
            batch_emb = embeddings[i:batch_end]
            
            # Compute similarity to all other embeddings
            sims = cosine_similarity(batch_emb, embeddings)
            
            for j in range(batch_end - i):
                global_idx = i + j
                
                # Find high-similarity pairs
                high_sim_indices = np.where(sims[j] >= min_similarity)[0]
                
                for k in high_sim_indices:
                    if k <= global_idx:  # Avoid duplicates
                        continue
                    
                    p1, p2 = patent_ids[global_idx], patent_ids[k]
                    
                    pairs.append(PatentPair(
                        patent_id_1=p1,
                        patent_id_2=p2,
                        label=1,
                        pair_type='embedding_sim',
                        similarity_score=float(sims[j, k])
                    ))
                    
                    if len(pairs) >= n_pairs:
                        break
                
                if len(pairs) >= n_pairs:
                    break
            
            if len(pairs) >= n_pairs:
                break
        
        print(f"Generated {len(pairs)} high-similarity pairs")
        return pairs
    
    def generate_hard_negative_pairs(
        self,
        bm25_scores: Dict[str, List[Tuple[str, float]]],
        embeddings: np.ndarray,
        patent_ids: List[str],
        n_pairs: int = 5000,
        bm25_top_k: int = 20,
        embedding_threshold: float = 0.5
    ) -> List[PatentPair]:
        """
        Generate hard negatives: high BM25 score but low embedding similarity.
        
        These are lexically similar but semantically different - critical for
        training a robust model.
        """
        print(f"\nGenerating {n_pairs} hard negative pairs...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        pairs = []
        pid_to_idx = {pid: i for i, pid in enumerate(patent_ids)}
        
        for query_pid, candidates in tqdm(bm25_scores.items()):
            if query_pid not in pid_to_idx:
                continue
            
            query_idx = pid_to_idx[query_pid]
            query_emb = embeddings[query_idx:query_idx+1]
            
            # Check top BM25 candidates
            for cand_pid, bm25_score in candidates[:bm25_top_k]:
                if cand_pid not in pid_to_idx:
                    continue
                
                cand_idx = pid_to_idx[cand_pid]
                cand_emb = embeddings[cand_idx:cand_idx+1]
                
                # Compute embedding similarity
                emb_sim = cosine_similarity(query_emb, cand_emb)[0, 0]
                
                # Hard negative: high BM25, low embedding similarity
                if emb_sim < embedding_threshold:
                    p1, p2 = query_pid, cand_pid
                    if p1 > p2:
                        p1, p2 = p2, p1
                    
                    pairs.append(PatentPair(
                        patent_id_1=p1,
                        patent_id_2=p2,
                        label=0,
                        pair_type='hard_neg',
                        similarity_score=emb_sim
                    ))
                    
                    if len(pairs) >= n_pairs:
                        break
            
            if len(pairs) >= n_pairs:
                break
        
        print(f"Generated {len(pairs)} hard negative pairs")
        return pairs
    
    def generate_balanced_dataset(
        self,
        n_positive: int = 25000,
        n_negative: int = 25000
    ) -> Tuple[List[PatentPair], List[PatentPair], List[PatentPair]]:
        """
        Generate a balanced dataset of positive and negative pairs.
        
        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs)
        """
        print("\nGenerating Balanced Training Dataset")
        
        # Generate positive pairs
        cpc_positives = self.generate_cpc_positive_pairs(n_positive)
        
        # If not enough CPC positives, we'll add embedding-based later
        all_positives = cpc_positives
        
        # Generate negative pairs
        random_negatives = self.generate_random_negative_pairs(int(n_negative * 0.6))
        temporal_negatives = self.generate_temporal_negative_pairs(int(n_negative * 0.4))
        all_negatives = random_negatives + temporal_negatives
        
        # Combine and shuffle
        all_pairs = all_positives + all_negatives
        random.shuffle(all_pairs)
        
        # Split into train/val/test (70/15/15)
        n_total = len(all_pairs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:n_train + n_val]
        test_pairs = all_pairs[n_train + n_val:]
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_pairs)} pairs")
        print(f"  Val: {len(val_pairs)} pairs")
        print(f"  Test: {len(test_pairs)} pairs")
        
        return train_pairs, val_pairs, test_pairs
    
    def save_pairs(
        self,
        pairs: List[PatentPair],
        filename: str
    ):
        """Save pairs to JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + '\n')
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
    def generate_and_save_all(
        self,
        n_positive: int = 25000,
        n_negative: int = 25000
    ):
        """Generate all pairs and save to files."""
        train, val, test = self.generate_balanced_dataset(n_positive, n_negative)
        
        self.save_pairs(train, 'train_pairs.jsonl')
        self.save_pairs(val, 'val_pairs.jsonl')
        self.save_pairs(test, 'test_pairs.jsonl')
        
        # Save stats
        stats = {
            'n_positive': n_positive,
            'n_negative': n_negative,
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'total_patents_loaded': len(self.patents)
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset generation complete!")
        return train, val, test


def demo_label_generation():
    """Demo the label generator with a small sample."""
    print("Label Generation Demo\n")
    
    generator = LabeledDataGenerator()
    
    # Load a small sample (2000 patents from 2024)
    generator.load_patents(years=[2024], sample_size=2000)
    
    # Generate pairs
    train, val, test = generator.generate_and_save_all(
        n_positive=500,
        n_negative=500
    )
    
    # Show sample
    print("\nSample Pairs")
    for pair in train[:3]:
        print(f"  {pair.patent_id_1} <-> {pair.patent_id_2}")
        print(f"    Label: {pair.label}, Type: {pair.pair_type}")


if __name__ == "__main__":
    demo_label_generation()


