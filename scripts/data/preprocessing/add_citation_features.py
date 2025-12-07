"""
Add citation-based features from PatentsView API.

New features:
1. citation_overlap - Jaccard similarity of citation lists
2. shared_citations_count - Number of patents both cite
3. cocitation_score - How often they're cited together
4. citation_age_similarity - Similarity in citation recency patterns
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


class CitationFeatureExtractor:
    """Extract citation-based features for patent pairs."""
    
    def __init__(self, citations_path: str = 'data/citations/filtered_citations.jsonl'):
        """
        Initialize with citation data.
        
        Args:
            citations_path: Path to citation JSONL file
        """
        self.patent_citations: Dict[str, Set[str]] = defaultdict(set)  # patent -> patents it cites
        self.patent_cited_by: Dict[str, Set[str]] = defaultdict(set)   # patent -> patents that cite it
        self.citation_years: Dict[str, List[int]] = defaultdict(list)  # patent -> years of cited patents
        
        self._load_citations(citations_path)
    
    def _load_citations(self, path: str):
        """Load citation data from file."""
        print(f"Loading citations from {path}...")
        
        if not Path(path).exists():
            print(f"Citation file not found: {path}")
            print("Attempting to load from TSV...")
            self._load_from_tsv()
            return
        
        with open(path, 'r') as f:
            for line in tqdm(f, desc="Loading citations"):
                data = json.loads(line)
                citing = str(data.get('citing_patent_id', ''))
                cited = str(data.get('cited_patent_id', ''))
                
                if citing and cited:
                    self.patent_citations[citing].add(cited)
                    self.patent_cited_by[cited].add(citing)
        
        print(f"Loaded citations for {len(self.patent_citations)} citing patents")
        print(f"Loaded cited_by for {len(self.patent_cited_by)} cited patents")
    
    def _load_from_tsv(self):
        """Load from raw TSV if JSONL not available."""
        tsv_path = 'data/citations/g_us_patent_citation.tsv'
        
        if not Path(tsv_path).exists():
            print(f"TSV file not found: {tsv_path}")
            return
        
        print(f"Loading from {tsv_path}...")
        with open(tsv_path, 'r') as f:
            header = f.readline()  # Skip header
            for line in tqdm(f, desc="Loading TSV"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    citing = parts[0]
                    cited = parts[1]
                    self.patent_citations[citing].add(cited)
                    self.patent_cited_by[cited].add(citing)
        
        print(f"Loaded {len(self.patent_citations)} citing patents")
    
    def compute_citation_overlap(self, patent1_id: str, patent2_id: str) -> float:
        """
        Compute Jaccard similarity between citation lists.
        
        Measures: Do these patents cite similar prior art?
        """
        cites1 = self.patent_citations.get(str(patent1_id), set())
        cites2 = self.patent_citations.get(str(patent2_id), set())
        
        if not cites1 and not cites2:
            return 0.0
        
        intersection = len(cites1 & cites2)
        union = len(cites1 | cites2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_shared_citations_count(self, patent1_id: str, patent2_id: str) -> int:
        """
        Count number of patents that both cite.
        
        Measures: How many common references do they have?
        """
        cites1 = self.patent_citations.get(str(patent1_id), set())
        cites2 = self.patent_citations.get(str(patent2_id), set())
        
        return len(cites1 & cites2)
    
    def compute_cocitation_score(self, patent1_id: str, patent2_id: str) -> float:
        """
        Compute co-citation score.
        
        Measures: How often are these patents cited together by other patents?
        """
        cited_by_1 = self.patent_cited_by.get(str(patent1_id), set())
        cited_by_2 = self.patent_cited_by.get(str(patent2_id), set())
        
        if not cited_by_1 and not cited_by_2:
            return 0.0
        
        # Patents that cite both
        cociting = len(cited_by_1 & cited_by_2)
        
        # Normalize by geometric mean of citation counts
        norm = np.sqrt(len(cited_by_1) * len(cited_by_2)) if cited_by_1 and cited_by_2 else 1
        
        return cociting / norm if norm > 0 else 0.0
    
    def compute_bibliographic_coupling(self, patent1_id: str, patent2_id: str) -> float:
        """
        Compute bibliographic coupling strength.
        
        Measures: Normalized count of shared references.
        """
        cites1 = self.patent_citations.get(str(patent1_id), set())
        cites2 = self.patent_citations.get(str(patent2_id), set())
        
        if not cites1 or not cites2:
            return 0.0
        
        shared = len(cites1 & cites2)
        
        # Normalize by minimum citation count
        min_cites = min(len(cites1), len(cites2))
        
        return shared / min_cites if min_cites > 0 else 0.0
    
    def extract_features(self, patent1_id: str, patent2_id: str) -> Dict[str, float]:
        """Extract all citation features for a patent pair."""
        
        citation_overlap = self.compute_citation_overlap(patent1_id, patent2_id)
        shared_count = self.compute_shared_citations_count(patent1_id, patent2_id)
        cocitation = self.compute_cocitation_score(patent1_id, patent2_id)
        biblio_coupling = self.compute_bibliographic_coupling(patent1_id, patent2_id)
        
        # Normalize shared count (log scale, cap at 10)
        shared_count_norm = min(np.log1p(shared_count) / np.log1p(10), 1.0)
        
        return {
            'citation_overlap': citation_overlap,
            'shared_citations_norm': shared_count_norm,
            'cocitation_score': cocitation,
            'bibliographic_coupling': biblio_coupling
        }


def load_pairs(split: str) -> List[Dict]:
    """Load pairs for a split."""
    path = f'data/training/{split}_pairs.jsonl'
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    print("=" * 60)
    print("ADDING CITATION-BASED FEATURES")
    print("=" * 60)
    
    # Initialize citation extractor
    extractor = CitationFeatureExtractor()
    
    if not extractor.patent_citations:
        print("\nNo citation data available. Creating placeholder features...")
        # Create zero features if no citation data
        features_dir = Path('data/features')
        
        for split in ['train', 'val', 'test']:
            X_old = np.load(features_dir / f'{split}_features.X.npy')
            n_samples = X_old.shape[0]
            
            # 4 zero columns for citation features
            citation_features = np.zeros((n_samples, 4))
            
            X_new = np.hstack([X_old, citation_features])
            np.save(features_dir / f'{split}_features_with_citations.X.npy', X_new)
            print(f"  {split}: {X_new.shape}")
        
        return
    
    # Process each split
    features_dir = Path('data/features')
    
    new_feature_names = [
        'citation_overlap',
        'shared_citations_norm',
        'cocitation_score',
        'bibliographic_coupling'
    ]
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split...")
        print(f"{'=' * 40}")
        
        # Load existing features
        X_old = np.load(features_dir / f'{split}_features.X.npy')
        y = np.load(features_dir / f'{split}_features.y.npy')
        
        print(f"Existing features shape: {X_old.shape}")
        
        # Load pairs
        pairs = load_pairs(split)
        print(f"Pairs: {len(pairs)}")
        
        # Compute citation features
        citation_features = []
        for pair in tqdm(pairs, desc="Computing citation features"):
            p1_id = str(pair['patent_id_1'])
            p2_id = str(pair['patent_id_2'])
            
            feats = extractor.extract_features(p1_id, p2_id)
            citation_features.append([
                feats['citation_overlap'],
                feats['shared_citations_norm'],
                feats['cocitation_score'],
                feats['bibliographic_coupling']
            ])
        
        citation_features = np.array(citation_features)
        print(f"Citation features shape: {citation_features.shape}")
        
        # Combine features
        X_new = np.hstack([X_old, citation_features])
        print(f"Combined features shape: {X_new.shape}")
        
        # Save
        np.save(features_dir / f'{split}_features_with_citations.X.npy', X_new)
        np.save(features_dir / f'{split}_features_with_citations.y.npy', y)
        
        # Show citation feature statistics
        print(f"\nCitation feature statistics for {split}:")
        for i, name in enumerate(new_feature_names):
            col = citation_features[:, i]
            nonzero = (col > 0).sum()
            print(f"  {name}: mean={col.mean():.4f}, nonzero={nonzero} ({nonzero/len(col):.1%})")
    
    # Update feature names
    with open(features_dir / 'feature_names.json', 'r') as f:
        old_names = json.load(f)
    
    all_names = old_names + new_feature_names
    with open(features_dir / 'feature_names_with_citations.json', 'w') as f:
        json.dump(all_names, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("CITATION FEATURES COMPUTED")
    print(f"{'=' * 60}")
    print(f"Total features: {len(all_names)}")
    print(f"New features: {new_feature_names}")


if __name__ == "__main__":
    main()

