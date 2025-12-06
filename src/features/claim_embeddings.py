"""
Claim-Level Embedding Features

Generates embeddings for individual patent claims and computes
claim-to-claim similarity features.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm


class ClaimEmbedder:
    """Generate and compare claim-level embeddings."""
    
    def __init__(self, model=None, batch_size: int = 32):
        """
        Initialize claim embedder.
        
        Args:
            model: Pre-loaded SentenceTransformer model (optional)
            batch_size: Batch size for encoding
        """
        self.model = model
        self.batch_size = batch_size
        self._load_model_if_needed()
    
    def _load_model_if_needed(self):
        """Load the PatentSBERTa model if not provided."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading PatentSBERTa for claim embeddings...")
            self.model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
            print("Model loaded.")
    
    def extract_claims(self, patent: dict) -> List[str]:
        """Extract claim texts from a patent."""
        claims = patent.get('claims', [])
        claim_texts = []
        
        for claim in claims:
            if isinstance(claim, dict):
                text = claim.get('text', '')
            else:
                text = str(claim)
            
            if text and len(text) > 10:  # Filter empty/tiny claims
                claim_texts.append(text[:1000])  # Truncate very long claims
        
        return claim_texts
    
    def extract_independent_claims(self, patent: dict) -> List[str]:
        """Extract only independent claim texts."""
        ind_claims = patent.get('independent_claims', [])
        
        if not ind_claims:
            # Fall back to first claim if no independent claims marked
            claims = self.extract_claims(patent)
            return claims[:1] if claims else []
        
        claim_texts = []
        for claim in ind_claims:
            if isinstance(claim, dict):
                text = claim.get('text', '')
            else:
                text = str(claim)
            
            if text and len(text) > 10:
                claim_texts.append(text[:1000])
        
        return claim_texts
    
    def encode_claims(self, claim_texts: List[str]) -> np.ndarray:
        """Encode claim texts to embeddings."""
        if not claim_texts:
            return np.array([])
        
        embeddings = self.model.encode(
            claim_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def compute_claim_similarity(
        self,
        patent1: dict,
        patent2: dict,
        use_independent_only: bool = False
    ) -> Dict[str, float]:
        """
        Compute claim-level similarity features between two patents.
        
        Returns:
            Dict with:
            - max_claim_similarity: Highest sim between any two claims
            - mean_claim_similarity: Average sim across all claim pairs
            - independent_claim_similarity: Sim between independent claims
        """
        # Extract claims
        if use_independent_only:
            claims1 = self.extract_independent_claims(patent1)
            claims2 = self.extract_independent_claims(patent2)
        else:
            claims1 = self.extract_claims(patent1)
            claims2 = self.extract_claims(patent2)
        
        # Handle empty claims
        if not claims1 or not claims2:
            return {
                'max_claim_similarity': 0.0,
                'mean_claim_similarity': 0.0,
                'independent_claim_similarity': 0.0
            }
        
        # Encode claims
        emb1 = self.encode_claims(claims1)
        emb2 = self.encode_claims(claims2)
        
        # Compute pairwise similarities
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        
        # Similarity matrix: (n_claims1 x n_claims2)
        sim_matrix = np.dot(emb1_norm, emb2_norm.T)
        
        max_sim = float(sim_matrix.max())
        mean_sim = float(sim_matrix.mean())
        
        # Independent claim similarity (first claims)
        ind_claims1 = self.extract_independent_claims(patent1)[:1]
        ind_claims2 = self.extract_independent_claims(patent2)[:1]
        
        if ind_claims1 and ind_claims2:
            ind_emb1 = self.encode_claims(ind_claims1)
            ind_emb2 = self.encode_claims(ind_claims2)
            ind_emb1_norm = ind_emb1 / (np.linalg.norm(ind_emb1) + 1e-8)
            ind_emb2_norm = ind_emb2 / (np.linalg.norm(ind_emb2) + 1e-8)
            ind_sim = float(np.dot(ind_emb1_norm.flatten(), ind_emb2_norm.flatten()))
        else:
            ind_sim = 0.0
        
        return {
            'max_claim_similarity': max_sim,
            'mean_claim_similarity': mean_sim,
            'independent_claim_similarity': ind_sim
        }


def demo_claim_embeddings():
    """Demo claim embedding features."""
    print("=== Claim Embedding Demo ===\n")
    
    # Load sample patents
    patents = []
    with open('data/sampled/patents_sampled.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            patents.append(json.loads(line))
    
    if len(patents) < 2:
        print("Need at least 2 patents for demo")
        return
    
    # Initialize embedder
    embedder = ClaimEmbedder()
    
    # Show claims
    for i, p in enumerate(patents):
        claims = embedder.extract_claims(p)
        ind_claims = embedder.extract_independent_claims(p)
        print(f"Patent {i+1} ({p['patent_id']}):")
        print(f"  Total claims: {len(claims)}")
        print(f"  Independent claims: {len(ind_claims)}")
        if claims:
            print(f"  First claim: {claims[0][:100]}...")
        print()
    
    # Compute similarity
    print("Computing claim-level similarities...")
    sims = embedder.compute_claim_similarity(patents[0], patents[1])
    
    print("\nClaim Similarity Features:")
    for name, value in sims.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    demo_claim_embeddings()


