"""
Feature Extraction Pipeline for Patent Novelty Assessment

Computes features for patent pair classification:
- BM25 document score
- BM25 best claim score  
- Cosine similarity (doc-level)
- Max claim cosine similarity
- Mean embedding difference
- Std embedding difference
- CPC overlap (Jaccard)
- Year difference
- Title Jaccard similarity
- Abstract length ratio
- Claim count ratio
- Shared rare terms ratio
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Container for computed features."""
    patent_id_1: str
    patent_id_2: str
    features: Dict[str, float]
    label: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            'patent_id_1': self.patent_id_1,
            'patent_id_2': self.patent_id_2,
            'features': self.features,
            'label': self.label
        }
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array in specified feature order."""
        return np.array([self.features.get(name, 0.0) for name in feature_names])


class FeatureExtractor:
    """Extract features from patent pairs for MLP classification."""
    
    # Feature names in order (12 base + 3 claim + 4 citation = 19 features)
    FEATURE_NAMES = [
        'bm25_doc_score',
        'bm25_best_claim_score',
        'cosine_doc_similarity',
        'cosine_max_claim_similarity',
        'embedding_diff_mean',
        'embedding_diff_std',
        'cpc_jaccard',
        'year_diff',
        'title_jaccard',
        'abstract_length_ratio',
        'claim_count_ratio',
        'shared_rare_terms_ratio',
        # Claim-level features
        'max_claim_similarity',
        'mean_claim_similarity',
        'independent_claim_similarity',
        # Citation-based features
        'citation_overlap',
        'shared_citations_norm',
        'cocitation_score',
        'bibliographic_coupling',
    ]
    
    # Base features only (for backward compatibility)
    BASE_FEATURE_NAMES = [
        'bm25_doc_score',
        'bm25_best_claim_score',
        'cosine_doc_similarity',
        'cosine_max_claim_similarity',
        'embedding_diff_mean',
        'embedding_diff_std',
        'cpc_jaccard',
        'year_diff',
        'title_jaccard',
        'abstract_length_ratio',
        'claim_count_ratio',
        'shared_rare_terms_ratio',
    ]
    
    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        patent_id_to_idx: Optional[Dict[str, int]] = None,
        bm25_retriever = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            embeddings: Pre-computed patent embeddings (N x D)
            patent_id_to_idx: Mapping from patent_id to embedding index
            bm25_retriever: BM25Retriever instance for lexical scoring
            tfidf_vectorizer: Fitted TF-IDF vectorizer for rare term detection
        """
        self.embeddings = embeddings
        self.patent_id_to_idx = patent_id_to_idx or {}
        self.bm25_retriever = bm25_retriever
        self.tfidf_vectorizer = tfidf_vectorizer
        
    def set_embeddings(self, embeddings: np.ndarray, patent_ids: List[str]):
        """Set embeddings and create ID mapping."""
        self.embeddings = embeddings
        self.patent_id_to_idx = {pid: i for i, pid in enumerate(patent_ids)}
        
    def get_embedding(self, patent_id: str) -> Optional[np.ndarray]:
        """Get embedding for a patent."""
        if self.embeddings is None:
            return None
        idx = self.patent_id_to_idx.get(patent_id)
        if idx is None:
            return None
        return self.embeddings[idx]
    
    # Text-based Features
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def tokenize(text: str) -> set:
        """Simple tokenization for Jaccard computation."""
        if not text:
            return set()
        # Lowercase, split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)
    
    def compute_title_jaccard(self, patent1: dict, patent2: dict) -> float:
        """Compute Jaccard similarity between titles."""
        title1 = patent1.get('title', '')
        title2 = patent2.get('title', '')
        return self.jaccard_similarity(self.tokenize(title1), self.tokenize(title2))
    
    def compute_cpc_jaccard(self, patent1: dict, patent2: dict) -> float:
        """Compute Jaccard similarity between CPC codes."""
        cpc1 = set(patent1.get('cpc', []))
        cpc2 = set(patent2.get('cpc', []))
        return self.jaccard_similarity(cpc1, cpc2)
    
    def compute_year_diff(self, patent1: dict, patent2: dict) -> float:
        """Compute absolute year difference (normalized to 0-1 range)."""
        year1 = patent1.get('year', 2023)
        year2 = patent2.get('year', 2023)
        diff = abs(year1 - year2)
        # Normalize: assume max diff of 10 years
        return min(diff / 10.0, 1.0)
    
    def compute_abstract_length_ratio(self, patent1: dict, patent2: dict) -> float:
        """Compute ratio of abstract lengths (smaller/larger)."""
        len1 = len(patent1.get('abstract', ''))
        len2 = len(patent2.get('abstract', ''))
        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0
        return min(len1, len2) / max(len1, len2)
    
    def compute_claim_count_ratio(self, patent1: dict, patent2: dict) -> float:
        """Compute ratio of claim counts (smaller/larger)."""
        count1 = patent1.get('num_claims', len(patent1.get('claims', [])))
        count2 = patent2.get('num_claims', len(patent2.get('claims', [])))
        if count1 == 0 and count2 == 0:
            return 1.0
        if count1 == 0 or count2 == 0:
            return 0.0
        return min(count1, count2) / max(count1, count2)
    
    # Embedding-based Features
    
    def compute_cosine_doc_similarity(self, patent1: dict, patent2: dict) -> float:
        """Compute cosine similarity between document embeddings."""
        emb1 = self.get_embedding(str(patent1.get('patent_id', '')))
        emb2 = self.get_embedding(str(patent2.get('patent_id', '')))
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Compute cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def compute_embedding_diff_stats(self, patent1: dict, patent2: dict) -> Tuple[float, float]:
        """Compute mean and std of embedding difference."""
        emb1 = self.get_embedding(str(patent1.get('patent_id', '')))
        emb2 = self.get_embedding(str(patent2.get('patent_id', '')))
        
        if emb1 is None or emb2 is None:
            return 0.0, 0.0
        
        diff = np.abs(emb1 - emb2)
        return float(np.mean(diff)), float(np.std(diff))
    
    # BM25-based Features
    
    def compute_bm25_doc_score(self, patent1: dict, patent2: dict) -> float:
        """Compute BM25 score between documents using actual BM25 index."""
        if self.bm25_retriever is None or self.bm25_retriever.bm25 is None:
            return 0.0
        
        # Use patent1's text as query
        query_text = f"{patent1.get('title', '')} {patent1.get('abstract', '')}"
        patent2_id = str(patent2.get('patent_id', ''))
        
        # Search for patent2 in the index
        try:
            # Get BM25 score for patent2 using patent1 as query
            query_tokens = self.bm25_retriever.tokenizer.tokenize(query_text)
            if not query_tokens:
                return 0.0
            
            # Get scores for all documents
            scores = self.bm25_retriever.bm25.get_scores(query_tokens)
            
            # Find patent2's index and return its score
            if patent2_id in self.bm25_retriever.patent_ids:
                idx = self.bm25_retriever.patent_ids.index(patent2_id)
                score = float(scores[idx])
                # Normalize BM25 score to [0, 1] range
                # BM25 scores are typically positive, use tanh normalization for stability
                # This ensures scores are bounded and similar to other features
                return max(0.0, min(1.0, np.tanh(score / 10.0)))
            return 0.0
        except Exception:
            return 0.0
    
    def compute_bm25_best_claim_score(self, patent1: dict, patent2: dict) -> float:
        """Compute best BM25 score between claims."""
        if self.bm25_retriever is None or self.bm25_retriever.bm25 is None:
            return 0.0
        
        # Get claims from both patents
        claims1 = patent1.get('claims', []) or patent1.get('independent_claims', [])
        claims2 = patent2.get('claims', []) or patent2.get('independent_claims', [])
        
        if not claims1 or not claims2:
            return 0.0
        
        # Compute BM25 score for each claim pair, return the maximum
        best_score = 0.0
        patent2_id = str(patent2.get('patent_id', ''))
        
        try:
            for claim1 in claims1[:5]:  # Limit to first 5 claims for efficiency
                query_tokens = self.bm25_retriever.tokenizer.tokenize(str(claim1))
                if not query_tokens:
                    continue
                
                scores = self.bm25_retriever.bm25.get_scores(query_tokens)
                
                if patent2_id in self.bm25_retriever.patent_ids:
                    idx = self.bm25_retriever.patent_ids.index(patent2_id)
                    score = float(scores[idx])
                    normalized = max(0.0, min(1.0, np.tanh(score / 10.0)))
                    best_score = max(best_score, normalized)
            
            return best_score
        except Exception:
            return 0.0
    
    # Rare Terms Feature
    
    def compute_shared_rare_terms_ratio(self, patent1: dict, patent2: dict) -> float:
        """
        Compute ratio of shared rare terms between patents.
        Rare terms are those with low document frequency.
        """
        if self.tfidf_vectorizer is None:
            # Fall back to simple rare word heuristic
            return self._simple_rare_terms_ratio(patent1, patent2)
        
        # Use TF-IDF to identify rare terms
        text1 = f"{patent1.get('title', '')} {patent1.get('abstract', '')}"
        text2 = f"{patent2.get('title', '')} {patent2.get('abstract', '')}"
        
        try:
            tfidf1 = self.tfidf_vectorizer.transform([text1])
            tfidf2 = self.tfidf_vectorizer.transform([text2])
            
            # Get terms with high TF-IDF (rare but present)
            threshold = 0.1
            rare1 = set(np.where(tfidf1.toarray()[0] > threshold)[0])
            rare2 = set(np.where(tfidf2.toarray()[0] > threshold)[0])
            
            return self.jaccard_similarity(rare1, rare2)
        except:
            return self._simple_rare_terms_ratio(patent1, patent2)
    
    def _simple_rare_terms_ratio(self, patent1: dict, patent2: dict) -> float:
        """Simple heuristic for rare terms (long words, technical terms)."""
        text1 = f"{patent1.get('title', '')} {patent1.get('abstract', '')}"
        text2 = f"{patent2.get('title', '')} {patent2.get('abstract', '')}"
        
        # Consider words > 10 chars as potentially rare/technical
        words1 = set(w.lower() for w in re.findall(r'\b\w{10,}\b', text1))
        words2 = set(w.lower() for w in re.findall(r'\b\w{10,}\b', text2))
        
        return self.jaccard_similarity(words1, words2)
    
    # Main Extraction Method
    
    def extract_features(
        self,
        patent1: dict,
        patent2: dict,
        label: Optional[int] = None
    ) -> FeatureVector:
        """
        Extract all features for a patent pair.
        
        Args:
            patent1: First patent dict
            patent2: Second patent dict
            label: Optional ground truth label
            
        Returns:
            FeatureVector with all computed features
        """
        # Compute embedding-based features
        cosine_sim = self.compute_cosine_doc_similarity(patent1, patent2)
        emb_diff_mean, emb_diff_std = self.compute_embedding_diff_stats(patent1, patent2)
        
        features = {
            # BM25 features (using actual BM25 index)
            'bm25_doc_score': self.compute_bm25_doc_score(patent1, patent2),
            'bm25_best_claim_score': self.compute_bm25_best_claim_score(patent1, patent2),
            
            # Embedding features
            'cosine_doc_similarity': cosine_sim,
            'cosine_max_claim_similarity': cosine_sim,  # Same as doc for now
            'embedding_diff_mean': emb_diff_mean,
            'embedding_diff_std': emb_diff_std,
            
            # Metadata features
            'cpc_jaccard': self.compute_cpc_jaccard(patent1, patent2),
            'year_diff': self.compute_year_diff(patent1, patent2),
            
            # Text features
            'title_jaccard': self.compute_title_jaccard(patent1, patent2),
            'abstract_length_ratio': self.compute_abstract_length_ratio(patent1, patent2),
            'claim_count_ratio': self.compute_claim_count_ratio(patent1, patent2),
            'shared_rare_terms_ratio': self.compute_shared_rare_terms_ratio(patent1, patent2),
        }
        
        return FeatureVector(
            patent_id_1=str(patent1.get('patent_id', '')),
            patent_id_2=str(patent2.get('patent_id', '')),
            features=features,
            label=label
        )
    
    def extract_batch(
        self,
        pairs: List[Tuple[dict, dict, Optional[int]]],
        show_progress: bool = True
    ) -> List[FeatureVector]:
        """Extract features for a batch of patent pairs."""
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(pairs, desc="Extracting features") if show_progress else pairs
        
        for patent1, patent2, label in iterator:
            fv = self.extract_features(patent1, patent2, label)
            results.append(fv)
        
        return results
    
    def to_numpy(
        self,
        feature_vectors: List[FeatureVector]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert feature vectors to numpy arrays.
        
        Returns:
            X: Feature matrix (N x F)
            y: Label vector (N,)
            feature_names: List of feature names
        """
        X = np.array([fv.to_array(self.FEATURE_NAMES) for fv in feature_vectors])
        y = np.array([fv.label if fv.label is not None else 0 for fv in feature_vectors])
        
        return X, y, self.FEATURE_NAMES


def demo_feature_extraction():
    """Demo the feature extractor with sample patents."""
    print("=== Feature Extraction Demo ===\n")
    
    # Sample patents
    patent1 = {
        'patent_id': '12345',
        'title': 'Method for neural network optimization',
        'abstract': 'A method for optimizing deep neural networks using gradient descent...',
        'claims': [{'claim_num': 1, 'text': 'A method comprising...'}],
        'num_claims': 5,
        'cpc': ['G06N', 'G06F'],
        'year': 2023
    }
    
    patent2 = {
        'patent_id': '12346',
        'title': 'Neural network training system',
        'abstract': 'A system for training neural networks with backpropagation...',
        'claims': [{'claim_num': 1, 'text': 'A system comprising...'}],
        'num_claims': 8,
        'cpc': ['G06N', 'H04L'],
        'year': 2022
    }
    
    # Extract features
    extractor = FeatureExtractor()
    fv = extractor.extract_features(patent1, patent2, label=1)
    
    print("Features extracted:")
    for name, value in fv.features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\nFeature vector shape: {len(fv.features)} features")


if __name__ == "__main__":
    demo_feature_extraction()


