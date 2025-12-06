"""
BM25 Retrieval Module

Implements lexical search using BM25 algorithm for finding candidate prior art patents.
BM25 is effective for keyword-based matching and serves as the first stage of our
two-stage retrieval pipeline (BM25 → PatentSBERTa reranking).

Features:
- Tokenization with patent-specific preprocessing
- Efficient indexing and retrieval
- Batch query support
- Persistence (save/load index)
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    patent_id: str
    score: float
    rank: int
    text_snippet: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'patent_id': self.patent_id,
            'score': self.score,
            'rank': self.rank,
            'text_snippet': self.text_snippet
        }


class PatentTokenizer:
    """
    Patent-specific tokenizer with preprocessing.
    
    Handles:
    - Lowercasing
    - Patent number normalization
    - Technical term preservation
    - Stop word removal
    - Legal phrase handling
    """
    
    # Patent-specific stop words to remove
    PATENT_STOPWORDS = {
        'claim', 'claims', 'wherein', 'comprising', 'consists',
        'consisting', 'includes', 'including', 'having', 'method',
        'apparatus', 'system', 'device', 'means', 'step', 'steps',
        'according', 'invention', 'present', 'embodiment', 'embodiments',
        'figure', 'figures', 'fig', 'figs', 'described', 'shown',
        'provided', 'comprises', 'thereof', 'therein', 'herein',
        'said', 'first', 'second', 'third', 'fourth', 'fifth',
        'one', 'two', 'three', 'four', 'five', 'plurality'
    }
    
    # Terms to preserve (don't remove even if in stopwords)
    PRESERVE_TERMS = {
        'neural', 'network', 'machine', 'learning', 'deep',
        'algorithm', 'data', 'processing', 'computer', 'software',
        'hardware', 'circuit', 'signal', 'sensor', 'wireless',
        'battery', 'electric', 'magnetic', 'optical', 'chemical',
        'compound', 'composition', 'polymer', 'protein', 'gene'
    }
    
    def __init__(self, remove_stopwords: bool = True, min_token_length: int = 2):
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        
        # Combine NLTK stopwords with patent-specific ones
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
        
        self.stopwords.update(self.PATENT_STOPWORDS)
        
        # Remove preserve terms from stopwords
        self.stopwords -= self.PRESERVE_TERMS
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with patent-specific preprocessing."""
        if not text:
            return []
        
        # Lowercase
        text = text.lower()
        
        # Normalize patent numbers (e.g., US7,654,321 → US7654321)
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        
        # Remove special characters but keep alphanumeric and hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        
        # Filter tokens
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Skip pure numbers (but keep alphanumeric like "h264")
            if token.isdigit():
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered


class BM25Retriever:
    """
    BM25-based patent retrieval system.
    
    Usage:
        retriever = BM25Retriever()
        retriever.build_index(patents)
        results = retriever.search("neural network image classification", top_k=100)
    """
    
    def __init__(
        self,
        tokenizer: Optional[PatentTokenizer] = None,
        index_dir: str = 'data/indices'
    ):
        self.tokenizer = tokenizer or PatentTokenizer()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.bm25 = None
        self.patent_ids: List[str] = []
        self.patent_texts: List[str] = []  # Store for snippet extraction
        self.corpus: List[List[str]] = []  # Tokenized corpus
        
    def _get_patent_text(self, patent: dict) -> str:
        """
        Extract searchable text from patent.
        
        Combines: title + abstract/summary + independent claims
        """
        parts = []
        
        # Title
        if patent.get('title'):
            parts.append(patent['title'])
        
        # Abstract/Summary
        if patent.get('abstract'):
            parts.append(patent['abstract'])
        elif patent.get('summary'):
            parts.append(patent['summary'])
        
        # Independent claims (most important for novelty)
        for claim in patent.get('independent_claims', []):
            if isinstance(claim, str):
                parts.append(claim)
            elif isinstance(claim, dict) and 'text' in claim:
                parts.append(claim['text'])
        
        return ' '.join(parts)
    
    def build_index(
        self,
        patents: List[dict],
        show_progress: bool = True
    ):
        """
        Build BM25 index from list of patents.
        
        Args:
            patents: List of patent dicts with 'patent_id', 'abstract', 'claims', etc.
            show_progress: Show progress bar
        """
        print(f"Building BM25 index for {len(patents)} patents...")
        
        self.patent_ids = []
        self.patent_texts = []
        self.corpus = []
        
        iterator = tqdm(patents, desc="Tokenizing") if show_progress else patents
        
        for patent in iterator:
            pid = str(patent.get('patent_id', ''))
            text = self._get_patent_text(patent)
            tokens = self.tokenizer.tokenize(text)
            
            if tokens:  # Only add if we have tokens
                self.patent_ids.append(pid)
                self.patent_texts.append(text[:500])  # Store snippet
                self.corpus.append(tokens)
        
        print(f"Building BM25 model from {len(self.corpus)} documents...")
        self.bm25 = BM25Okapi(self.corpus)
        
        print(f"Index built successfully!")
        print(f"  Documents: {len(self.patent_ids)}")
        print(f"  Avg tokens/doc: {np.mean([len(c) for c in self.corpus]):.1f}")
    
    def build_index_from_files(
        self,
        patent_files: List[str],
        sample_per_file: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Build index from JSONL files.
        
        Args:
            patent_files: List of paths to patent JSONL files
            sample_per_file: Optional limit per file (for memory management)
        """
        all_patents = []
        
        for pf in patent_files:
            print(f"Loading {pf}...")
            count = 0
            
            with open(pf, 'r') as f:
                for line in tqdm(f, desc=f"Reading {Path(pf).name}"):
                    if sample_per_file and count >= sample_per_file:
                        break
                    
                    patent = json.loads(line)
                    all_patents.append(patent)
                    count += 1
        
        self.build_index(all_patents, show_progress)
    
    def search(
        self,
        query: str,
        top_k: int = 100,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar patents.
        
        Args:
            query: Query text (e.g., a patent abstract or claim)
            top_k: Number of results to return
            exclude_ids: Patent IDs to exclude from results
            
        Returns:
            List of RetrievalResult objects sorted by score
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        if exclude_ids:
            # Mask excluded IDs
            for i, pid in enumerate(self.patent_ids):
                if pid in exclude_ids:
                    scores[i] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            
            results.append(RetrievalResult(
                patent_id=self.patent_ids[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                text_snippet=self.patent_texts[idx] if idx < len(self.patent_texts) else None
            ))
        
        return results
    
    def batch_search(
        self,
        queries: List[Tuple[str, str]],  # (query_id, query_text)
        top_k: int = 100,
        exclude_self: bool = True,
        show_progress: bool = True
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of (query_id, query_text) tuples
            top_k: Results per query
            exclude_self: Exclude query patent from its own results
            
        Returns:
            Dict mapping query_id to list of results
        """
        results = {}
        
        iterator = tqdm(queries, desc="Searching") if show_progress else queries
        
        for query_id, query_text in iterator:
            exclude_ids = {query_id} if exclude_self else None
            results[query_id] = self.search(query_text, top_k, exclude_ids)
        
        return results
    
    def save_index(self, name: str = 'bm25_index'):
        """Save index to disk."""
        index_path = self.index_dir / f'{name}.pkl'
        
        data = {
            'patent_ids': self.patent_ids,
            'patent_texts': self.patent_texts,
            'corpus': self.corpus
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, name: str = 'bm25_index'):
        """Load index from disk."""
        index_path = self.index_dir / f'{name}.pkl'
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        print(f"Loading index from {index_path}...")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        self.patent_ids = data['patent_ids']
        self.patent_texts = data['patent_texts']
        self.corpus = data['corpus']
        
        # Rebuild BM25 model
        print("Rebuilding BM25 model...")
        self.bm25 = BM25Okapi(self.corpus)
        
        print(f"Index loaded! {len(self.patent_ids)} documents")
    
    def get_statistics(self) -> dict:
        """Get index statistics."""
        if not self.corpus:
            return {'status': 'empty'}
        
        token_counts = [len(c) for c in self.corpus]
        
        return {
            'num_documents': len(self.patent_ids),
            'avg_tokens_per_doc': np.mean(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_tokens': sum(token_counts),
            'unique_patents': len(set(self.patent_ids))
        }


def demo_bm25_retrieval():
    """Demo the BM25 retriever with sample data."""
    print("=== BM25 Retrieval Demo ===\n")
    
    # Sample patents
    sample_patents = [
        {
            'patent_id': 'US001',
            'title': 'Neural Network Image Classification',
            'abstract': 'A deep learning method for classifying images using convolutional neural networks with attention mechanisms.',
            'independent_claims': ['A method for image classification using deep neural networks.']
        },
        {
            'patent_id': 'US002', 
            'title': 'Natural Language Processing System',
            'abstract': 'A transformer-based system for natural language understanding and generation using attention mechanisms.',
            'independent_claims': ['A system for processing natural language text using transformer models.']
        },
        {
            'patent_id': 'US003',
            'title': 'Battery Management System',
            'abstract': 'An electric vehicle battery management system with thermal control and charge optimization.',
            'independent_claims': ['A battery management apparatus for electric vehicles.']
        },
        {
            'patent_id': 'US004',
            'title': 'Image Recognition with CNN',
            'abstract': 'A convolutional neural network architecture for real-time image recognition and object detection.',
            'independent_claims': ['A method for recognizing objects in images using convolutional neural networks.']
        },
        {
            'patent_id': 'US005',
            'title': 'Wireless Communication Protocol',
            'abstract': 'A novel wireless communication protocol for IoT devices with low power consumption.',
            'independent_claims': ['A wireless communication method for Internet of Things devices.']
        }
    ]
    
    # Build index
    retriever = BM25Retriever()
    retriever.build_index(sample_patents, show_progress=False)
    
    # Print statistics
    stats = retriever.get_statistics()
    print(f"\nIndex Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Search
    query = "deep learning neural network for image classification"
    print(f"\nQuery: '{query}'")
    print("\nResults:")
    
    results = retriever.search(query, top_k=3)
    for r in results:
        print(f"  Rank {r.rank}: {r.patent_id} (score: {r.score:.2f})")
        print(f"    Snippet: {r.text_snippet[:100]}...")
    
    # Test batch search
    print("\n--- Batch Search Test ---")
    queries = [
        ('US001', 'neural network image classification'),
        ('US003', 'battery management electric vehicle')
    ]
    
    batch_results = retriever.batch_search(queries, top_k=2, show_progress=False)
    for qid, results in batch_results.items():
        print(f"\nQuery {qid}:")
        for r in results:
            print(f"  Rank {r.rank}: {r.patent_id} (score: {r.score:.2f})")


if __name__ == "__main__":
    demo_bm25_retrieval()


