"""
PatentSBERTa Embedding Module
Generates embeddings using the AI-Growth-Lab/PatentSBERTa model.

Model: https://huggingface.co/AI-Growth-Lab/PatentSBERTa
- 768-dimensional embeddings
- Max sequence length: 512 tokens
- CLS pooling
- Based on MPNet architecture
"""

import numpy as np
import torch
from typing import List, Union, Optional
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512


class PatentEmbedder:
    """Generate patent embeddings using PatentSBERTa."""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the PatentSBERTa embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None for auto-detection
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Lazy loading - model loaded on first use
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading PatentSBERTa model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._model.to(self.device)
            logger.info(f"Model loaded successfully. Embedding dim: {EMBEDDING_DIM}")
        return self._model
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar
            normalize: L2 normalize embeddings (recommended for cosine similarity)
            
        Returns:
            numpy array of shape (n_texts, 768)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device
        )
        
        return embeddings
    
    def encode_patent(
        self, 
        title: str, 
        abstract: str, 
        claims: Optional[List[str]] = None,
        include_claims_in_doc: bool = True
    ) -> dict:
        """
        Encode a patent with document-level and claim-level embeddings.
        
        Args:
            title: Patent title
            abstract: Patent abstract
            claims: List of claim texts
            include_claims_in_doc: Include independent claims in doc embedding
            
        Returns:
            dict with 'doc_embedding' and optionally 'claim_embeddings'
        """
        # Build document text
        doc_parts = [title, abstract]
        
        if include_claims_in_doc and claims:
            # Include first claim (usually independent) in doc embedding
            doc_parts.append(claims[0])
        
        doc_text = " ".join(filter(None, doc_parts))
        
        # Truncate if too long (model handles this, but good to be aware)
        doc_embedding = self.encode(doc_text, show_progress=False)
        
        result = {
            'doc_embedding': doc_embedding[0]
        }
        
        # Generate claim-level embeddings if claims provided
        if claims:
            claim_embeddings = self.encode(claims, show_progress=False)
            result['claim_embeddings'] = claim_embeddings
        
        return result
    
    def encode_patents_batch(
        self, 
        patents: List[dict],
        include_claims: bool = True
    ) -> List[dict]:
        """
        Encode multiple patents efficiently.
        
        Args:
            patents: List of patent dicts with 'title', 'abstract', 'claims'
            include_claims: Generate claim-level embeddings
            
        Returns:
            List of dicts with embeddings
        """
        # Collect all texts for batch encoding
        doc_texts = []
        claim_texts = []
        claim_indices = []  # Track which claims belong to which patent
        
        for i, patent in enumerate(patents):
            title = patent.get('title', '')
            abstract = patent.get('abstract', '')
            claims = patent.get('claims', [])
            
            # Build document text
            doc_parts = [title, abstract]
            if claims:
                doc_parts.append(claims[0] if isinstance(claims[0], str) else claims[0].get('text', ''))
            doc_texts.append(" ".join(filter(None, doc_parts)))
            
            # Collect claims
            if include_claims and claims:
                for claim in claims:
                    claim_text = claim if isinstance(claim, str) else claim.get('text', '')
                    claim_texts.append(claim_text)
                    claim_indices.append(i)
        
        # Batch encode documents
        logger.info(f"Encoding {len(doc_texts)} documents...")
        doc_embeddings = self.encode(doc_texts, show_progress=True)
        
        # Batch encode claims
        claim_embeddings = None
        if claim_texts:
            logger.info(f"Encoding {len(claim_texts)} claims...")
            claim_embeddings = self.encode(claim_texts, show_progress=True)
        
        # Assemble results
        results = []
        for i, patent in enumerate(patents):
            result = {
                'patent_id': patent.get('patent_id'),
                'doc_embedding': doc_embeddings[i]
            }
            
            if include_claims and claim_embeddings is not None:
                # Get claims for this patent
                patent_claim_embeds = [
                    claim_embeddings[j] 
                    for j, idx in enumerate(claim_indices) 
                    if idx == i
                ]
                if patent_claim_embeds:
                    result['claim_embeddings'] = np.array(patent_claim_embeds)
            
            results.append(result)
        
        return results
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        path: Union[str, Path],
        patent_ids: Optional[List[str]] = None
    ):
        """Save embeddings to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}")
        
        # Save ID mapping if provided
        if patent_ids:
            id_path = path.with_suffix('.ids.npy')
            np.save(id_path, np.array(patent_ids))
            logger.info(f"Saved ID mapping to {id_path}")
    
    def load_embeddings(self, path: Union[str, Path]) -> tuple:
        """Load embeddings from disk."""
        path = Path(path)
        embeddings = np.load(path)
        
        id_path = path.with_suffix('.ids.npy')
        patent_ids = None
        if id_path.exists():
            patent_ids = np.load(id_path, allow_pickle=True).tolist()
        
        return embeddings, patent_ids


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    # If already normalized, dot product = cosine similarity
    return float(np.dot(emb1, emb2.T).squeeze())


def test_model():
    """Test the PatentSBERTa model."""
    embedder = PatentEmbedder()
    
    # Test with sample patent text
    sample_texts = [
        "A method for training neural networks using backpropagation",
        "System and apparatus for wireless communication in mobile devices",
        "Chemical composition for treating inflammatory diseases"
    ]
    
    embeddings = embedder.encode(sample_texts)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    
    # Test similarity
    sim_01 = compute_cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = compute_cosine_similarity(embeddings[0], embeddings[2])
    sim_12 = compute_cosine_similarity(embeddings[1], embeddings[2])
    
    print(f"\nCosine similarities:")
    print(f"  Neural networks vs Wireless: {sim_01:.4f}")
    print(f"  Neural networks vs Chemical: {sim_02:.4f}")
    print(f"  Wireless vs Chemical: {sim_12:.4f}")


if __name__ == "__main__":
    test_model()





