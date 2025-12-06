"""
Unified Patent Analyzer

Handles all input modalities:
1. Novelty Assessment - Is this patent/idea novel?
2. Prior Art Search - Find patents related to X
3. Document Analysis - Upload and analyze a document

Integrates:
- PatentSBERTa embeddings
- BM25 retrieval
- MLP similarity scoring
- Phi-3 explanations
- PatentsView API evidence
- Online patent search (Google Patents via PatentsView API)
- LLM-based keyword extraction for smarter search
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.input_handler import InputHandler, InputMode, ParsedInput

# Lazy imports - only load heavy dependencies when needed
# This speeds up Streamlit startup significantly


@dataclass
class AnalysisResult:
    """Result of patent analysis."""
    mode: str
    success: bool
    
    # For novelty assessment
    novelty_score: Optional[float] = None
    assessment: Optional[str] = None
    
    # Prior art results
    similar_patents: Optional[List[Dict]] = None
    
    # Explanation
    explanation: Optional[str] = None
    recommendation: Optional[str] = None
    
    # USPTO evidence
    patentsview_data: Optional[List[Dict]] = None
    
    # Original input
    parsed_input: Optional[ParsedInput] = None
    
    # LLM-extracted keywords (NEW: Hybrid RAG feature)
    extracted_keywords: Optional[Dict] = None
    
    # Online search results (NEW: Hybrid RAG feature)
    online_patents: Optional[List[Dict]] = None
    
    # Search metadata
    search_metadata: Optional[Dict] = None
    
    # Errors
    error: Optional[str] = None


class PatentAnalyzer:
    """
    Main analyzer class that handles all input types and modes.
    
    Features hybrid RAG architecture:
    - Local search: PatentSBERTa embeddings via cosine similarity (200K patents)
    - Online search: Google Patents via SerpAPI (millions of patents)
    - LLM keyword extraction for smarter queries
    - BM25 features: Used for classifier features (not for retrieval)
    """
    
    def __init__(
        self,
        patents_path: str = 'data/sampled/patents_sampled.jsonl',
        embeddings_path: str = 'data/embeddings/patent_embeddings.npy',
        patent_ids_path: str = 'data/embeddings/patent_ids.json',
        use_full_phi3: bool = False,
        use_patentsview: bool = True,
        use_online_search: bool = True,
        use_llm_keywords: bool = True,
        serpapi_key: str = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            patents_path: Path to patent database
            embeddings_path: Path to pre-computed embeddings
            patent_ids_path: Path to patent IDs list
            use_full_phi3: Use full Phi-3 model (requires GPU)
            use_patentsview: Fetch evidence from PatentsView API
            use_online_search: Enable online patent search (NEW)
            use_llm_keywords: Use LLM for keyword extraction (NEW)
        """
        self.use_full_phi3 = use_full_phi3
        self.use_patentsview = use_patentsview
        self.use_online_search = use_online_search
        self.use_llm_keywords = use_llm_keywords
        self.serpapi_key = serpapi_key
        
        self.patents = None
        self.embeddings = None
        self.patent_ids = None
        self.st_model = None
        self.input_handler = InputHandler()
        self.explainer = None
        self.patentsview_api = None
        
        # Hybrid RAG components
        self.keyword_extractor = None
        self.online_searcher = None
        
        # PyTorch model for novelty scoring (retraining on 13 features)
        self.pytorch_model = None
        self.feature_extractor = None
        self.feature_names = None
        
        # Paths
        self.patents_path = patents_path
        self.embeddings_path = embeddings_path
        self.patent_ids_path = patent_ids_path
        
        self._loaded = False
    
    def load(self):
        """Load all resources."""
        if self._loaded:
            return
        
        print("Loading Patent Analyzer resources...")
        
        # Optimized loading: Use faster JSON parser and load in chunks
        print("  [INFO] Loading patents database (optimized loading)...")
        self.patents = {}
        
        # Try to use orjson for faster parsing (falls back to json if not available)
        try:
            import orjson
            use_orjson = True
        except ImportError:
            use_orjson = False
        
        try:
            # Optimized loading with buffered I/O and faster JSON parsing
            line_count = 0
            start_time = time.time()
            
            with open(self.patents_path, 'rb', buffering=2*1024*1024) as f:  # 2MB buffer
                buffer = b''
                for chunk in iter(lambda: f.read(4*1024*1024), b''):  # 4MB chunks
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line.strip():
                            try:
                                if use_orjson:
                                    p = orjson.loads(line)
                                else:
                                    p = json.loads(line.decode('utf-8'))
                                self.patents[str(p['patent_id'])] = p
                                line_count += 1
                                
                                # Progress update every 50K patents
                                if line_count % 50000 == 0:
                                    elapsed = time.time() - start_time
                                    rate = line_count / elapsed if elapsed > 0 else 0
                                    remaining = (200000 - line_count) / rate if rate > 0 else 0
                                    print(f"    Loaded {line_count:,}/{200000:,} patents ({rate:.0f} patents/sec, ~{remaining:.0f}s remaining)...", end='\r')
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                # Skip malformed lines
                                continue
                
                # Process remaining buffer
                if buffer.strip():
                    try:
                        if use_orjson:
                            p = orjson.loads(buffer)
                        else:
                            p = json.loads(buffer.decode('utf-8'))
                        self.patents[str(p['patent_id'])] = p
                        line_count += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            
            load_time = time.time() - start_time
            print(f"  [OK] {len(self.patents):,} patents loaded in {load_time:.1f}s ({len(self.patents)/load_time:.0f} patents/sec)")
        except Exception as e:
            # Fallback to original method if optimization fails
            print(f"  [WARN] Optimized loading failed, using standard method: {e}")
            import traceback
            traceback.print_exc()
            with open(self.patents_path, 'r') as f:
                line_count = 0
                for line in f:
                    p = json.loads(line)
                    self.patents[str(p['patent_id'])] = p
                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"    Loaded {line_count:,} patents...", end='\r')
            print(f"  [OK] {len(self.patents):,} patents (local database)")
        
        # Load embeddings
        self.embeddings = np.load(self.embeddings_path)
        with open(self.patent_ids_path, 'r') as f:
            self.patent_ids = json.load(f)
        print(f"  [OK] {len(self.embeddings):,} embeddings (FAISS)")
        
        # Load sentence transformer (lazy import)
        from sentence_transformers import SentenceTransformer
        self.st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
        print(f"  [OK] PatentSBERTa model")
        
        # Initialize explainer (use Ollama for faster inference with KV caching) - lazy import
        from src.explainability.phi3_explainer import get_explainer
        self.explainer = get_explainer(use_full_model=self.use_full_phi3, use_ollama=True)
        mode_str = 'Ollama + KV cache' if self.use_full_phi3 else 'template'
        print(f"  [OK] Phi-3 explainer ({mode_str})")
        
        # Initialize PatentsView API - lazy import
        if self.use_patentsview:
            from data.api.patentsview_api import PatentsViewAPI
            self.patentsview_api = PatentsViewAPI()
            print(f"  [OK] PatentsView API")
        
        # Initialize LLM keyword extractor (Hybrid RAG) - lazy import
        if self.use_llm_keywords:
            from data.api.online_search import LLMKeywordExtractor
            self.keyword_extractor = LLMKeywordExtractor()
            print(f"  [OK] LLM Keyword Extractor (Phi-3)")
        
        # Initialize online search (Hybrid RAG) - lazy import
        if self.use_online_search:
            import os
            from data.api.online_search import GooglePatentsSearch
            # Use provided key, or fall back to environment variable
            api_key = self.serpapi_key or os.environ.get('SERPAPI_KEY')
            self.online_searcher = GooglePatentsSearch(serpapi_key=api_key)
            if api_key:
                print(f"  [OK] Online Patent Search (SerpAPI - Google Patents)")
            else:
                print(f"  [WARN] Online Patent Search (PatentsView API fallback - no SerpAPI key)")
        
        # Load PyTorch model for novelty scoring (retraining on 13 features) - lazy import
        try:
            from src.models.pytorch_classifier import PyTorchPatentClassifier
            from src.features.feature_extractor import FeatureExtractor
            self.pytorch_model = PyTorchPatentClassifier()
            self.pytorch_model.load('models/pytorch_nn')
            print(f"  [OK] PyTorch Neural Network (13 features)")
            
            # Load feature names
            with open('data/features/feature_names_v2.json', 'r') as f:
                self.feature_names = json.load(f)
            
            # Load BM25 index for feature extraction
            from src.retrieval.bm25_retriever import BM25Retriever
            bm25_retriever = BM25Retriever()
            try:
                bm25_retriever.load_index('bm25_index')
                print(f"  [OK] BM25 index loaded ({len(bm25_retriever.patent_ids):,} documents)")
            except Exception as e:
                print(f"  [WARN] BM25 index not available: {e}")
                bm25_retriever = None
            
            # Initialize feature extractor with embeddings and BM25
            self.feature_extractor = FeatureExtractor(
                embeddings=self.embeddings,
                patent_id_to_idx={pid: i for i, pid in enumerate(self.patent_ids)},
                bm25_retriever=bm25_retriever
            )
            print(f"  [OK] Feature Extractor initialized (13 features with real BM25)")
        except Exception as e:
            print(f"  [WARN] PyTorch model loading failed: {e}")
            print(f"     Falling back to similarity-based scoring")
            self.pytorch_model = None
        
        self._loaded = True
        print("Ready! (Hybrid RAG + PyTorch NN enabled)" if (self.use_online_search or self.use_llm_keywords) else "Ready!")
    
    def analyze(self, input_data: Union[str, Dict], status_callback=None) -> AnalysisResult:
        """
        Main entry point - analyze any input.
        
        Args:
            input_data: Can be:
                - Free text (idea, search query, or structured patent)
                - Dict with patent fields
                - Dict with file_path for document upload
            status_callback: Optional function(status_message: str) to update UI
        
        Returns:
            AnalysisResult with novelty assessment or search results
        """
        self.load()
        
        try:
            # Parse input
            parsed = self.input_handler.process(input_data)
            
            # Route to appropriate handler
            if parsed.mode == InputMode.PRIOR_ART_SEARCH:
                return self._handle_search(parsed, status_callback=status_callback)
            else:
                return self._handle_novelty(parsed, status_callback=status_callback)
        
        except Exception as e:
            import traceback
            return AnalysisResult(
                mode="error",
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )
    
    def _handle_novelty(self, parsed: ParsedInput, status_callback=None) -> AnalysisResult:
        """
        Handle novelty assessment with hybrid RAG.
        
        Follows reference implementation pattern:
        1. Generate multiple search terms (LLM)
        2. Search each term on Google Patents (SerpAPI)
        3. Merge all results
        4. Assess novelty
        """
        
        # Convert to patent dict
        query_patent = parsed.to_patent_dict()
        query_text = query_patent.get('abstract', '')[:500]
        
        # Step 1: Generate multiple search terms using LLM (like reference: generate_search_terms)
        extracted_keywords = None
        search_terms = []
        
        if self.use_llm_keywords and self.keyword_extractor:
            if status_callback:
                status_callback("Generating search keywords with LLM...")
            try:
                # Generate optimized search terms for Google Patents
                search_terms = self.keyword_extractor.generate_search_terms(query_text)
                if status_callback:
                    status_callback(f"Generated {len(search_terms)} search terms: {', '.join(search_terms[:3])}...")
                print(f"  [OK] Generated {len(search_terms)} search terms: {search_terms[:3]}...")
                
                # Also extract structured keywords for display
                extracted_keywords = self.keyword_extractor.extract_keywords(query_text)
            except Exception as e:
                if status_callback:
                    status_callback(f"WARNING: LLM keyword generation failed, using query text")
                print(f"  [WARN] LLM keyword generation failed: {e}")
                search_terms = [query_text[:200]]  # Fallback to single query
        
        # Step 2: Local search using PatentSBERTa embeddings
        if status_callback:
            status_callback("Searching local database (200K patents)...")
        query_embedding = self.st_model.encode(query_text)
        local_similar = self._find_similar(query_embedding, top_k=10)
        if status_callback:
            status_callback(f"Found {len(local_similar)} local patents")
        
        # Step 3: Online search with multiple terms (like reference: search_on_google_patents)
        online_patents = []
        search_metadata = {
            "local_count": len(local_similar), 
            "online_count": 0,
            "search_terms": search_terms,
            "patents_per_term": {}
        }
        
        if self.use_online_search and self.online_searcher:
            # Use search terms if available, otherwise use query text as fallback
            if not search_terms:
                search_terms = [query_text[:200]]  # Fallback to query text
                print(f"  [INFO] No LLM keywords, using query text for online search")
            
            if status_callback:
                status_callback(f"Searching online patents with {len(search_terms)} terms...")
            
            
            try:
                # Search each term separately (like reference implementation)
                results_by_term = self.online_searcher.search_multiple_terms(
                    search_terms, 
                    max_per_term=10  # Top 10 per term
                )
                
                
                seen_ids = set()
                for term, results in results_by_term.items():
                    search_metadata["patents_per_term"][term] = len(results)
                    if status_callback:
                        status_callback(f"Term '{term[:50]}...': {len(results)} patents found")
                    
                    # Convert to dict format and deduplicate
                    for r in results:
                        if r.patent_id not in seen_ids:
                            seen_ids.add(r.patent_id)
                            online_patents.append({
                                'patent_id': r.patent_id,
                                'title': r.title,
                                'abstract': r.abstract,
                                'year': r.year,
                                'similarity': r.relevance_score * 0.8,  # Adjust for comparability
                                'source': 'online',
                                'url': r.url,
                                'inventor': r.inventor,
                                'assignee': r.assignee
                            })
                
                search_metadata["online_count"] = len(online_patents)
                if status_callback:
                    status_callback(f"Online search: {len(online_patents)} unique patents found")
                print(f"  [OK] Online search found {len(online_patents)} unique patents across {len(search_terms)} terms")
            except Exception as e:
                if status_callback:
                    status_callback(f"ERROR: Online search failed: {str(e)[:100]}")
                print(f"  [ERROR] Online search failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 4: Merge results (local + online, deduplicated)
        all_similar = self._merge_results(local_similar, online_patents)
        
        # Step 5: Compute novelty score using PyTorch model
        if self.pytorch_model and self.feature_extractor and all_similar:
            try:
                # Extract features for top similar patent
                top_similar = all_similar[0]
                patent_id = str(top_similar['patent_id'])
                similar_patent_data = self.patents.get(patent_id) if self.patents else None
                
                if similar_patent_data:
                    # Ensure query patent has embedding (needed for feature extraction)
                    if 'embedding' not in query_patent:
                        query_text = query_patent.get('abstract', '')[:500]
                        query_patent['embedding'] = self.st_model.encode(query_text)
                    
                    # Extract features
                    feature_vector = self.feature_extractor.extract_features(
                        query_patent,
                        similar_patent_data
                    )
                    
                    # Convert to array in correct order (using feature_names_v2.json order)
                    feature_array = feature_vector.to_array(self.feature_names).reshape(1, -1)
                    
                    # Predict similarity probability (0 = not similar, 1 = similar)
                    similarity_prob = self.pytorch_model.predict_proba(feature_array)[0][1]
                    
                    # Novelty = 1 - similarity
                    novelty_score = 1 - similarity_prob
                    
                    print(f"  [OK] PyTorch model scored: similarity={similarity_prob:.3f}, novelty={novelty_score:.3f}")
                else:
                    # Fallback if patent data not found
                    max_sim = all_similar[0]['similarity'] if all_similar else 0
                    novelty_score = 1 - max_sim
                    print(f"  [WARN] Using similarity fallback (patent data not found)")
            except Exception as e:
                # Fallback to similarity-based scoring
                max_sim = all_similar[0]['similarity'] if all_similar else 0
                novelty_score = 1 - max_sim
                print(f"  [WARN] PyTorch scoring failed: {e}, using similarity fallback")
                import traceback
                traceback.print_exc()
        else:
            # Fallback to similarity-based scoring
            max_sim = all_similar[0]['similarity'] if all_similar else 0
            novelty_score = 1 - max_sim
            if not self.pytorch_model:
                print(f"  [WARN] PyTorch model not loaded, using similarity-based scoring")
        
        # Determine assessment
        if novelty_score > 0.7:
            assessment = "HIGHLY NOVEL"
        elif novelty_score > 0.5:
            assessment = "MODERATELY NOVEL"
        elif novelty_score > 0.3:
            assessment = "LOW NOVELTY"
        else:
            assessment = "NOT NOVEL"
        
        # Fetch PatentsView evidence
        patentsview_data = None
        if self.use_patentsview and self.patentsview_api:
            patentsview_data = self._fetch_patentsview_evidence(all_similar[:3])
        
        # Generate explanation
        if status_callback:
            status_callback("Generating AI explanation...")
        report = self.explainer.generate_explanation(
            query_patent=query_patent,
            similar_patents=all_similar,
            novelty_score=novelty_score,
            patentsview_evidence=patentsview_data
        )
        if status_callback:
            status_callback("Analysis complete!")
        
        return AnalysisResult(
            mode="novelty",
            success=True,
            novelty_score=novelty_score,
            assessment=assessment,
            similar_patents=all_similar,
            explanation=report.full_explanation,
            recommendation=report.recommendation,
            patentsview_data=patentsview_data,
            parsed_input=parsed,
            extracted_keywords=extracted_keywords,
            online_patents=online_patents,
            search_metadata=search_metadata
        )
    
    def _handle_search(self, parsed: ParsedInput, status_callback=None) -> AnalysisResult:
        """
        Handle prior art search using same retrieval pipeline as novelty assessment.
        Returns results without scoring/explanation for faster exploration.
        """
        
        query = parsed.search_query or parsed.raw_text
        
        # Step 1: LLM keyword extraction (same as novelty)
        search_terms = []
        if self.use_llm_keywords and self.keyword_extractor:
            if status_callback:
                status_callback("Generating search keywords...")
            try:
                search_terms = self.keyword_extractor.generate_search_terms(query)
            except Exception:
                search_terms = [query[:200]]
        
        # Step 2: Local search (same as novelty)
        if status_callback:
            status_callback("Searching local database...")
        query_embedding = self.st_model.encode(query)
        local_similar = self._find_similar(query_embedding, top_k=20)
        
        # Step 3: Online search (same as novelty, but optional)
        online_patents = []
        if self.use_online_search and self.online_searcher and search_terms:
            if status_callback:
                status_callback("Searching online patents...")
            try:
                results_by_term = self.online_searcher.search_multiple_terms(
                    search_terms, max_per_term=10
                )
                seen_ids = set()
                for term, results in results_by_term.items():
                    for r in results:
                        if r.patent_id not in seen_ids:
                            seen_ids.add(r.patent_id)
                            online_patents.append({
                                'patent_id': r.patent_id,
                                'title': r.title,
                                'abstract': r.abstract,
                                'year': r.year,
                                'similarity': r.relevance_score * 0.8,
                                'source': 'online'
                            })
            except Exception:
                pass
        
        # Step 4: Merge results (same as novelty)
        all_results = self._merge_results(local_similar, online_patents)
        
        # Format results
        results = []
        for p in all_results[:20]:  # Top 20
            patent_data = self.patents.get(p['patent_id'], {})
            results.append({
                'patent_id': p['patent_id'],
                'similarity': p.get('similarity', 0),
                'title': patent_data.get('title', p.get('title', 'N/A')),
                'abstract': patent_data.get('abstract', p.get('abstract', 'N/A'))[:500],
                'year': patent_data.get('year', p.get('year', 'N/A')),
                'source': p.get('source', 'local')
            })
        
        # Generate summary
        summary = f"""## Prior Art Search Results

### Query: "{query}"

Found **{len(results)} relevant patents** ({len(local_similar)} local + {len(online_patents)} online).

### Top Results:
"""
        for i, r in enumerate(results[:5], 1):
            summary += f"""
**{i}. Patent {r['patent_id']}** (Relevance: {r['similarity']:.1%}, Source: {r['source']})
- Title: {r['title'][:100]}
- Year: {r['year']}
- Abstract: {r['abstract'][:200]}...
"""
        
        return AnalysisResult(
            mode="search",
            success=True,
            similar_patents=results,
            explanation=summary,
            parsed_input=parsed,
            online_patents=online_patents if online_patents else None
        )
    
    def _find_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Find similar patents using cosine similarity."""
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        all_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(all_norms, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            pid = self.patent_ids[idx]
            patent_data = self.patents.get(pid, {})
            results.append({
                'patent_id': pid,
                'similarity': float(similarities[idx]),
                'title': patent_data.get('title', 'N/A'),
                'abstract': patent_data.get('abstract', 'N/A'),
                'year': patent_data.get('year', 'N/A'),
                'claims': patent_data.get('claims', [])
            })
        
        return results
    
    def _merge_results(self, local_results: List[Dict], online_results: List[Dict]) -> List[Dict]:
        """
        Merge local and online search results.
        
        - Deduplicates by patent_id
        - Re-ranks by similarity
        - Tags source (local/online)
        """
        seen_ids = set()
        merged = []
        
        # Add local results first
        for r in local_results:
            pid = str(r.get('patent_id', ''))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                r['source'] = 'local'
                merged.append(r)
        
        # Add online results (if not duplicate)
        for r in online_results:
            pid = str(r.get('patent_id', ''))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                r['source'] = 'online'
                merged.append(r)
        
        # Sort by similarity
        merged.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return merged
    
    def _fetch_patentsview_evidence(self, patents: List[Dict]) -> List[Dict]:
        """Fetch evidence from PatentsView API."""
        
        evidence = []
        for p in patents:
            try:
                details = self.patentsview_api.get_patent_details(p['patent_id'])
                if details:
                    evidence.append(details)
            except Exception as e:
                # PatentsView API may return 410 Gone for some patents (deprecated endpoint or missing data)
                # This is non-critical - we continue without the evidence
                if "410" not in str(e):
                    print(f"API Error for patent {p['patent_id']}: {e}")
                # Silently skip 410 errors as they're expected for some patents
                pass
        
        return evidence


def demo():
    """Demo the analyzer with different input types."""
    
    print("=" * 60)
    print("PATENT ANALYZER DEMO")
    print("=" * 60)
    
    analyzer = PatentAnalyzer(use_patentsview=False)  # Disable API for demo
    
    # Test 1: Free-text idea
    print("\n" + "-" * 40)
    print("[TEST 1] Free-text Idea")
    print("-" * 40)
    
    idea = """
    A smart water bottle with embedded sensors that tracks hydration levels 
    in real-time. It connects to a smartphone app via Bluetooth and uses 
    machine learning to provide personalized hydration recommendations based 
    on activity level, weather, and health data.
    """
    
    result = analyzer.analyze(idea)
    print(f"\nMode: {result.mode}")
    if result.success:
        print(f"Novelty Score: {result.novelty_score:.2f}")
        print(f"Assessment: {result.assessment}")
        print(f"Top Prior Art: {result.similar_patents[0]['patent_id']} ({result.similar_patents[0]['similarity']:.1%})")
    else:
        print(f"Error: {result.error}")
    
    # Test 2: Search query
    print("\n" + "-" * 40)
    print("[TEST 2] Prior Art Search")
    print("-" * 40)
    
    query = "Find patents about machine learning for medical diagnosis"
    result = analyzer.analyze(query)
    
    print(f"\nMode: {result.mode}")
    print(f"Results found: {len(result.similar_patents)}")
    print("\nTop 3 results:")
    for p in result.similar_patents[:3]:
        print(f"  - {p['patent_id']}: {p['title'][:50]}... ({p['similarity']:.1%})")
    
    # Test 3: Structured patent
    print("\n" + "-" * 40)
    print("[TEST 3] Structured Patent")
    print("-" * 40)
    
    patent = {
        "title": "Blockchain-based Supply Chain Tracking System",
        "abstract": """
        A distributed ledger system for tracking goods through a supply chain.
        The system uses blockchain technology to create immutable records of 
        product movement, enabling real-time visibility and authentication.
        Smart contracts automatically verify compliance with regulations.
        """,
        "claims": [
            "A method for supply chain tracking comprising distributed ledger technology",
            "The method of claim 1, wherein smart contracts verify compliance"
        ]
    }
    
    result = analyzer.analyze(patent)
    print(f"\nMode: {result.mode}")
    print(f"Novelty Score: {result.novelty_score:.2f}")
    print(f"Assessment: {result.assessment}")
    print(f"Recommendation: {result.recommendation}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()

