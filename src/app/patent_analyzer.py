"""
Unified Patent Analyzer

Handles all input modalities:

1. Novelty Assessment - Is this patent/idea novel?

2. Prior Art Search - Find patents related to X

3. Document Analysis - Upload and analyze a document

Integrates:

- PatentSBERTa embeddings

- MLP similarity scoring

- Phi-3 explanations

- Online patent search (Google Patents via SerpAPI)

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
    """
    
    def __init__(
        self,
        patents_path: str = 'data/sampled/patents_sampled.jsonl',
        embeddings_path: str = 'data/embeddings/patent_embeddings.npy',
        patent_ids_path: str = 'data/embeddings/patent_ids.json',
        use_full_phi3: bool = False,
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
            use_online_search: Enable online patent search via SerpAPI
            use_llm_keywords: Use LLM for keyword extraction
        """
        self.use_full_phi3 = use_full_phi3
        self.use_online_search = use_online_search
        self.use_llm_keywords = use_llm_keywords
        self.serpapi_key = serpapi_key
        
        self.patents = {}  # Will be loaded lazily
        self.embeddings = None
        self.patent_ids = None
        self.st_model = None
        self.input_handler = InputHandler()
        self.explainer = None
        
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
    
    def load(self, status_callback=None):
        """Load all resources lazily with progress updates."""
        if self._loaded:
            return
        
        def update(msg):
            if status_callback:
                status_callback(msg)
            print(f"  {msg}")
        
        update("Loading Patent Analyzer resources...")
        
        if self.embeddings is None:
            update("Loading embeddings...")
            self.embeddings = np.load(self.embeddings_path, mmap_mode='r')
            with open(self.patent_ids_path, 'r') as f:
                self.patent_ids = json.load(f)
            update(f"Loaded {len(self.embeddings):,} embeddings")
        
        if self.st_model is None:
            update("Loading PatentSBERTa model...")
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
            update("PatentSBERTa model loaded")
        
        if self.explainer is None:
            update("Initializing Phi-3 explainer...")
            from src.app.phi3_explainer import Phi3OllamaExplainer
            self.explainer = Phi3OllamaExplainer()
            update("Phi-3 explainer ready")
        
        if self.use_llm_keywords and self.keyword_extractor is None:
            update("Initializing LLM keyword extractor...")
            from data.api.online_search import LLMKeywordExtractor
            self.keyword_extractor = LLMKeywordExtractor()
            update("LLM Keyword Extractor ready")
        
        if self.use_online_search and self.online_searcher is None:
            update("Initializing online search...")
            import os
            from data.api.online_search import GooglePatentsSearch
            api_key = self.serpapi_key or os.environ.get('SERPAPI_KEY')
            self.online_searcher = GooglePatentsSearch(serpapi_key=api_key)
            if api_key:
                update("Online Patent Search ready (SerpAPI)")
            else:
                update("Online Patent Search disabled (SerpAPI key not configured)")
        
        if self.pytorch_model is None:
            try:
                update("Loading PyTorch model...")
                from src.app.pytorch_classifier import PyTorchPatentClassifier
                from src.features.feature_extract import FeatureExtractor
                self.pytorch_model = PyTorchPatentClassifier()
                self.pytorch_model.load('models/pytorch_nn')
                
                with open('data/features/feature_names_v2.json', 'r') as f:
                    self.feature_names = json.load(f)
                
                self.feature_extractor = FeatureExtractor(
                    embeddings=self.embeddings,
                    patent_id_to_idx={pid: i for i, pid in enumerate(self.patent_ids)}
                )
                update("PyTorch model and feature extractor ready")
            except Exception as e:
                update(f"PyTorch model loading failed: {e}")
                update("Falling back to similarity-based scoring")
                self.pytorch_model = None
        
        self.patents = {}
        self._patents_file_handle = None
        
        self._loaded = True
        update("Ready! (Components loaded on-demand)")
    
    def _load_patent(self, patent_id: str) -> Optional[Dict]:
        """Load a single patent on-demand from the JSONL file."""
        if patent_id in self.patents:
            return self.patents[patent_id]
        
        self._load_patents_batch([patent_id])
        return self.patents.get(patent_id)
    
    def _load_patents_batch(self, patent_ids: List[str], status_callback=None):
        """Load multiple patents efficiently in one pass."""
        if not patent_ids:
            return
        
        to_load = [str(pid) for pid in patent_ids if str(pid) not in self.patents]
        if not to_load:
            return
        
        to_load_set = set(to_load)
        loaded_count = 0
        max_to_load = len(to_load_set)
        
        try:
            import orjson
            use_orjson = True
        except ImportError:
            use_orjson = False
        
        try:
            with open(self.patents_path, 'rb', buffering=1024*1024) as f:
                for line in f:
                    if not line.strip():
                        continue
                    if loaded_count >= max_to_load:
                        break
                    try:
                        if use_orjson:
                            p = orjson.loads(line)
                        else:
                            p = json.loads(line.decode('utf-8'))
                        pid = str(p.get('patent_id', ''))
                        if pid and pid in to_load_set:
                            self.patents[pid] = p
                            loaded_count += 1
                            to_load_set.discard(pid)
                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                        continue
        except Exception as e:
            if status_callback:
                status_callback(f"Warning: Could not load some patents: {e}")
            print(f"Warning: Could not load some patents: {e}")
    
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
        self.load(status_callback=status_callback)
        
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
        
        query_patent = parsed.to_patent_dict()
        # Get query text - prefer abstract, fallback to title, then raw_text
        query_text = query_patent.get('abstract', '') or query_patent.get('title', '') or parsed.raw_text or ''
        query_text = query_text[:500] if query_text else ''
        
        if not query_text or not query_text.strip():
            if status_callback:
                status_callback("ERROR: No valid text found in input. Please provide title, abstract, or description.")
            return AnalysisResult(
                mode="novelty",
                success=False,
                error="No valid text found in input. Please provide a title, abstract, or description of your invention.",
                parsed_input=parsed
            )
        
        extracted_keywords = None
        search_terms = []
        
        if self.use_llm_keywords and self.keyword_extractor:
            if status_callback:
                status_callback("Generating search keywords with LLM...")
            try:
                search_terms = self.keyword_extractor.generate_search_terms(query_text)
                if status_callback:
                    status_callback(f"Generated {len(search_terms)} search terms: {', '.join(search_terms[:3])}...")
                print(f"Generated {len(search_terms)} search terms: {search_terms[:3]}...")
                
                extracted_keywords = self.keyword_extractor.extract_keywords(query_text)
            except Exception as e:
                if status_callback:
                    status_callback(f"WARNING: LLM keyword generation failed, using query text")
                print(f"LLM keyword generation failed: {e}")
                search_terms = [query_text[:200]]
        
        if status_callback:
            status_callback("Searching local database (200K patents)...")
        query_embedding = self.st_model.encode(query_text, show_progress_bar=False)
        local_similar = self._find_similar(query_embedding, top_k=10, status_callback=status_callback)
        if status_callback:
            status_callback(f"Found {len(local_similar)} local patents")
        
        online_patents = []
        search_metadata = {
            "local_count": len(local_similar), 
            "online_count": 0,
            "search_terms": search_terms,
            "patents_per_term": {}
        }
        
        if self.use_online_search and self.online_searcher:
            if not search_terms:
                search_terms = [query_text[:200]]
                print(f"No LLM keywords, using query text for online search")
            
            if status_callback:
                status_callback(f"Searching online patents with {len(search_terms)} terms...")
            
            try:
                print(f"Calling online_searcher.search_multiple_terms with {len(search_terms)} terms")
                print(f"Online searcher type: {type(self.online_searcher)}")
                print(f"Online searcher use_serpapi: {getattr(self.online_searcher, 'use_serpapi', 'N/A')}")
                
                results_by_term = self.online_searcher.search_multiple_terms(
                    search_terms, 
                    max_per_term=10
                )
                
                print(f"search_multiple_terms returned {len(results_by_term)} terms with results")
                
                seen_ids = set()
                for term, results in results_by_term.items():
                    search_metadata["patents_per_term"][term] = len(results)
                    if status_callback:
                        status_callback(f"Term '{term[:50]}...': {len(results)} patents found")
                    print(f"Term '{term[:50]}...': {len(results)} results (type: {type(results)})")
                    
                    if not isinstance(results, list):
                        print(f"WARNING: results is not a list, it's {type(results)}")
                        continue
                    
                    for r in results:
                        if not hasattr(r, 'patent_id'):
                            print(f"WARNING: result object missing patent_id attribute: {type(r)}")
                            continue
                            
                        if r.patent_id not in seen_ids:
                            seen_ids.add(r.patent_id)
                            online_patents.append({
                                'patent_id': r.patent_id,
                                'title': str(r.title) if hasattr(r, 'title') else 'Unknown',
                                'abstract': str(r.abstract) if hasattr(r, 'abstract') else '',
                                'year': r.year if hasattr(r, 'year') else None,
                                'similarity': r.relevance_score * 0.8 if hasattr(r, 'relevance_score') else 0.5,
                                'source': 'online',
                                'url': r.url if hasattr(r, 'url') else None,
                                'inventor': r.inventor if hasattr(r, 'inventor') else None,
                                'assignee': r.assignee if hasattr(r, 'assignee') else None
                            })
                
                search_metadata["online_count"] = len(online_patents)
                if status_callback:
                    status_callback(f"Online search: {len(online_patents)} unique patents found")
                print(f"Online search found {len(online_patents)} unique patents across {len(search_terms)} terms")
            except Exception as e:
                if status_callback:
                    status_callback(f"ERROR: Online search failed: {str(e)[:100]}")
                print(f"Online search failed: {e}")
                import traceback
                traceback.print_exc()
        
        all_similar = self._merge_results(local_similar, online_patents)
        
        # Ranking-based assessment: score top-K candidates
        top_k = min(20, len(all_similar))
        scored_patents = []
        
        if self.pytorch_model and self.feature_extractor and all_similar:
            try:
                if 'embedding' not in query_patent:
                    embed_text = query_patent.get('abstract', '') or query_patent.get('title', '') or parsed.raw_text or ''
                    embed_text = embed_text[:500] if embed_text else ''
                    if embed_text:
                        query_patent['embedding'] = self.st_model.encode(embed_text, show_progress_bar=False)
                
                if status_callback:
                    status_callback(f"Scoring top {top_k} candidates with PyTorch model...")
                
                for i, similar_patent in enumerate(all_similar[:top_k]):
                    patent_id = str(similar_patent['patent_id'])
                    similar_patent_data = self.patents.get(patent_id)
                    
                    if not similar_patent_data:
                        similar_patent_data = self._load_patent(patent_id)
                    
                    if similar_patent_data:
                        try:
                            feature_vector = self.feature_extractor.extract_features(
                                query_patent,
                                similar_patent_data
                            )
                            feature_array = feature_vector.to_array(self.feature_names).reshape(1, -1)
                            similarity_prob = self.pytorch_model.predict_proba(feature_array)[0][1]
                            novelty_prob = 1 - similarity_prob
                            
                            similar_patent['model_similarity'] = float(similarity_prob)
                            similar_patent['model_novelty'] = float(novelty_prob)
                            scored_patents.append(similar_patent)
                        except Exception as e:
                            print(f"Failed to score patent {patent_id}: {e}")
                            similar_patent['model_similarity'] = similar_patent.get('similarity', 0)
                            similar_patent['model_novelty'] = 1 - similar_patent.get('similarity', 0)
                            scored_patents.append(similar_patent)
                    else:
                        similar_patent['model_similarity'] = similar_patent.get('similarity', 0)
                        similar_patent['model_novelty'] = 1 - similar_patent.get('similarity', 0)
                        scored_patents.append(similar_patent)
                
                if scored_patents:
                    # Sort by model similarity (lowest = most novel)
                    scored_patents.sort(key=lambda x: x.get('model_similarity', 1.0))
                    
                    # Add rank to each patent
                    for rank, patent in enumerate(scored_patents, 1):
                        patent['rank'] = rank
                    
                    # Calculate rank distribution metrics
                    model_similarities = [p.get('model_similarity', 1.0) for p in scored_patents]
                    mean_similarity = float(np.mean(model_similarities))
                    mean_novelty = 1 - mean_similarity
                    
                    # Calculate percentile rank (how novel compared to top-K)
                    # Lower similarity = higher novelty
                    # Percentile represents: what % of patents are LESS similar (more novel) than this one
                    # So if percentile = 100%, it means this is the LEAST novel (most similar)
                    # We want to show it as "bottom 100%" or invert to show "top 0%"
                    query_similarity = model_similarities[0] if model_similarities else 1.0
                    # Count how many patents are MORE novel (less similar) than this one
                    more_novel_count = np.sum(np.array(model_similarities) < query_similarity)
                    # Percentile: what % are less similar (more novel)
                    percentile = (more_novel_count / len(model_similarities)) * 100 if model_similarities else 0.0
                    
                    # Novelty score from rank distribution
                    novelty_score = mean_novelty
                    
                    search_metadata['top_k_scored'] = len(scored_patents)
                    search_metadata['rank_percentile'] = float(percentile)
                    search_metadata['mean_similarity'] = float(mean_similarity)
                    
                    print(f"Ranking-based assessment: scored {len(scored_patents)} patents, mean similarity={mean_similarity:.3f}, novelty={mean_novelty:.3f}, percentile={percentile:.1f}%")
                    
                    # Update all_similar with scored patents (in rank order) + unscored patents
                    all_similar = scored_patents + all_similar[top_k:]
                else:
                    max_sim = all_similar[0]['similarity'] if all_similar else 0
                    novelty_score = 1 - max_sim
                    print(f"Using similarity fallback (no patents could be scored)")
            except Exception as e:
                max_sim = all_similar[0]['similarity'] if all_similar else 0
                novelty_score = 1 - max_sim
                print(f"PyTorch scoring failed: {e}, using similarity fallback")
                import traceback
                traceback.print_exc()
        else:
            max_sim = all_similar[0]['similarity'] if all_similar else 0
            novelty_score = 1 - max_sim
            if not self.pytorch_model:
                print(f"PyTorch model not loaded, using similarity-based scoring")
        
        # Determine assessment (for UI display only - actual score is continuous and ranking-based)
        # These thresholds are NOT used for scoring, only for display labels
        if novelty_score > 0.7:
            assessment = "HIGHLY NOVEL"
        elif novelty_score > 0.5:
            assessment = "MODERATELY NOVEL"
        elif novelty_score > 0.3:
            assessment = "LOW NOVELTY"
        else:
            assessment = "NOT NOVEL"
        
        # Generate explanation
        if status_callback:
            status_callback("Generating AI explanation...")
        report = self.explainer.generate_explanation(
            query_patent=query_patent,
            similar_patents=all_similar,
            novelty_score=novelty_score
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
            patentsview_data=None,
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
        
        search_terms = []
        if self.use_llm_keywords and self.keyword_extractor:
            if status_callback:
                status_callback("Generating search keywords...")
            try:
                search_terms = self.keyword_extractor.generate_search_terms(query)
            except Exception:
                search_terms = [query[:200]]
        
        if status_callback:
            status_callback("Searching local database...")
        query_embedding = self.st_model.encode(query, show_progress_bar=False)
        local_similar = self._find_similar(query_embedding, top_k=20, status_callback=status_callback)
        
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
        
        all_results = self._merge_results(local_similar, online_patents)
        
        patent_ids_to_load = [str(p['patent_id']) for p in all_results[:20]]
        self._load_patents_batch(patent_ids_to_load, status_callback=status_callback)
        
        results = []
        for p in all_results[:20]:
            patent_id = str(p['patent_id'])
            patent_data = self.patents.get(patent_id, {})
            if not patent_data:
                patent_data = self._load_patent(patent_id) or {}
            results.append({
                'patent_id': p['patent_id'],
                'similarity': p.get('similarity', 0),
                'title': patent_data.get('title', p.get('title', 'N/A')),
                'abstract': patent_data.get('abstract', p.get('abstract', 'N/A'))[:500],
                'year': patent_data.get('year', p.get('year', 'N/A')),
                'source': p.get('source', 'local')
            })
        
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
    
    def _find_similar(self, query_embedding: np.ndarray, top_k: int = 10, status_callback=None) -> List[Dict]:
        """Find similar patents using cosine similarity."""
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        all_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(all_norms, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        patent_ids_to_load = [self.patent_ids[idx] for idx in top_indices]
        if status_callback:
            status_callback(f"Loading patent details for top {len(patent_ids_to_load)} results...")
        self._load_patents_batch(patent_ids_to_load, status_callback=status_callback)
        
        results = []
        for idx in top_indices:
            pid = self.patent_ids[idx]
            patent_data = self.patents.get(str(pid), {})
            if not patent_data:
                patent_data = self._load_patent(str(pid)) or {}
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
        """Merge local and online search results."""
        seen_ids = set()
        merged = []
        
        for r in local_results:
            pid = str(r.get('patent_id', ''))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                r['source'] = 'local'
                merged.append(r)
        
        for r in online_results:
            pid = str(r.get('patent_id', ''))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                r['source'] = 'online'
                merged.append(r)
        
        merged.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return merged

def demo():
    """Demo the analyzer with different input types."""
    
    print("=" * 60)
    print("PATENT ANALYZER DEMO")
    print("=" * 60)
    
    analyzer = PatentAnalyzer()  # Demo analyzer
    
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
