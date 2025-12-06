"""
Online Patent Search Module

Adds online search capability to complement local database:
1. LLM-based keyword extraction from user input (Phi-3)
2. Google Patents search via SerpAPI (millions of patents)
3. Result merging with local FAISS results

This creates a true hybrid RAG system:
- Local: Fast, controlled, 200K patents (PatentSBERTa + FAISS)
- Online: Comprehensive, real-time, millions of patents (SerpAPI + Google Patents)
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import serpapi
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    logger.warning("serpapi package not installed. Run: pip install google-search-results")


@dataclass
class PatentSearchResult:
    """Unified patent search result from Google Patents or PatentsView."""
    patent_id: str
    title: str
    abstract: str
    year: Optional[int]
    source: str  # 'local', 'google_patents', or 'patentsview_api'
    relevance_score: float
    url: Optional[str] = None
    inventor: Optional[str] = None
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'patent_id': self.patent_id,
            'title': self.title,
            'abstract': self.abstract,
            'year': self.year,
            'source': self.source,
            'similarity': self.relevance_score,  # For compatibility with existing code
            'relevance_score': self.relevance_score,
            'url': self.url,
            'inventor': self.inventor,
            'assignee': self.assignee,
            'filing_date': self.filing_date
        }


class LLMKeywordExtractor:
    """
    Extract optimized search keywords from user input using Phi-3.
    
    Generates Google Patents-compatible search queries like:
    - (rabbit toy)
    - (coffee brew) AND (pot) OR (top)
    - (stabilization system)
    - (vr heading) OR (logic freq)
    
    Similar to the reference implementation using GPT-4.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", num_search_terms: int = 5):
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.num_search_terms = num_search_terms
    
    def generate_search_terms(self, user_input: str) -> List[str]:
        """
        Generate optimized Google Patents search queries.
        
        Similar to reference: generate_search_terms()
        
        Returns:
            List of search query strings optimized for Google Patents
        """
        prompt = f"""As a search specialist with expertise in optimizing searches in the Google Patents database, your task is to generate {self.num_search_terms} optimal keyword or keyword list searches to find similar patents.

RULES:
- Generate single and multiple keywords
- Use parentheses for grouping: (term1) AND (term2) OR (term3)
- Don't be too specific - aim for at least 10 results per query
- Focus on technical terms and key concepts

INVENTION IDEA:
---BEGINNING---
{user_input}
---END---

OUTPUT: Return ONLY a numbered list of search queries, one per line:
1. (first search query)
2. (second search query)
...
"""

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": "phi3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 500, "temperature": 0.3}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                
                # Parse numbered list
                queries = []
                for line in text.split('\n'):
                    line = line.strip()
                    # Match "1. query" or "1) query" patterns
                    match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                    if match:
                        query = match.group(1).strip()
                        if query:
                            queries.append(query)
                
                if queries:
                    logger.info(f"Generated {len(queries)} search terms via LLM")
                    return queries[:self.num_search_terms]
                
            return self._fallback_search_terms(user_input)
            
        except Exception as e:
            logger.warning(f"LLM search term generation failed: {e}")
            return self._fallback_search_terms(user_input)
    
    def extract_keywords(self, user_input: str) -> Dict:
        """
        Extract structured keywords from user input.
        
        Returns:
            Dict with: main_concept, technical_terms, search_queries, suggested_cpc
        """
        prompt = f"""Extract search keywords from this patent idea. Be specific and technical.

INPUT:
{user_input}

OUTPUT (JSON format):
{{
    "main_concept": "one phrase describing the core invention",
    "technical_terms": ["list", "of", "5-10", "technical", "keywords"],
    "search_queries": ["(query1) AND (query2)", "(other query)"],
    "suggested_cpc": ["A61K", "G06F"],
    "invention_type": "method/device/composition/system"
}}

Respond with ONLY the JSON, no explanation."""

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": "phi3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 500, "temperature": 0.2}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                
                # Parse JSON from response
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    # Also generate proper search terms
                    if not parsed.get('search_queries'):
                        parsed['search_queries'] = self._fallback_search_terms(user_input)
                    return parsed
                
                return self._fallback_extraction(user_input)
            
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")
            return self._fallback_extraction(user_input)
    
    def _fallback_search_terms(self, text: str) -> List[str]:
        """Generate simple search terms as fallback."""
        from collections import Counter
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'for', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'with',
                      'that', 'this', 'which', 'using', 'based', 'system', 'method'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        common = Counter(keywords).most_common(10)
        
        # Create search queries
        queries = []
        if len(common) >= 2:
            queries.append(f"({common[0][0]}) AND ({common[1][0]})")
        if len(common) >= 4:
            queries.append(f"({common[2][0]}) OR ({common[3][0]})")
        if len(common) >= 1:
            queries.append(f"({common[0][0]})")
        
        return queries if queries else [text[:100]]
    
    def _fallback_extraction(self, text: str) -> Dict:
        """Simple fallback keyword extraction."""
        from collections import Counter
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'for', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'with'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        common = Counter(keywords).most_common(10)
        
        return {
            "main_concept": text[:100],
            "technical_terms": [w for w, _ in common],
            "search_queries": self._fallback_search_terms(text),
            "suggested_cpc": [],
            "invention_type": "unknown"
        }


class GooglePatentsSearch:
    """
    Search Google Patents using SerpAPI.
    
    SerpAPI provides access to Google Patents with:
    - Millions of patents worldwide
    - Full patent metadata
    - Scholar citations
    - Clustered results
    
    Fallback to PatentsView API if no SerpAPI key.
    """
    
    def __init__(self, serpapi_key: Optional[str] = None):
        """
        Initialize Google Patents search.
        
        Args:
            serpapi_key: SerpAPI key (get from https://serpapi.com)
        """
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY')
        key_valid = self.serpapi_key and len(str(self.serpapi_key).strip()) > 0
        self.use_serpapi = SERPAPI_AVAILABLE and key_valid
        
        if self.use_serpapi:
            logger.info(f"[OK] Using SerpAPI for Google Patents search (millions of patents)")
            logger.info(f"  API key configured: {str(self.serpapi_key)[:8]}...")
        else:
            logger.warning("[WARN] SerpAPI not configured. Using PatentsView API fallback.")
            if not SERPAPI_AVAILABLE:
                logger.warning("  Install with: pip install google-search-results")
            if not key_valid:
                logger.warning("  Set SERPAPI_KEY environment variable or pass to constructor")
    
    def search(self, query: str, max_results: int = 10) -> List[PatentSearchResult]:
        """
        Search Google Patents.
        
        Args:
            query: Search query (supports Google Patents syntax)
            max_results: Maximum results to return
            
        Returns:
            List of PatentSearchResult
        """
        if self.use_serpapi:
            return self._search_serpapi(query, max_results)
        else:
            return self._search_patentsview(query, max_results)
    
    def search_multiple_terms(self, terms: List[str], max_per_term: int = 10) -> Dict[str, List[PatentSearchResult]]:
        """
        Search multiple terms and return results grouped by term.
        
        Similar to the reference implementation's search_on_google_patents().
        
        Args:
            terms: List of search terms
            max_per_term: Max results per term
            
        Returns:
            Dict mapping term -> list of results
        """
        logger.info(f"Searching {len(terms)} terms with max_per_term={max_per_term}")
        logger.info(f"Using SerpAPI: {self.use_serpapi}, API key present: {bool(self.serpapi_key)}")
        
        results_by_term = {}
        
        for term in terms:
            logger.info(f"Searching term: '{term[:100]}...'")
            results = self.search(term, max_per_term)
            results_by_term[term] = results
            logger.info(f"  '{term[:50]}...': {len(results)} patents found")
            if results:
                logger.info(f"    First result: {results[0].patent_id} - {results[0].title[:50]}...")
        
        return results_by_term
    
    def _search_serpapi(self, query: str, max_results: int) -> List[PatentSearchResult]:
        """
        Search using SerpAPI's Google Patents engine.
        
        Follows the reference implementation pattern.
        """
        if not SERPAPI_AVAILABLE:
            logger.error("serpapi package not installed")
            return []
        
        try:
            if not self.serpapi_key:
                logger.warning("SerpAPI key not provided")
                return []
            
            logger.info(f"Searching SerpAPI with query: {query[:100]}...")
            logger.info(f"API key present: {bool(self.serpapi_key)} (first 8 chars: {str(self.serpapi_key)[:8]}...)")
            
            # SerpAPI Google Patents requires num to be between 10 and 100
            num_results = max(10, min(max_results, 100))
            if max_results < 10:
                logger.warning(f"SerpAPI requires num >= 10, adjusting from {max_results} to {num_results}")
            
            params = {
                "engine": "google_patents",
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results
            }
            
            logger.info(f"Making SerpAPI request with params: engine=google_patents, q={query[:50]}..., num={num_results} (requested {max_results})")
            search = GoogleSearch(params)
            search_results = search.get_dict()
            
            logger.info(f"SerpAPI response keys: {list(search_results.keys())[:10]}")
            
            if search_results.get('error'):
                logger.error(f"SerpAPI error: {search_results['error']}")
                return []
            
            organic_results = search_results.get("organic_results", [])
            logger.info(f"Found {len(organic_results)} organic results")
            
            results = []
            
            for result in organic_results[:max_results]:
                # Extract patent info following reference implementation
                patent_id = result.get("publication_number") or result.get("patent_id") or result.get("patent_number")
                
                if not patent_id:
                    logger.warning(f"Skipping result without patent_id: {result.get('title', 'Unknown')[:50]}")
                    continue
                    
                    results.append(PatentSearchResult(
                    patent_id=str(patent_id),
                        title=result.get("title", "Unknown"),
                        abstract=result.get("snippet", result.get("abstract", "")),
                        year=self._extract_year(result.get("publication_date", "")),
                        source='google_patents',
                        relevance_score=1.0 - (len(results) * 0.05),  # Decay by position
                    url=result.get("link", result.get("serpapi_link", f"https://patents.google.com/patent/{patent_id}")),
                        inventor=result.get("inventor"),
                        assignee=result.get("assignee"),
                        filing_date=result.get("filing_date")
                    ))
            
            logger.info(f"SerpAPI found {len(results)} patents for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _search_patentsview(self, query: str, max_results: int) -> List[PatentSearchResult]:
        """
        Fallback search using free PatentsView API.
        
        Note: PatentsView has fewer patents than Google Patents.
        """
        try:
            response = requests.post(
                "https://api.patentsview.org/patents/query",
                json={
                    "q": {"_text_any": {"patent_abstract": query}},
                    "f": ["patent_number", "patent_title", "patent_abstract", "patent_date"],
                    "o": {"per_page": max_results}
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for patent in data.get('patents', [])[:max_results]:
                    year = None
                    if patent.get('patent_date'):
                        try:
                            year = int(patent['patent_date'][:4])
                        except:
                            pass
                    
                    patent_num = patent.get('patent_number', 'Unknown')
                    results.append(PatentSearchResult(
                        patent_id=patent_num,
                        title=patent.get('patent_title', 'Unknown'),
                        abstract=patent.get('patent_abstract', ''),
                        year=year,
                        source='patentsview_api',
                        relevance_score=1.0 - (len(results) * 0.05),
                        url=f"https://patents.google.com/patent/US{patent_num}"
                    ))
                
                logger.info(f"PatentsView found {len(results)} patents")
                return results
                
        except Exception as e:
            logger.warning(f"PatentsView API search failed: {e}")
        
        return []
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        try:
            match = re.search(r'(\d{4})', date_str)
            if match:
                return int(match.group(1))
        except:
            pass
        return None


class HybridPatentSearch:
    """
    Hybrid search combining local database with online Google Patents.
    
    Architecture (similar to reference implementation):
    1. Generate search terms using LLM (Phi-3)
    2. Search local FAISS + BM25 database
    3. Search online Google Patents via SerpAPI
    4. Merge and deduplicate results
    5. Optionally check similarity using LLM
    """
    
    def __init__(
        self,
        local_searcher=None,  # Your existing FAISS/BM25 searcher
        serpapi_key: Optional[str] = None,
        use_online: bool = True,
        num_search_terms: int = 5
    ):
        self.local_searcher = local_searcher
        self.keyword_extractor = LLMKeywordExtractor(num_search_terms=num_search_terms)
        self.online_searcher = GooglePatentsSearch(serpapi_key) if use_online else None
        self.use_online = use_online
    
    def search(
        self,
        user_input: str,
        max_local: int = 50,
        max_per_term: int = 10
    ) -> Tuple[List[PatentSearchResult], Dict]:
        """
        Perform hybrid search.
        
        Similar to reference implementation workflow:
        1. generate_search_terms()
        2. search_on_google_patents()
        3. merge and deduplicate
        
        Returns:
            Tuple of (results, metadata)
        """
        metadata = {
            "keywords_extracted": {},
            "search_terms": [],
            "local_count": 0,
            "online_count": 0,
            "patents_per_term": {},
            "total_unique": 0
        }
        
        all_results = []
        seen_ids = set()
        
        # Step 1: Generate search terms using LLM
        logger.info("Generating search terms with LLM...")
        keywords = self.keyword_extractor.extract_keywords(user_input)
        metadata["keywords_extracted"] = keywords
        
        search_terms = keywords.get('search_queries', [])
        if not search_terms:
            search_terms = self.keyword_extractor.generate_search_terms(user_input)
        metadata["search_terms"] = search_terms
        
        logger.info(f"Generated {len(search_terms)} search terms: {search_terms}")
        
        # Step 2: Local search (if available)
        if self.local_searcher is not None:
            logger.info("Searching local database...")
            # Implement based on your existing searcher
            pass
        
        # Step 3: Online search with multiple terms
        if self.use_online and self.online_searcher is not None:
            logger.info("Searching Google Patents online...")
            
            # Search each term (like reference: search_on_google_patents)
            results_by_term = self.online_searcher.search_multiple_terms(
                search_terms, 
                max_per_term=max_per_term
            )
            
            for term, results in results_by_term.items():
                metadata["patents_per_term"][term] = len(results)
                
                for result in results:
                    if result.patent_id not in seen_ids:
                        seen_ids.add(result.patent_id)
                        all_results.append(result)
                
                metadata["online_count"] += len(results)
        
        metadata["total_unique"] = len(all_results)
        
        # Step 4: Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Hybrid search complete: {len(all_results)} unique patents")
        
        return all_results, metadata
    
    def search_and_rank(
        self,
        user_input: str,
        max_results: int = 20
    ) -> Tuple[List[Dict], Dict]:
        """
        Search and return results as dictionaries.
        
        Convenience method that converts PatentSearchResult to dict.
        """
        results, metadata = self.search(user_input, max_per_term=max_results // 3)
        
        # Convert to dicts
        dict_results = [r.to_dict() for r in results[:max_results]]
        
        return dict_results, metadata


def demo_keyword_extraction():
    """Demo the LLM keyword extraction (like reference generate_search_terms)."""
    print("=" * 60)
    print("LLM SEARCH TERM GENERATION DEMO")
    print("(Similar to reference: generate_search_terms)")
    print("=" * 60)
    
    extractor = LLMKeywordExtractor(num_search_terms=5)
    
    test_input = """
    A machine learning system for analyzing patent novelty that uses 
    natural language processing to compare patent claims against prior art.
    The system includes a neural network classifier trained on patent embeddings
    and generates explanations using a large language model.
    """
    
    print(f"\nInput: {test_input[:150]}...")
    
    print("\n1. Generating search terms...")
    search_terms = extractor.generate_search_terms(test_input)
    print(f"\nGenerated {len(search_terms)} search terms:")
    for i, term in enumerate(search_terms, 1):
        print(f"  {i}. {term}")
    
    print("\n2. Extracting structured keywords...")
    keywords = extractor.extract_keywords(test_input)
    print("\nExtracted Keywords:")
    print(json.dumps(keywords, indent=2))


def demo_online_search():
    """Demo the Google Patents search via SerpAPI."""
    print("\n" + "=" * 60)
    print("GOOGLE PATENTS SEARCH DEMO")
    print("(Similar to reference: search_on_google_patents)")
    print("=" * 60)
    
    serpapi_key = os.environ.get('SERPAPI_KEY')
    
    if not serpapi_key:
        print("\n[WARN] SERPAPI_KEY not set. Using PatentsView fallback.")
        print("  For full Google Patents access, set: export SERPAPI_KEY=your_key")
    
    searcher = GooglePatentsSearch(serpapi_key)
    
    # Single query search
    query = "(machine learning) AND (patent classification)"
    print(f"\nSearching for: {query}")
    
    results = searcher.search(query, max_results=5)
    
    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.patent_id}")
        print(f"   Title: {r.title[:80]}...")
        print(f"   Year: {r.year}")
        print(f"   Source: {r.source}")
        if r.assignee:
            print(f"   Assignee: {r.assignee}")


def demo_hybrid_search():
    """Demo the full hybrid search pipeline."""
    print("\n" + "=" * 60)
    print("HYBRID PATENT SEARCH DEMO")
    print("(LLM Keywords + Google Patents + Local)")
    print("=" * 60)
    
    hybrid = HybridPatentSearch(use_online=True, num_search_terms=3)
    
    test_input = """
    A smart water bottle with embedded sensors that tracks hydration levels 
    in real-time. It connects to a smartphone app via Bluetooth and uses 
    machine learning to provide personalized hydration recommendations.
    """
    
    print(f"\nInput: {test_input[:100]}...")
    print("\nRunning hybrid search...")
    
    results, metadata = hybrid.search_and_rank(test_input, max_results=10)
    
    print(f"\n--- SEARCH METADATA ---")
    print(f"Search terms generated: {metadata['search_terms']}")
    print(f"Online patents found: {metadata['online_count']}")
    print(f"Unique patents: {metadata['total_unique']}")
    
    print(f"\n--- TOP RESULTS ---")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['patent_id']} ({r['source']})")
        print(f"   Title: {r['title'][:60]}...")
        print(f"   Score: {r['relevance_score']:.2f}")


if __name__ == "__main__":
    demo_keyword_extraction()
    demo_online_search()
    demo_hybrid_search()

