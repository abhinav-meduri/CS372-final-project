"""
PatentsView API Integration

Fetches detailed patent information including:
- Full claims text
- Citations (forward and backward)
- CPC classifications
- Inventor/assignee information
"""

import requests
import json
from typing import Dict, List, Optional
import time


class PatentsViewAPI:
    """Interface to PatentsView API for fetching patent details."""
    
    BASE_URL = "https://api.patentsview.org/patents/query"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize API client.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_patent_details(self, patent_id: str) -> Optional[Dict]:
        """
        Fetch detailed information for a single patent.
        
        Args:
            patent_id: Patent number (e.g., "11234567")
            
        Returns:
            Dictionary with patent details or None if not found
        """
        self._rate_limit()
        
        # Clean patent ID (remove any prefixes)
        clean_id = patent_id.replace("US", "").replace(",", "").strip()
        
        query = {
            "q": {"patent_number": clean_id},
            "f": [
                "patent_number",
                "patent_title",
                "patent_abstract",
                "patent_date",
                "patent_type",
                "patent_num_claims",
                # Claims
                "claim_text",
                "claim_number",
                "claim_dependent",
                # Citations
                "cited_patent_number",
                "cited_patent_title",
                "cited_patent_date",
                "citedby_patent_number",
                # CPC
                "cpc_group_id",
                "cpc_group_title",
                # Inventors
                "inventor_first_name",
                "inventor_last_name",
                # Assignee
                "assignee_organization"
            ],
            "o": {"per_page": 1}
        }
        
        try:
            response = requests.post(
                self.BASE_URL,
                json=query,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("patents") and len(data["patents"]) > 0:
                return self._parse_patent_response(data["patents"][0])
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"API Error for patent {patent_id}: {e}")
            return None
    
    def _parse_patent_response(self, patent_data: Dict) -> Dict:
        """Parse the API response into a cleaner format."""
        result = {
            "patent_number": patent_data.get("patent_number"),
            "title": patent_data.get("patent_title"),
            "abstract": patent_data.get("patent_abstract"),
            "date": patent_data.get("patent_date"),
            "type": patent_data.get("patent_type"),
            "num_claims": patent_data.get("patent_num_claims"),
            "claims": [],
            "citations": [],
            "cited_by": [],
            "cpc_codes": [],
            "inventors": [],
            "assignees": []
        }
        
        # Parse claims
        if patent_data.get("claims"):
            for claim in patent_data["claims"]:
                result["claims"].append({
                    "number": claim.get("claim_number"),
                    "text": claim.get("claim_text"),
                    "is_dependent": claim.get("claim_dependent") == "1"
                })
        
        # Parse citations (patents this patent cites)
        if patent_data.get("cited_patents"):
            for cited in patent_data["cited_patents"]:
                result["citations"].append({
                    "patent_number": cited.get("cited_patent_number"),
                    "title": cited.get("cited_patent_title"),
                    "date": cited.get("cited_patent_date")
                })
        
        # Parse cited_by (patents that cite this patent)
        if patent_data.get("citedby_patents"):
            for citing in patent_data["citedby_patents"]:
                result["cited_by"].append({
                    "patent_number": citing.get("citedby_patent_number")
                })
        
        # Parse CPC codes
        if patent_data.get("cpcs"):
            for cpc in patent_data["cpcs"]:
                result["cpc_codes"].append({
                    "code": cpc.get("cpc_group_id"),
                    "title": cpc.get("cpc_group_title")
                })
        
        # Parse inventors
        if patent_data.get("inventors"):
            for inv in patent_data["inventors"]:
                result["inventors"].append(
                    f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                )
        
        # Parse assignees
        if patent_data.get("assignees"):
            for assignee in patent_data["assignees"]:
                if assignee.get("assignee_organization"):
                    result["assignees"].append(assignee["assignee_organization"])
        
        return result
    
    def get_citation_chain(self, patent_id: str, depth: int = 1) -> List[Dict]:
        """
        Get citation chain for a patent.
        
        Args:
            patent_id: Starting patent
            depth: How many levels of citations to follow
            
        Returns:
            List of patent details in the citation chain
        """
        chain = []
        visited = set()
        
        def fetch_citations(pid: str, current_depth: int):
            if current_depth > depth or pid in visited:
                return
            visited.add(pid)
            
            details = self.get_patent_details(pid)
            if details:
                chain.append(details)
                
                # Follow citations
                for cited in details.get("citations", [])[:5]:  # Limit to 5 per level
                    if cited.get("patent_number"):
                        fetch_citations(cited["patent_number"], current_depth + 1)
        
        fetch_citations(patent_id, 0)
        return chain
    
    def search_patents(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search patents by text query.
        
        Args:
            query_text: Search terms
            limit: Max results to return
            
        Returns:
            List of matching patents
        """
        self._rate_limit()
        
        # Use full-text search
        query = {
            "q": {"_text_any": {"patent_abstract": query_text}},
            "f": [
                "patent_number",
                "patent_title", 
                "patent_abstract",
                "patent_date"
            ],
            "o": {"per_page": limit}
        }
        
        try:
            response = requests.post(
                self.BASE_URL,
                json=query,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("patents", [])
            
        except requests.exceptions.RequestException as e:
            print(f"Search error: {e}")
            return []


# Quick test
if __name__ == "__main__":
    api = PatentsViewAPI()
    
    # Test with a known patent
    print("Testing PatentsView API...")
    details = api.get_patent_details("11234567")
    
    if details:
        print(f"\nPatent: {details['patent_number']}")
        print(f"Title: {details['title']}")
        print(f"Claims: {len(details['claims'])}")
        print(f"Citations: {len(details['citations'])}")
    else:
        print("Patent not found or API error")


