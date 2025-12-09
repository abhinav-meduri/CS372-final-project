"""
Phi-3 Local LLM Explainer for Patent Novelty
Uses Microsoft's Phi-3-mini model locally via Ollama for generating patent novelty explanations.
Ollama backend (fastest, with KV caching and Metal acceleration)
Requires: Ollama running with phi3 model (ollama pull phi3)
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass 
class NoveltyReport:
    """Structured novelty assessment report."""
    novelty_score: float
    assessment: str 
    summary: str
    claim_analysis: List[Dict]
    prior_art_citations: List[Dict]
    recommendation: str
    full_explanation: str


class Phi3OllamaExplainer:
    """
    Phi-3 explainer using Ollama backend.
    
    Benefits:
    - KV caching built-in (automatic)
    - Optimized for Apple Metal (3-5x faster than PyTorch+MPS)
    - Same Phi-3 model, just faster runtime
    """
    
    def __init__(
        self,
        model_name: str = "phi3",
        base_url: str = "http://localhost:11434",
        max_new_tokens: int = 4000
    ):
        """
        Initialize Ollama-based Phi-3 explainer.
        
        Args:
            model_name: Ollama model name (default: phi3)
            base_url: Ollama API URL
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.api_endpoint = f"{base_url}/api/generate"
        
        print(f"Phi-3 Ollama Explainer initialized")
        print(f"  Model: {model_name}")
        print(f"  KV caching: Enabled (built-in)")
        print(f"  Metal acceleration: Enabled")
    
    def load_model(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                if self.model_name not in model_names and f"{self.model_name}:latest" not in [m.get('name') for m in models]:
                    print(f"[WARN] Model '{self.model_name}' not found. Run: ollama pull {self.model_name}")
                else:
                    print(f"[OK] Ollama connected, {self.model_name} ready")
            else:
                print("[WARN] Ollama not responding. Run: ollama serve")
        except requests.exceptions.ConnectionError:
            print("[WARN] Cannot connect to Ollama. Make sure it's running: brew services start ollama")
    
    def _build_prompt(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float
    ) -> str:
        """Build a detailed prompt for Phi-3 with evidence requirements."""
        
        query_abstract = query_patent.get('abstract', 'N/A')[:1000]
        query_title = query_patent.get('title', 'N/A')[:250]
        
        if novelty_score > 0.65:
            suggested_verdict = "NOVEL"
        elif novelty_score > 0.45:
            suggested_verdict = "MODERATELY NOVEL"
        elif novelty_score > 0.25:
            suggested_verdict = "LOW NOVELTY"
        else:
            suggested_verdict = "NOT NOVEL"
        
        prompt = f"""You are a USPTO patent examiner. Provide a detailed novelty analysis with specific evidence.

PATENT APPLICATION UNDER REVIEW

TITLE: {query_title}

ABSTRACT:
{query_abstract}

PRIOR ART SEARCH RESULTS (Novelty Score: {novelty_score:.0%})
"""
        
        for i, p in enumerate(similar_patents[:4], 1):
            similarity = p.get('similarity', 0)
            title = p.get('title', 'N/A')[:200]
            abstract = p.get('abstract', 'N/A')[:600]
            patent_id = p.get('patent_id', 'Unknown')
            year = p.get('year', 'N/A')
            
            prompt += f"""
PRIOR ART {i}: Patent {patent_id} ({year})
Similarity: {similarity:.0%}
Title: {title}
Abstract: {abstract}
"""
        
        prompt += f"""

YOUR DETAILED ANALYSIS (Be thorough and cite evidence)

## VERDICT: {suggested_verdict}

## EXECUTIVE SUMMARY
[Write 3-4 detailed sentences. Reference specific patent numbers and explain WHY this verdict was reached based on the evidence above.]

## TECHNICAL OVERLAP ANALYSIS

For each prior art patent, identify specific overlapping elements:

**Patent {similar_patents[0].get('patent_id', 'X') if similar_patents else 'X'}:**
- Overlapping concepts: [List specific technical elements that appear in BOTH the application and this prior art]
- Key quote from prior art: [Quote a specific phrase that shows overlap]

**Patent {similar_patents[1].get('patent_id', 'Y') if len(similar_patents) > 1 else 'Y'}:**
- Overlapping concepts: [List specific technical elements]
- Key quote from prior art: [Quote a specific phrase]

## NOVEL ELEMENTS (What's NEW in this application)
[List 2-3 specific technical features in the application that are NOT found in ANY of the prior art above. Be specific.]

## RECOMMENDATION

**Decision:** {'APPROVE - Sufficient novelty demonstrated' if novelty_score > 0.6 else 'REVISE - Narrow claims to distinguish from prior art' if novelty_score > 0.35 else 'REJECT - Insufficient novelty'}

**Reasoning:** [Explain in 2-3 sentences why this decision was made, citing specific patents]

**If revision needed:** [Suggest specific claim amendments to improve novelty]

---
This analysis is based on automated similarity scoring and should be verified by a human examiner."""
        
        return prompt
    
    def generate_explanation(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float
    ) -> 'NoveltyReport':
        """
        Generate a novelty explanation using Ollama.
        
        Args:
            query_patent: Patent being assessed
            similar_patents: List of similar prior art
            novelty_score: Computed novelty score (0-1)
            
        Returns:
            NoveltyReport with structured analysis
        """
        prompt = self._build_prompt(
            query_patent, similar_patents, novelty_score
        )
        
        # Call Ollama API
        print("Generating explanation via Ollama (with KV caching)...")
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_new_tokens,
                        "temperature": 0.4,
                        "top_p": 0.9,
                    }
                },
                timeout=180  # 3 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text_response = result.get('response', '')
                
                # Log timing info
                if 'total_duration' in result:
                    duration_sec = result['total_duration'] / 1e9
                    tokens = result.get('eval_count', 0)
                    print(f"[OK] Generated {tokens} tokens in {duration_sec:.1f}s ({tokens/duration_sec:.1f} tok/s)")
                
                # Check if response was truncated
                if result.get('done_reason') == 'length':
                    print("[WARN] Response may be truncated (hit token limit)")
                    text_response += "\n\n[Note: Analysis may be incomplete due to length limits]"
            else:
                print(f"[WARN] Ollama error: {response.status_code}")
                text_response = "Error generating explanation. Please try again."
                
        except requests.exceptions.Timeout:
            print("[WARN] Ollama timeout. Try reducing max_new_tokens.")
            text_response = "Generation timed out. Please try again."
        except requests.exceptions.ConnectionError:
            print("[WARN] Cannot connect to Ollama. Is it running?")
            text_response = "Cannot connect to Ollama. Run: brew services start ollama"
        
        # Parse into structured report
        report = self._parse_response(text_response, novelty_score, similar_patents)
        
        return report
    
    def _parse_response(
        self,
        response: str,
        novelty_score: float,
        similar_patents: List[Dict]
    ) -> 'NoveltyReport':
        """Parse LLM response into structured report."""
        
        # Determine assessment from score
        if novelty_score > 0.7:
            assessment = "NOVEL"
        elif novelty_score > 0.5:
            assessment = "MODERATELY_NOVEL"
        elif novelty_score > 0.3:
            assessment = "LOW_NOVELTY"
        else:
            assessment = "NOT_NOVEL"
        
        # Extract summary
        summary = ""
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'summary' in line.lower() and i + 1 < len(lines):
                summary = lines[i + 1].strip()
                break
        if not summary:
            summary = lines[0][:300] if lines else "Analysis generated."
        
        # Build citations list
        citations = []
        for p in similar_patents[:5]:
            citations.append({
                "patent_id": p.get('patent_id'),
                "title": p.get('title', '')[:100],
                "similarity": p.get('similarity', 0),
                "relevance": "High" if p.get('similarity', 0) > 0.7 else "Medium"
            })
        
        # Recommendation based on score
        if novelty_score > 0.6:
            recommendation = "APPROVE - Patent demonstrates sufficient novelty"
        elif novelty_score > 0.4:
            recommendation = "REVISE - Narrow claims to distinguish from prior art"
        else:
            recommendation = "REJECT - Significant overlap with existing patents"
        
        return NoveltyReport(
            novelty_score=novelty_score,
            assessment=assessment,
            summary=summary,
            claim_analysis=[],
            prior_art_citations=citations,
            recommendation=recommendation,
            full_explanation=response
        )


def get_explainer() -> Phi3OllamaExplainer:
    """
    Factory function to get the Phi-3 explainer (Ollama backend).
    
    Returns:
        Phi3OllamaExplainer instance
    """
    return Phi3OllamaExplainer()


# Test
if __name__ == "__main__":
    # Test with Ollama backend
    explainer = get_explainer()
    
    query = {
        "title": "Machine Learning Patent Analyzer",
        "abstract": "A system for analyzing patent novelty using ML techniques...",
        "claims": [{"text": "A method comprising analyzing patents..."}]
    }
    
    similar = [
        {"patent_id": "11111111", "title": "Similar Patent System", "similarity": 0.75, "year": 2023, "abstract": "A related system..."},
        {"patent_id": "22222222", "title": "Another Method", "similarity": 0.65, "year": 2022, "abstract": "Another approach..."}
    ]
    
    report = explainer.generate_explanation(query, similar, novelty_score=0.35)
    
    print(f"Assessment: {report.assessment}")
    print(f"Summary: {report.summary}")
    print(f"\nFull Explanation:\n{report.full_explanation}")


