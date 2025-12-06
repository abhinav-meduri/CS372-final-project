"""
LLM-based Patent Novelty Explainer

Generates human-readable explanations for patent novelty assessments,
citing specific claims and prior art.

Supports multiple backends:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local Llama, Mistral)
- HuggingFace (fallback)
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NoveltyExplanation:
    """Structured novelty explanation."""
    novelty_score: float
    overall_assessment: str
    key_findings: List[str]
    claim_analysis: List[Dict]
    prior_art_citations: List[Dict]
    recommendation: str
    raw_explanation: str


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from prompt."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a patent examination expert. Analyze patents for novelty and provide detailed, citation-backed explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a patent examination expert. Analyze patents for novelty and provide detailed, citation-backed explanations."
        )
        return response.content[0].text


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": f"You are a patent examination expert.\n\n{prompt}",
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


class MockBackend(LLMBackend):
    """Mock backend for testing without API keys."""
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        return """
## Novelty Assessment

Based on my analysis of the submitted patent application against the identified prior art:

### Key Findings:
1. The core technical approach shows significant overlap with existing patents
2. Claims 1-3 contain elements found in prior art
3. Some novel aspects exist in the specific implementation details

### Claim-by-Claim Analysis:
- **Claim 1**: Similar to Prior Art Patent #1, Claim 2. Both describe the same fundamental method.
- **Claim 2**: Partially novel - the specific configuration differs from known approaches.
- **Claim 3**: Novel - no direct prior art found for this specific feature.

### Prior Art Citations:
1. US Patent 11,XXX,XXX - "Similar System" (2023) - High relevance
2. US Patent 11,YYY,YYY - "Related Method" (2022) - Medium relevance

### Recommendation:
Consider narrowing claims 1-2 to focus on the novel aspects identified in claim 3. 
The application may proceed with modifications to distinguish from cited prior art.
"""


class PatentExplainer:
    """
    Main class for generating patent novelty explanations.
    """
    
    def __init__(self, backend: str = "mock", **backend_kwargs):
        """
        Initialize explainer with specified backend.
        
        Args:
            backend: One of "openai", "anthropic", "ollama", "mock"
            **backend_kwargs: Additional arguments for the backend
        """
        self.backend = self._create_backend(backend, **backend_kwargs)
    
    def _create_backend(self, backend: str, **kwargs) -> LLMBackend:
        """Create the appropriate LLM backend."""
        backends = {
            "openai": OpenAIBackend,
            "anthropic": AnthropicBackend,
            "ollama": OllamaBackend,
            "mock": MockBackend
        }
        
        if backend not in backends:
            raise ValueError(f"Unknown backend: {backend}. Choose from {list(backends.keys())}")
        
        return backends[backend](**kwargs)
    
    def _build_prompt(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float,
        patentsview_data: Optional[List[Dict]] = None
    ) -> str:
        """Build the prompt for the LLM."""
        
        prompt = f"""# Patent Novelty Analysis Request

## Query Patent (Under Examination)
**Title:** {query_patent.get('title', 'N/A')}

**Abstract:**
{query_patent.get('abstract', 'N/A')[:1500]}

**Claims:**
"""
        # Add claims if available
        claims = query_patent.get('claims', [])
        if claims:
            for i, claim in enumerate(claims[:5], 1):  # First 5 claims
                claim_text = claim.get('text', str(claim)) if isinstance(claim, dict) else str(claim)
                prompt += f"\nClaim {i}: {claim_text[:500]}..."
        else:
            prompt += "\n(Claims not available)"
        
        prompt += f"""

## Computed Novelty Score: {novelty_score:.2f} / 1.00

## Most Similar Prior Art Patents Found:
"""
        
        for i, patent in enumerate(similar_patents[:5], 1):
            prompt += f"""
### Prior Art {i}: Patent {patent.get('patent_id', 'Unknown')}
- **Similarity Score:** {patent.get('similarity', 0):.3f}
- **Title:** {patent.get('title', 'N/A')[:200]}
- **Year:** {patent.get('year', 'N/A')}
- **Abstract:** {patent.get('abstract', 'N/A')[:500]}...
"""
            # Add claims from prior art if available
            if patent.get('claims'):
                prompt += "- **Key Claims:** "
                for j, claim in enumerate(patent['claims'][:2], 1):
                    claim_text = claim.get('text', str(claim)) if isinstance(claim, dict) else str(claim)
                    prompt += f"\n  - Claim {j}: {claim_text[:300]}..."
        
        # Add PatentsView data if available
        if patentsview_data:
            prompt += "\n\n## Additional Patent Details from USPTO (PatentsView):\n"
            for pv_patent in patentsview_data[:3]:
                prompt += f"""
**Patent {pv_patent.get('patent_number', 'N/A')}:**
- Official Title: {pv_patent.get('title', 'N/A')}
- Filing Date: {pv_patent.get('date', 'N/A')}
- Number of Claims: {pv_patent.get('num_claims', 'N/A')}
- CPC Classifications: {', '.join([c.get('code', '') for c in pv_patent.get('cpc_codes', [])[:3]])}
- Assignee: {', '.join(pv_patent.get('assignees', ['N/A'])[:2])}
"""
        
        prompt += """

## Analysis Request

Please provide a comprehensive novelty assessment including:

1. **Overall Novelty Assessment**: Is this patent application novel? Why or why not?

2. **Claim-by-Claim Analysis**: For each major claim, identify:
   - Whether similar language/concepts exist in prior art
   - Specific prior art patent numbers and claim numbers that overlap
   - Any novel elements not found in prior art

3. **Key Differentiators**: What aspects (if any) distinguish this application from prior art?

4. **Prior Art Citations**: List specific prior art references with:
   - Patent number
   - Relevant claim numbers
   - Why it's relevant

5. **Recommendation**: Should this patent be:
   - Approved as-is
   - Approved with narrower claims
   - Rejected due to lack of novelty

Please cite specific patent numbers and claim numbers throughout your analysis.
"""
        
        return prompt
    
    def explain(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float,
        patentsview_data: Optional[List[Dict]] = None
    ) -> NoveltyExplanation:
        """
        Generate a comprehensive novelty explanation.
        
        Args:
            query_patent: The patent being assessed
            similar_patents: List of similar prior art patents
            novelty_score: Computed novelty score (0-1)
            patentsview_data: Optional additional data from PatentsView API
            
        Returns:
            NoveltyExplanation object with structured analysis
        """
        # Build prompt
        prompt = self._build_prompt(
            query_patent, similar_patents, novelty_score, patentsview_data
        )
        
        # Generate explanation
        raw_explanation = self.backend.generate(prompt)
        
        # Parse into structured format
        explanation = self._parse_explanation(
            raw_explanation, novelty_score, similar_patents
        )
        
        return explanation
    
    def _parse_explanation(
        self,
        raw_text: str,
        novelty_score: float,
        similar_patents: List[Dict]
    ) -> NoveltyExplanation:
        """Parse LLM output into structured format."""
        
        # Determine overall assessment from score
        if novelty_score > 0.7:
            assessment = "HIGHLY NOVEL"
        elif novelty_score > 0.5:
            assessment = "MODERATELY NOVEL"
        elif novelty_score > 0.3:
            assessment = "LOW NOVELTY"
        else:
            assessment = "NOT NOVEL"
        
        # Extract key findings (simple heuristic)
        key_findings = []
        lines = raw_text.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '-', '•', '*')):
                clean_line = line.strip().lstrip('0123456789.-•* ')
                if len(clean_line) > 20:
                    key_findings.append(clean_line[:200])
        
        # Build prior art citations
        citations = []
        for patent in similar_patents[:5]:
            citations.append({
                "patent_id": patent.get('patent_id', 'Unknown'),
                "title": patent.get('title', 'N/A')[:100],
                "similarity": patent.get('similarity', 0),
                "relevance": "High" if patent.get('similarity', 0) > 0.7 else "Medium"
            })
        
        # Determine recommendation
        if novelty_score > 0.6:
            recommendation = "APPROVE - Patent appears sufficiently novel"
        elif novelty_score > 0.4:
            recommendation = "REVISE - Consider narrowing claims to distinguish from prior art"
        else:
            recommendation = "REJECT - Significant overlap with existing prior art"
        
        return NoveltyExplanation(
            novelty_score=novelty_score,
            overall_assessment=assessment,
            key_findings=key_findings[:5],
            claim_analysis=[],  # Would need more sophisticated parsing
            prior_art_citations=citations,
            recommendation=recommendation,
            raw_explanation=raw_text
        )


# Test
if __name__ == "__main__":
    # Test with mock backend
    explainer = PatentExplainer(backend="mock")
    
    query = {
        "title": "Machine Learning Patent Analyzer",
        "abstract": "A system for analyzing patent novelty using ML...",
        "claims": [{"text": "A method comprising..."}]
    }
    
    similar = [
        {"patent_id": "11111111", "title": "Similar System", "similarity": 0.8, "year": 2023},
        {"patent_id": "22222222", "title": "Related Method", "similarity": 0.7, "year": 2022}
    ]
    
    explanation = explainer.explain(query, similar, novelty_score=0.35)
    
    print(f"Assessment: {explanation.overall_assessment}")
    print(f"Recommendation: {explanation.recommendation}")
    print(f"\nExplanation:\n{explanation.raw_explanation}")


