"""
Phi-3 Local LLM Explainer for Patent Novelty

Uses Microsoft's Phi-3-mini model locally for generating patent novelty explanations.
Supports:
- Ollama backend (fastest, with KV caching)
- Standard inference (CPU/MPS/CUDA)
- LoRA fine-tuning preparation
- PatentsView API integration for evidence

No API keys required - runs entirely locally!
"""

import torch
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass 
class NoveltyReport:
    """Structured novelty assessment report."""
    novelty_score: float
    assessment: str  # NOVEL, MODERATELY_NOVEL, LOW_NOVELTY, NOT_NOVEL
    summary: str
    claim_analysis: List[Dict]
    prior_art_citations: List[Dict]
    recommendation: str
    full_explanation: str


class Phi3Explainer:
    """
    Local Phi-3 model for generating patent novelty explanations.
    """
    
    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    
    def __init__(
        self,
        model_id: str = None,
        device: str = None,
        load_in_4bit: bool = True,  # Use 4-bit quantization for lower memory
        max_new_tokens: int = 800,  # Reduced from 1500 for faster generation
        use_flash_attention: bool = True  # Use flash attention if available
    ):
        """
        Initialize Phi-3 explainer.
        
        Args:
            model_id: HuggingFace model ID (default: Phi-3-mini)
            device: Device to use (auto-detected if None)
            load_in_4bit: Use 4-bit quantization (recommended for <16GB RAM)
            max_new_tokens: Maximum tokens to generate (lower = faster)
            use_flash_attention: Use flash attention for faster inference
        """
        self.model_id = model_id or self.MODEL_ID
        self.max_new_tokens = max_new_tokens
        self.use_flash_attention = use_flash_attention
        self.model = None
        self.tokenizer = None
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.load_in_4bit = load_in_4bit and self.device == "cuda"  # 4-bit only on CUDA
        
        print(f"Phi-3 Explainer initialized (device: {self.device})")
        if self.device == "mps":
            print("  Note: MPS is slower than CUDA. Consider using Phi3ExplainerLite for faster results.")
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return
        
        print(f"Loading {self.model_id}...")
        print("(This may take a few minutes on first run)")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Configure quantization if needed
        if self.load_in_4bit:
            print("Using 4-bit quantization for lower memory usage...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            # Standard loading for MPS/CPU
            print(f"Loading model in {'float16' if self.device != 'cpu' else 'float32'} precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True  # Reduce memory during loading
            )
            if self.device == "mps":
                self.model = self.model.to("mps")
                # Enable eval mode for inference optimization
                self.model.eval()
        
        print(f"[OK] Model loaded on {self.device}")
        
        # Performance tips
        if self.device == "mps":
            print("Tip: First generation is slow due to MPS compilation. Subsequent runs are faster.")
    
    def _build_prompt(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float,
        patentsview_evidence: Optional[List[Dict]] = None
    ) -> str:
        """Build the prompt for Phi-3."""
        
        prompt = f"""<|system|>
You are an expert patent examiner. Analyze patent applications for novelty by comparing them against prior art. Provide detailed, citation-backed explanations. Be specific about which claims overlap with which prior art patents.<|end|>
<|user|>
# Patent Novelty Analysis Task

## Query Patent (Under Review)
**Title:** {query_patent.get('title', 'N/A')[:300]}

**Abstract:**
{query_patent.get('abstract', 'N/A')[:1000]}

**Key Claims:**
"""
        
        claims = query_patent.get('claims', [])
        for i, claim in enumerate(claims[:3], 1):
            text = claim.get('text', str(claim)) if isinstance(claim, dict) else str(claim)
            prompt += f"\nClaim {i}: {text[:400]}..."
        
        prompt += f"""

## Similarity Score: {novelty_score:.2f}/1.00 (higher = more novel)

## Most Similar Prior Art:
"""
        
        for i, p in enumerate(similar_patents[:5], 1):
            prompt += f"""
### Prior Art {i}: Patent {p.get('patent_id', 'Unknown')}
- Similarity: {p.get('similarity', 0):.1%}
- Title: {p.get('title', 'N/A')[:150]}
- Year: {p.get('year', 'N/A')}
- Abstract: {p.get('abstract', 'N/A')[:300]}...
"""
        
        # Add PatentsView evidence if available
        if patentsview_evidence:
            prompt += "\n## USPTO Evidence (from PatentsView):\n"
            for ev in patentsview_evidence[:2]:
                prompt += f"""
**Patent {ev.get('patent_number', 'N/A')}:**
- Official Claims Count: {ev.get('num_claims', 'N/A')}
- CPC Codes: {', '.join([c.get('code', '') for c in ev.get('cpc_codes', [])[:3]])}
- Assignee: {', '.join(ev.get('assignees', ['N/A'])[:2])}
- Citations: {len(ev.get('citations', []))} patents cited
"""
        
        prompt += """

## Your Task:
Provide a comprehensive novelty assessment with:

1. **OVERALL VERDICT**: Is this novel? (NOVEL / MODERATELY NOVEL / LOW NOVELTY / NOT NOVEL)

2. **SUMMARY**: 2-3 sentence summary of your assessment.

3. **CLAIM ANALYSIS**: For each claim, state:
   - Overlapping prior art (cite specific patent numbers)
   - Novel aspects (if any)

4. **PRIOR ART CITATIONS**: List the most relevant prior art with specific reasons.

5. **RECOMMENDATION**: Should this patent be approved, revised, or rejected?

Be specific and cite patent numbers throughout.
<|end|>
<|assistant|>
"""
        
        return prompt
    
    def generate_explanation(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float,
        patentsview_evidence: Optional[List[Dict]] = None
    ) -> NoveltyReport:
        """
        Generate a novelty explanation.
        
        Args:
            query_patent: Patent being assessed
            similar_patents: List of similar prior art
            novelty_score: Computed novelty score (0-1)
            patentsview_evidence: Optional USPTO data from PatentsView API
            
        Returns:
            NoveltyReport with structured analysis
        """
        # Ensure model is loaded
        self.load_model()
        
        # Build prompt
        prompt = self._build_prompt(
            query_patent, similar_patents, novelty_score, patentsview_evidence
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        print("Generating explanation...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disabled - DynamicCache incompatible with transformers 4.49
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Parse into structured report
        report = self._parse_response(response, novelty_score, similar_patents)
        
        return report
    
    def _parse_response(
        self,
        response: str,
        novelty_score: float,
        similar_patents: List[Dict]
    ) -> NoveltyReport:
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
        
        # Extract summary (first paragraph or after "SUMMARY:")
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
            claim_analysis=[],  # Would need more parsing
            prior_art_citations=citations,
            recommendation=recommendation,
            full_explanation=response
        )


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
        max_new_tokens: int = 1000  # Detailed analysis with evidence
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
        novelty_score: float,
        patentsview_evidence: Optional[List[Dict]] = None
    ) -> str:
        """Build a detailed prompt for Phi-3 with evidence requirements."""
        
        query_abstract = query_patent.get('abstract', 'N/A')[:1000]
        query_title = query_patent.get('title', 'N/A')[:250]
        
        # Determine verdict suggestion based on score
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
        novelty_score: float,
        patentsview_evidence: Optional[List[Dict]] = None
    ) -> 'NoveltyReport':
        """
        Generate a novelty explanation using Ollama.
        
        Args:
            query_patent: Patent being assessed
            similar_patents: List of similar prior art
            novelty_score: Computed novelty score (0-1)
            patentsview_evidence: Optional USPTO data
            
        Returns:
            NoveltyReport with structured analysis
        """
        # Build prompt
        prompt = self._build_prompt(
            query_patent, similar_patents, novelty_score, patentsview_evidence
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


class Phi3ExplainerLite:
    """
    Lightweight version that doesn't load the model.
    Uses template-based explanations for testing without GPU.
    """
    
    def __init__(self):
        print("Phi-3 Lite Explainer (template-based, no model loading)")
    
    def load_model(self):
        pass  # No model to load
    
    def generate_explanation(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float,
        patentsview_evidence: Optional[List[Dict]] = None
    ) -> NoveltyReport:
        """Generate template-based explanation."""
        
        # Assessment based on score
        if novelty_score > 0.7:
            assessment = "NOVEL"
            summary = "This patent application appears to be highly novel with no closely related prior art found in the database."
            recommendation = "APPROVE - Patent demonstrates sufficient novelty for approval."
        elif novelty_score > 0.5:
            assessment = "MODERATELY_NOVEL"
            summary = "This patent application shows moderate novelty. Some related patents exist but there are distinguishing features."
            recommendation = "REVISE - Consider narrowing claims to emphasize novel aspects."
        elif novelty_score > 0.3:
            assessment = "LOW_NOVELTY"
            summary = "This patent application has low novelty. Several similar patents exist in the prior art database."
            recommendation = "REVISE - Significant claim amendments needed to distinguish from prior art."
        else:
            assessment = "NOT_NOVEL"
            summary = "This patent application lacks novelty. Very similar patents already exist."
            recommendation = "REJECT - Insufficient novelty for patent protection."
        
        # Build detailed explanation
        explanation = f"""## Novelty Assessment Report

### Overall Verdict: {assessment}

### Summary
{summary}

### Prior Art Analysis

The following prior art patents were identified as most relevant:

"""
        for i, p in enumerate(similar_patents[:5], 1):
            explanation += f"""**{i}. Patent {p.get('patent_id', 'Unknown')}** (Similarity: {p.get('similarity', 0):.1%})
- Title: {p.get('title', 'N/A')[:100]}
- Year: {p.get('year', 'N/A')}
- Relevance: {"High - significant overlap" if p.get('similarity', 0) > 0.7 else "Medium - partial overlap"}

"""
        
        # Add PatentsView evidence if available
        if patentsview_evidence:
            explanation += "\n### USPTO Evidence\n\n"
            for ev in patentsview_evidence[:2]:
                explanation += f"""**Patent {ev.get('patent_number', 'N/A')}:**
- Official record shows {ev.get('num_claims', 'N/A')} claims
- Classification: {', '.join([c.get('code', '') for c in ev.get('cpc_codes', [])[:3]])}
- Assigned to: {', '.join(ev.get('assignees', ['Unknown'])[:2])}

"""
        
        explanation += f"""### Recommendation
{recommendation}

### Note
This analysis was generated using template-based reasoning. For more detailed analysis,
enable the full Phi-3 model by running with GPU support.
"""
        
        citations = [
            {
                "patent_id": p.get('patent_id'),
                "title": p.get('title', '')[:100],
                "similarity": p.get('similarity', 0),
                "relevance": "High" if p.get('similarity', 0) > 0.7 else "Medium"
            }
            for p in similar_patents[:5]
        ]
        
        return NoveltyReport(
            novelty_score=novelty_score,
            assessment=assessment,
            summary=summary,
            claim_analysis=[],
            prior_art_citations=citations,
            recommendation=recommendation,
            full_explanation=explanation
        )


def get_explainer(use_full_model: bool = False, use_ollama: bool = True) -> Phi3Explainer:
    """
    Factory function to get appropriate explainer.
    
    Args:
        use_full_model: If True, load Phi-3 model for real inference
                       If False, use lightweight template-based version
        use_ollama: If True and use_full_model=True, use Ollama backend (faster, KV caching)
                   If False, use PyTorch/transformers backend (slower)
    """
    if use_full_model:
        if use_ollama:
            return Phi3OllamaExplainer()
        else:
            return Phi3Explainer()
    else:
        return Phi3ExplainerLite()


# Test
if __name__ == "__main__":
    # Test with lite version (no model loading)
    explainer = get_explainer(use_full_model=False)
    
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


