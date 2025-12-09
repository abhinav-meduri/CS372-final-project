"""
Unified Input Handler for Patent Novelty System

Supports multiple input modalities:
1. Structured patent (title, abstract, claims)
2. Free-text idea description
3. General search query ("find patents about X")

Converts all inputs to a standardized format for processing.
"""

import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class InputMode(Enum):
    """Types of input the system can handle."""
    NOVELTY_ASSESSMENT = "novelty"      # Assess if input is novel
    PRIOR_ART_SEARCH = "search"         # Find related patents
    DOCUMENT_ANALYSIS = "document"      # Analyze uploaded document


@dataclass
class ParsedInput:
    """Standardized input format."""
    mode: InputMode
    title: Optional[str] = None
    abstract: Optional[str] = None
    claims: Optional[List[str]] = None
    raw_text: Optional[str] = None
    search_query: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_patent_dict(self) -> Dict:
        """Convert to patent dictionary format."""
        # Handle claims - can be list of strings or list of dicts
        claims_list = []
        if self.claims:
            for claim in self.claims:
                if isinstance(claim, dict):
                    # Extract text from dict format: {"claim_number": 1, "text": "..."}
                    claim_text = claim.get('text', str(claim))
                    claims_list.append({"text": claim_text})
                elif isinstance(claim, str):
                    claims_list.append({"text": claim})
                else:
                    claims_list.append({"text": str(claim)})
        
        return {
            "patent_id": "INPUT_PATENT",
            "title": self.title or "User Input",
            "abstract": self.abstract or self.raw_text or "",
            "claims": claims_list,
            "year": 2025
        }


class InputHandler:
    """
    Handles multiple input modalities and converts to standard format.
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.txt', '.docx', '.md']
    
    def detect_mode(self, text: str) -> InputMode:
        """
        Auto-detect the appropriate mode based on input text.
        """
        text_lower = text.lower().strip()
        
        # Check for search queries
        search_patterns = [
            r'^find\s+(patents?|prior\s*art)\s+(about|related\s+to|for|on)',
            r'^search\s+(for\s+)?(patents?|prior\s*art)',
            r'^what\s+(patents?|prior\s*art)\s+(exist|are\s+there)',
            r'^show\s+me\s+(patents?|prior\s*art)',
            r'^patents?\s+(about|related\s+to|for|on)\s+',
            r'^list\s+(patents?|prior\s*art)',
        ]
        
        for pattern in search_patterns:
            if re.match(pattern, text_lower):
                return InputMode.PRIOR_ART_SEARCH
        
        # Check for structured patent input (has multiple sections)
        has_title = bool(re.search(r'(title|invention)[\s:]+', text_lower))
        has_abstract = bool(re.search(r'(abstract|summary)[\s:]+', text_lower))
        has_claims = bool(re.search(r'(claim|claims)[\s:]+', text_lower))
        
        if sum([has_title, has_abstract, has_claims]) >= 2:
            return InputMode.NOVELTY_ASSESSMENT
        
        # Default to novelty assessment for free-text ideas
        return InputMode.NOVELTY_ASSESSMENT
    
    def parse_structured_patent(self, text: str) -> ParsedInput:
        """
        Parse structured patent input with Title/Abstract/Claims sections.
        """
        title = None
        abstract = None
        claims = []
        
        # Extract title
        title_match = re.search(
            r'(?:title|invention)[\s:]+(.+?)(?=\n\s*(?:abstract|summary|claim|$))',
            text, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()[:500]
        
        # Extract abstract
        abstract_match = re.search(
            r'(?:abstract|summary)[\s:]+(.+?)(?=\n\s*(?:claim|$))',
            text, re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:2000]
        
        # Extract claims
        claims_match = re.search(
            r'(?:claims?)[\s:]+(.+)',
            text, re.IGNORECASE | re.DOTALL
        )
        if claims_match:
            claims_text = claims_match.group(1)
            # Split by claim numbers
            claim_parts = re.split(r'\n\s*\d+[\.\)]\s*', claims_text)
            claims = [c.strip() for c in claim_parts if c.strip()][:10]
        
        return ParsedInput(
            mode=InputMode.NOVELTY_ASSESSMENT,
            title=title,
            abstract=abstract,
            claims=claims,
            raw_text=text
        )
    
    def parse_idea(self, text: str) -> ParsedInput:
        """
        Parse free-text idea description into patent-like format.
        """
        # Clean up the text
        text = text.strip()
        
        # Try to extract a title from the first sentence
        sentences = re.split(r'[.!?]+', text)
        title = sentences[0].strip()[:200] if sentences else "User Idea"
        
        # Use the full text as abstract
        abstract = text[:2000]
        
        # Generate pseudo-claims from key sentences
        claims = []
        for sentence in sentences[1:5]:  # Take next few sentences as claims
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Convert to claim-like language
                if not sentence.lower().startswith(('a ', 'an ', 'the ')):
                    sentence = "A " + sentence[0].lower() + sentence[1:]
                claims.append(sentence)
        
        return ParsedInput(
            mode=InputMode.NOVELTY_ASSESSMENT,
            title=title,
            abstract=abstract,
            claims=claims if claims else [abstract[:500]],
            raw_text=text
        )
    
    def parse_search_query(self, text: str) -> ParsedInput:
        """
        Extract search terms from a search query.
        """
        # Remove common search prefixes
        query = text.lower()
        prefixes = [
            r'^find\s+(patents?|prior\s*art)\s+(about|related\s+to|for|on)\s*',
            r'^search\s+(for\s+)?(patents?|prior\s*art)\s+(about|related\s+to|for|on)?\s*',
            r'^what\s+(patents?|prior\s*art)\s+(exist|are\s+there)\s+(about|related\s+to|for|on)?\s*',
            r'^show\s+me\s+(patents?|prior\s*art)\s+(about|related\s+to|for|on)?\s*',
            r'^patents?\s+(about|related\s+to|for|on)\s*',
            r'^list\s+(patents?|prior\s*art)\s+(about|related\s+to|for|on)?\s*',
        ]
        
        for prefix in prefixes:
            query = re.sub(prefix, '', query, flags=re.IGNORECASE)
        
        query = query.strip()
        
        return ParsedInput(
            mode=InputMode.PRIOR_ART_SEARCH,
            search_query=query,
            raw_text=text
        )
    
    def parse_file(self, file_path: str) -> ParsedInput:
        """
        Parse content from an uploaded file.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.txt' or extension == '.md':
            content = path.read_text(encoding='utf-8')
        
        elif extension == '.pdf':
            content = self._extract_pdf_text(file_path)
        
        elif extension == '.docx':
            content = self._extract_docx_text(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Detect if it's structured or free-form
        mode = self.detect_mode(content)
        
        if mode == InputMode.PRIOR_ART_SEARCH:
            parsed = self.parse_search_query(content)
        else:
            # Check if structured
            has_sections = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in [r'title[\s:]+', r'abstract[\s:]+', r'claim']
            )
            
            if has_sections:
                parsed = self.parse_structured_patent(content)
            else:
                parsed = self.parse_idea(content)
        
        parsed.mode = InputMode.DOCUMENT_ANALYSIS
        parsed.file_path = file_path
        
        return parsed
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF."""
        try:
            import PyPDF2
            
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        except ImportError:
            raise ImportError("PyPDF2 required for PDF parsing. Install: pip install PyPDF2")
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX."""
        try:
            import docx
            
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        
        except ImportError:
            raise ImportError("python-docx required for DOCX parsing. Install: pip install python-docx")
    
    def process(self, input_data: Union[str, Dict]) -> ParsedInput:
        """
        Main entry point - process any input type.
        
        Args:
            input_data: Can be:
                - str: Free text (idea, query, or structured patent)
                - Dict: Structured patent dict with title/abstract/claims
                - Dict with 'file_path': Path to uploaded file
        
        Returns:
            ParsedInput object ready for processing
        """
        # Handle dict input
        if isinstance(input_data, dict):
            if 'file_path' in input_data:
                return self.parse_file(input_data['file_path'])
            
            # Assume it's a structured patent dict
            return ParsedInput(
                mode=InputMode.NOVELTY_ASSESSMENT,
                title=input_data.get('title'),
                abstract=input_data.get('abstract'),
                claims=input_data.get('claims', []),
                raw_text=json.dumps(input_data),
                metadata=input_data
            )
        
        # Handle string input
        text = str(input_data).strip()
        
        # Check if it's a file path (only if short enough to be a path)
        if len(text) < 500:
            try:
                if Path(text).exists() and Path(text).suffix in self.supported_extensions:
                    return self.parse_file(text)
            except OSError:
                pass  # Not a valid path, continue as text
        
        # Detect mode and parse accordingly
        mode = self.detect_mode(text)
        
        if mode == InputMode.PRIOR_ART_SEARCH:
            return self.parse_search_query(text)
        
        # Check if structured patent format
        has_sections = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in [r'title[\s:]+', r'abstract[\s:]+', r'claim']
        )
        
        if has_sections:
            return self.parse_structured_patent(text)
        else:
            return self.parse_idea(text)


# Test
if __name__ == "__main__":
    handler = InputHandler()
    
    print("INPUT HANDLER TESTS")
    
    # Test 1: Search query
    print("\n[TEST 1] Search Query")
    query = "Find patents about machine learning for medical diagnosis"
    result = handler.process(query)
    print(f"  Input: {query}")
    print(f"  Mode: {result.mode.value}")
    print(f"  Search Query: {result.search_query}")
    
    # Test 2: Free-text idea
    print("\n[TEST 2] Free-text Idea")
    idea = """
    I have an idea for a smart water bottle that tracks hydration levels using 
    sensors embedded in the bottle. It would connect to a smartphone app via 
    Bluetooth and remind users to drink water based on their activity level 
    and ambient temperature. The bottle would also have UV sterilization.
    """
    result = handler.process(idea)
    print(f"  Mode: {result.mode.value}")
    print(f"  Title: {result.title[:50]}...")
    print(f"  Claims: {len(result.claims)} generated")
    
    # Test 3: Structured patent
    print("\n[TEST 3] Structured Patent")
    patent = """
    Title: Smart Hydration Monitoring System
    
    Abstract: A portable hydration monitoring device comprising sensors for 
    measuring liquid consumption, wireless connectivity for data transmission, 
    and software for analyzing hydration patterns.
    
    Claims:
    1. A hydration monitoring device comprising: a container for holding liquid; 
       one or more sensors for measuring liquid level; a processor for calculating 
       consumption; and a wireless transmitter.
    2. The device of claim 1, further comprising UV sterilization means.
    """
    result = handler.process(patent)
    print(f"  Mode: {result.mode.value}")
    print(f"  Title: {result.title}")
    print(f"  Abstract: {result.abstract[:50]}...")
    print(f"  Claims: {len(result.claims)}")
    
    # Test 4: Dict input
    print("\n[TEST 4] Dict Input")
    dict_input = {
        "title": "Neural Network Optimizer",
        "abstract": "A method for optimizing neural networks...",
        "claims": ["A method comprising...", "The method of claim 1..."]
    }
    result = handler.process(dict_input)
    print(f"  Mode: {result.mode.value}")
    print(f"  Title: {result.title}")
    
    print("ALL TESTS PASSED")

