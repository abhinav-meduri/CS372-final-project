"""
Claim Parser Module
Handles claim canonicalization, parsing, and structure extraction.

This is critical for:
- Identifying independent vs dependent claims
- Extracting claim elements for comparison
- Preserving legal phrasing
- Supporting evidence highlighting in explainability
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class ClaimType(Enum):
    """Type of patent claim."""
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    MULTIPLE_DEPENDENT = "multiple_dependent"


class ClaimScope(Enum):
    """Scope/breadth of claim based on transition phrases."""
    OPEN = "open"           # "comprising" - allows additional elements
    CLOSED = "closed"       # "consisting of" - excludes other elements
    PARTIALLY_CLOSED = "partially_closed"  # "consisting essentially of"
    UNKNOWN = "unknown"


@dataclass
class ParsedClaim:
    """Structured representation of a parsed patent claim."""
    claim_num: int
    claim_type: ClaimType
    claim_scope: ClaimScope
    raw_text: str
    preamble: str
    transition: str
    body: str
    elements: List[str]
    depends_on: List[int]
    key_terms: List[str]
    
    def to_dict(self) -> dict:
        return {
            'claim_num': self.claim_num,
            'claim_type': self.claim_type.value,
            'claim_scope': self.claim_scope.value,
            'raw_text': self.raw_text,
            'preamble': self.preamble,
            'transition': self.transition,
            'body': self.body,
            'elements': self.elements,
            'depends_on': self.depends_on,
            'key_terms': self.key_terms
        }


class ClaimParser:
    """Parse and canonicalize patent claims."""
    
    # Transition phrases that determine claim scope
    OPEN_TRANSITIONS = [
        'comprising', 'including', 'containing', 'having',
        'characterized by', 'which comprises', 'which includes'
    ]
    
    CLOSED_TRANSITIONS = [
        'consisting of', 'composed of', 'which consists of'
    ]
    
    PARTIALLY_CLOSED_TRANSITIONS = [
        'consisting essentially of', 'consisting primarily of'
    ]
    
    # Legal keywords important for claim interpretation
    IMPORTANT_KEYWORDS = [
        'means for', 'step of', 'configured to', 'adapted to',
        'operable to', 'capable of', 'wherein', 'whereby',
        'further comprising', 'additionally', 'optionally'
    ]
    
    # Patterns for detecting claim dependencies
    DEPENDENCY_PATTERNS = [
        r'(?:The|A|An)\s+\w+\s+(?:of|according to)\s+claim\s+(\d+)',
        r'claim\s+(\d+)',
        r'claims?\s+(\d+)\s*[-–]\s*(\d+)',  # Range: claims 1-3
        r'claims?\s+(\d+(?:\s*,\s*\d+)+)',   # List: claims 1, 2, 3
        r'any\s+(?:one\s+)?of\s+claims?\s+(\d+(?:\s*[-–,]\s*\d+)*)'
    ]
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self._dep_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEPENDENCY_PATTERNS]
        self._transition_pattern = self._build_transition_pattern()
    
    def _build_transition_pattern(self) -> re.Pattern:
        """Build regex pattern for transition phrase detection."""
        all_transitions = (
            self.OPEN_TRANSITIONS + 
            self.CLOSED_TRANSITIONS + 
            self.PARTIALLY_CLOSED_TRANSITIONS
        )
        # Sort by length (longest first) to match "consisting of" before "consisting"
        all_transitions.sort(key=len, reverse=True)
        pattern = r'\b(' + '|'.join(re.escape(t) for t in all_transitions) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def parse_claim(
        self, 
        claim_text: str, 
        claim_num: int,
        dependent_flag: Optional[bool] = None
    ) -> ParsedClaim:
        """
        Parse a single claim into structured format.
        
        Args:
            claim_text: Raw claim text
            claim_num: Claim number
            dependent_flag: If known from source data (True=dependent, False=independent)
            
        Returns:
            ParsedClaim object with structured data
        """
        # Clean the claim text
        clean_text = self._clean_claim_text(claim_text)
        
        # Extract dependencies
        depends_on = self._extract_dependencies(clean_text)
        
        # Determine claim type
        if dependent_flag is not None:
            claim_type = ClaimType.DEPENDENT if dependent_flag else ClaimType.INDEPENDENT
        elif depends_on:
            claim_type = ClaimType.MULTIPLE_DEPENDENT if len(depends_on) > 1 else ClaimType.DEPENDENT
        else:
            claim_type = ClaimType.INDEPENDENT
        
        # Extract claim structure
        preamble, transition, body = self._split_claim_structure(clean_text)
        
        # Determine claim scope from transition phrase
        claim_scope = self._determine_scope(transition)
        
        # Extract claim elements (for independent claims)
        elements = self._extract_elements(body) if claim_type == ClaimType.INDEPENDENT else []
        
        # Extract important legal terms
        key_terms = self._extract_key_terms(clean_text)
        
        return ParsedClaim(
            claim_num=claim_num,
            claim_type=claim_type,
            claim_scope=claim_scope,
            raw_text=claim_text,
            preamble=preamble,
            transition=transition,
            body=body,
            elements=elements,
            depends_on=depends_on,
            key_terms=key_terms
        )
    
    def _clean_claim_text(self, text: str) -> str:
        """Clean and normalize claim text."""
        # Remove leading claim number (e.g., "1. ", "12. ")
        text = re.sub(r'^\d+\.\s*', '', text.strip())
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize dashes and special characters
        text = text.replace('—', '-').replace('–', '-')
        
        return text.strip()
    
    def _extract_dependencies(self, text: str) -> List[int]:
        """Extract claim numbers this claim depends on."""
        dependencies = set()
        
        for pattern in self._dep_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle range or multiple matches
                    for m in match:
                        if m:
                            self._parse_claim_refs(m, dependencies)
                else:
                    self._parse_claim_refs(match, dependencies)
        
        return sorted(list(dependencies))
    
    def _parse_claim_refs(self, ref_str: str, deps: Set[int]):
        """Parse claim reference string into individual claim numbers."""
        # Handle comma-separated: "1, 2, 3"
        if ',' in ref_str:
            for part in ref_str.split(','):
                part = part.strip()
                if part.isdigit():
                    deps.add(int(part))
        # Handle range: "1-3"
        elif '-' in ref_str:
            parts = ref_str.split('-')
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                start, end = int(parts[0].strip()), int(parts[1].strip())
                deps.update(range(start, end + 1))
        # Handle single number
        elif ref_str.strip().isdigit():
            deps.add(int(ref_str.strip()))
    
    def _split_claim_structure(self, text: str) -> Tuple[str, str, str]:
        """
        Split claim into preamble, transition phrase, and body.
        
        Example:
            "A computer-implemented method comprising: receiving data; processing data"
            → preamble: "A computer-implemented method"
            → transition: "comprising"
            → body: "receiving data; processing data"
        """
        match = self._transition_pattern.search(text)
        
        if match:
            preamble = text[:match.start()].strip()
            transition = match.group(1).lower()
            body = text[match.end():].strip()
            # Remove leading colon or comma from body
            body = re.sub(r'^[:\s,]+', '', body)
            return preamble, transition, body
        else:
            # No clear transition found - treat entire text as body
            return "", "", text
    
    def _determine_scope(self, transition: str) -> ClaimScope:
        """Determine claim scope from transition phrase."""
        transition_lower = transition.lower()
        
        if any(t in transition_lower for t in ['consisting essentially', 'consisting primarily']):
            return ClaimScope.PARTIALLY_CLOSED
        elif any(t in transition_lower for t in ['consisting of', 'composed of']):
            return ClaimScope.CLOSED
        elif any(t in transition_lower for t in ['comprising', 'including', 'containing', 'having']):
            return ClaimScope.OPEN
        else:
            return ClaimScope.UNKNOWN
    
    def _extract_elements(self, body: str) -> List[str]:
        """
        Extract claim elements from the body.
        
        Handles formats like:
        - "a) first element; b) second element"
        - "(i) first element, (ii) second element"  
        - "first element; second element; third element"
        """
        elements = []
        
        # Try to split by common element markers
        # Pattern: (a), a), (i), i., etc.
        element_pattern = r'(?:^|[;,])\s*(?:\([a-z]\)|\([ivx]+\)|[a-z]\)|[ivx]+\.)\s*'
        
        parts = re.split(element_pattern, body, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) > 1:
            elements = parts
        else:
            # Fallback: split by semicolons
            parts = [p.strip() for p in body.split(';') if p.strip()]
            if len(parts) > 1:
                elements = parts
            else:
                # Single element claim
                elements = [body] if body else []
        
        return elements
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important legal/technical terms from claim."""
        found_terms = []
        text_lower = text.lower()
        
        for keyword in self.IMPORTANT_KEYWORDS:
            if keyword in text_lower:
                found_terms.append(keyword)
        
        return found_terms
    
    def parse_all_claims(
        self, 
        claims: List[dict]
    ) -> List[ParsedClaim]:
        """
        Parse all claims for a patent.
        
        Args:
            claims: List of claim dicts with 'claim_num', 'text', 'dependent'
            
        Returns:
            List of ParsedClaim objects
        """
        parsed = []
        for claim in claims:
            parsed_claim = self.parse_claim(
                claim_text=claim.get('text', ''),
                claim_num=claim.get('claim_num', 0),
                dependent_flag=claim.get('dependent', None)
            )
            parsed.append(parsed_claim)
        
        return parsed
    
    def get_independent_claims(self, claims: List[dict]) -> List[ParsedClaim]:
        """Get only the independent claims from a claim set."""
        all_parsed = self.parse_all_claims(claims)
        return [c for c in all_parsed if c.claim_type == ClaimType.INDEPENDENT]
    
    def compare_claim_elements(
        self, 
        claim1: ParsedClaim, 
        claim2: ParsedClaim
    ) -> Dict[str, any]:
        """
        Compare two claims element-by-element.
        
        Returns:
            Dict with overlap metrics and matched elements
        """
        elements1 = set(e.lower().strip() for e in claim1.elements)
        elements2 = set(e.lower().strip() for e in claim2.elements)
        
        intersection = elements1 & elements2
        union = elements1 | elements2
        
        return {
            'jaccard_similarity': len(intersection) / len(union) if union else 0,
            'claim1_coverage': len(intersection) / len(elements1) if elements1 else 0,
            'claim2_coverage': len(intersection) / len(elements2) if elements2 else 0,
            'matched_elements': list(intersection),
            'claim1_unique': list(elements1 - elements2),
            'claim2_unique': list(elements2 - elements1)
        }


def demo_claim_parsing():
    """Demonstrate claim parsing."""
    parser = ClaimParser()
    
    # Example claims
    claims = [
        {
            'claim_num': 1,
            'text': "1. A computer-implemented method comprising: receiving input data from a sensor; processing the input data using a neural network; and outputting a prediction based on the processed data.",
            'dependent': False
        },
        {
            'claim_num': 2,
            'text': "2. The method of claim 1, wherein the neural network comprises a convolutional layer.",
            'dependent': True
        },
        {
            'claim_num': 3,
            'text': "3. A system consisting of: a processor; a memory; and a display device.",
            'dependent': False
        }
    ]
    
    print("=== Claim Parsing Demo ===\n")
    
    for claim in claims:
        parsed = parser.parse_claim(
            claim['text'], 
            claim['claim_num'],
            claim['dependent']
        )
        
        print(f"Claim {parsed.claim_num}:")
        print(f"  Type: {parsed.claim_type.value}")
        print(f"  Scope: {parsed.claim_scope.value}")
        print(f"  Preamble: {parsed.preamble}")
        print(f"  Transition: {parsed.transition}")
        print(f"  Body: {parsed.body[:100]}...")
        print(f"  Elements: {parsed.elements}")
        print(f"  Depends on: {parsed.depends_on}")
        print(f"  Key terms: {parsed.key_terms}")
        print()


if __name__ == "__main__":
    demo_claim_parsing()


