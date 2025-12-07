"""
Data Access API Scripts

This module contains scripts for accessing external APIs to fetch patent data:
- Google Patents Search: Via SerpAPI for comprehensive online search
"""

from .online_search import LLMKeywordExtractor, GooglePatentsSearch, HybridPatentSearch

__all__ = [
    'LLMKeywordExtractor',
    'GooglePatentsSearch',
    'HybridPatentSearch'
]

