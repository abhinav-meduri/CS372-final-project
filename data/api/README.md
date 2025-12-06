# Data Access API Scripts

This folder contains scripts for accessing external APIs to fetch patent data.

## Files

### `patentsview_api.py`
- **Purpose**: Access PatentsView API (official USPTO patent data)
- **Features**: 
  - Fetch detailed patent information
  - Get citations (forward and backward)
  - Retrieve CPC classifications
  - Get inventor/assignee information

### `online_search.py`
- **Purpose**: Online patent search via Google Patents (SerpAPI)
- **Features**:
  - LLM-based keyword extraction (Phi-3)
  - Google Patents search via SerpAPI
  - Result merging with local FAISS results
  - Hybrid RAG system (local + online)

## Usage

```python
from data.api.patentsview_api import PatentsViewAPI
from data.api.online_search import GooglePatentsSearch, LLMKeywordExtractor

# PatentsView API
api = PatentsViewAPI()
patent_data = api.get_patent("12345678")

# Online search
searcher = GooglePatentsSearch()
results = searcher.search("smart water bottle")
```

## Dependencies

- `requests` - HTTP requests
- `serpapi` - SerpAPI client (optional, for Google Patents search)

