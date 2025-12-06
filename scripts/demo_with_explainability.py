"""
Full Patent Novelty Assessment Demo with Explainability

This demonstrates the complete pipeline:
1. Input patent (existing or custom)
2. Find similar prior art
3. Compute novelty score
4. Fetch additional details from PatentsView API
5. Generate LLM explanation with citations
"""

import json
import numpy as np
from pathlib import Path
import sys
import pickle
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.explainability.patentsview_api import PatentsViewAPI
from src.explainability.llm_explainer import PatentExplainer, NoveltyExplanation


def load_resources():
    """Load all necessary resources."""
    print("Loading resources...")
    
    # Load patents
    patents = {}
    with open('data/sampled/patents_sampled.jsonl', 'r') as f:
        for line in f:
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    print(f"  ✓ {len(patents)} patents loaded")
    
    # Load embeddings
    embeddings = np.load('data/embeddings/patent_embeddings.npy')
    with open('data/embeddings/patent_ids.json', 'r') as f:
        patent_ids = json.load(f)
    pid_to_idx = {pid: i for i, pid in enumerate(patent_ids)}
    print(f"  ✓ {len(embeddings)} embeddings loaded")
    
    # Load sentence transformer
    st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    print(f"  ✓ PatentSBERTa loaded")
    
    return patents, embeddings, patent_ids, pid_to_idx, st_model


def find_similar_patents(query_embedding, all_embeddings, patent_ids, patents, top_k=10):
    """Find most similar patents."""
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    all_norms = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    similarities = np.dot(all_norms, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        pid = patent_ids[idx]
        patent_data = patents.get(pid, {})
        results.append({
            'patent_id': pid,
            'similarity': float(similarities[idx]),
            'title': patent_data.get('title', 'N/A'),
            'abstract': patent_data.get('abstract', 'N/A'),
            'year': patent_data.get('year', 'N/A'),
            'claims': patent_data.get('claims', [])
        })
    
    return results


def assess_novelty_with_explanation(
    query_patent: dict,
    patents: dict,
    embeddings: np.ndarray,
    patent_ids: list,
    pid_to_idx: dict,
    st_model,
    llm_backend: str = "mock",
    use_patentsview: bool = True,
    **llm_kwargs
):
    """
    Full novelty assessment with explanation.
    
    Args:
        query_patent: Patent to assess (dict with title, abstract, claims)
        patents: All patents database
        embeddings: Patent embeddings
        patent_ids: List of patent IDs
        pid_to_idx: Patent ID to index mapping
        st_model: SentenceTransformer model
        llm_backend: LLM backend to use ("openai", "anthropic", "ollama", "mock")
        use_patentsview: Whether to fetch additional data from PatentsView API
        **llm_kwargs: Additional arguments for LLM backend
    
    Returns:
        Tuple of (novelty_score, similar_patents, explanation)
    """
    
    print("\n" + "="*70)
    print("PATENT NOVELTY ASSESSMENT")
    print("="*70)
    
    # 1. Generate embedding for query patent
    print("\n[1/5] Generating embedding for query patent...")
    query_text = query_patent.get('abstract', '')[:500]
    query_embedding = st_model.encode(query_text)
    
    # 2. Find similar patents
    print("[2/5] Finding similar prior art...")
    query_id = query_patent.get('patent_id', '')
    similar = find_similar_patents(query_embedding, embeddings, patent_ids, patents, top_k=15)
    # Exclude self if present
    similar = [s for s in similar if s['patent_id'] != query_id][:10]
    
    # 3. Compute novelty score
    print("[3/5] Computing novelty score...")
    max_similarity = similar[0]['similarity'] if similar else 0
    novelty_score = 1 - max_similarity
    
    # 4. Fetch additional data from PatentsView API (optional)
    patentsview_data = []
    if use_patentsview:
        print("[4/5] Fetching details from PatentsView API...")
        api = PatentsViewAPI()
        for s in similar[:3]:  # Get details for top 3
            try:
                details = api.get_patent_details(s['patent_id'])
                if details:
                    patentsview_data.append(details)
                    print(f"  ✓ Retrieved details for Patent {s['patent_id']}")
            except Exception as e:
                print(f"  ✗ Could not fetch Patent {s['patent_id']}: {e}")
    else:
        print("[4/5] Skipping PatentsView API (disabled)")
    
    # 5. Generate LLM explanation
    print(f"[5/5] Generating explanation with {llm_backend} backend...")
    try:
        explainer = PatentExplainer(backend=llm_backend, **llm_kwargs)
        explanation = explainer.explain(
            query_patent=query_patent,
            similar_patents=similar,
            novelty_score=novelty_score,
            patentsview_data=patentsview_data if patentsview_data else None
        )
    except Exception as e:
        print(f"  ⚠ LLM error: {e}")
        print("  Falling back to mock backend...")
        explainer = PatentExplainer(backend="mock")
        explanation = explainer.explain(
            query_patent=query_patent,
            similar_patents=similar,
            novelty_score=novelty_score
        )
    
    return novelty_score, similar, explanation


def print_results(novelty_score, similar_patents, explanation):
    """Print formatted results."""
    
    print("\n" + "="*70)
    print("NOVELTY ASSESSMENT RESULTS")
    print("="*70)
    
    # Score visualization
    bar_length = 40
    filled = int(novelty_score * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n📊 NOVELTY SCORE: {novelty_score:.2f} / 1.00")
    print(f"   [{bar}]")
    print(f"   Assessment: {explanation.overall_assessment}")
    
    # Prior art
    print(f"\n📚 TOP PRIOR ART CITATIONS:")
    print("-" * 70)
    for i, patent in enumerate(similar_patents[:5], 1):
        relevance = "🔴 HIGH" if patent['similarity'] > 0.7 else "🟡 MEDIUM" if patent['similarity'] > 0.5 else "🟢 LOW"
        print(f"\n  {i}. Patent {patent['patent_id']}")
        print(f"     Similarity: {patent['similarity']:.3f} ({relevance})")
        print(f"     Title: {patent['title'][:70]}...")
        print(f"     Year: {patent['year']}")
    
    # Key findings
    if explanation.key_findings:
        print(f"\n🔍 KEY FINDINGS:")
        print("-" * 70)
        for finding in explanation.key_findings[:5]:
            print(f"  • {finding}")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    print("-" * 70)
    print(f"  {explanation.recommendation}")
    
    # Full explanation
    print(f"\n📝 DETAILED ANALYSIS:")
    print("-" * 70)
    print(explanation.raw_explanation)


def main():
    # Load resources
    patents, embeddings, patent_ids, pid_to_idx, st_model = load_resources()
    
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        llm_backend = "openai"
        print("\n✓ OpenAI API key found - using GPT-4")
    elif anthropic_key:
        llm_backend = "anthropic"
        print("\n✓ Anthropic API key found - using Claude")
    else:
        llm_backend = "mock"
        print("\n⚠ No API keys found - using mock backend")
        print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY for real explanations")
    
    print("\n" + "="*70)
    print("DEMO: PATENT NOVELTY ASSESSMENT WITH EXPLAINABILITY")
    print("="*70)
    
    # Demo 1: Test with a custom ML patent
    print("\n[DEMO] Testing custom patent about ML for patents...")
    
    custom_patent = {
        "patent_id": "CUSTOM_001",
        "title": "Machine Learning System for Automated Patent Prior Art Search",
        "abstract": """
        A computer-implemented system and method for automatically searching and 
        identifying relevant prior art for patent applications. The system uses 
        transformer-based neural network embeddings to encode patent claims and 
        abstracts into semantic vector representations. A similarity search module 
        compares the query patent against a database of existing patents using 
        cosine similarity. The system generates a novelty score and provides 
        natural language explanations identifying specific claim overlaps with 
        prior art documents. The method improves efficiency of patent examination 
        by automatically surfacing the most relevant prior art references.
        """,
        "claims": [
            {"text": "A computer-implemented method for assessing patent novelty, comprising: receiving a patent application including at least one claim; generating a semantic embedding of the claim using a transformer neural network; comparing the embedding against a database of prior art embeddings; computing a novelty score based on similarity metrics; and generating a report identifying relevant prior art."},
            {"text": "The method of claim 1, wherein the transformer neural network is pre-trained on patent text corpora."},
            {"text": "The method of claim 1, further comprising generating natural language explanations of claim overlaps using a large language model."}
        ]
    }
    
    novelty_score, similar, explanation = assess_novelty_with_explanation(
        query_patent=custom_patent,
        patents=patents,
        embeddings=embeddings,
        patent_ids=patent_ids,
        pid_to_idx=pid_to_idx,
        st_model=st_model,
        llm_backend=llm_backend,
        use_patentsview=True  # Set to False to skip API calls
    )
    
    print_results(novelty_score, similar, explanation)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo test with real LLM explanations, set environment variables:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export ANTHROPIC_API_KEY='your-key'")
    print("\nOr use Ollama for local inference:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Run: ollama pull llama3")
    print("  3. Modify this script to use backend='ollama'")


if __name__ == "__main__":
    main()


