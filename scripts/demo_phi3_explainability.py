"""
Patent Novelty Assessment with Phi-3 Explainability + PatentsView Evidence

Complete pipeline:
1. Find similar patents using embeddings
2. Fetch evidence from PatentsView API
3. Generate explanation using local Phi-3 model
"""

import json
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.explainability.patentsview_api import PatentsViewAPI
from src.explainability.phi3_explainer import get_explainer, NoveltyReport


def load_resources():
    """Load patents and embeddings."""
    print("=" * 60)
    print("LOADING RESOURCES")
    print("=" * 60)
    
    patents = {}
    with open('data/sampled/patents_sampled.jsonl', 'r') as f:
        for line in f:
            p = json.loads(line)
            patents[str(p['patent_id'])] = p
    print(f"✓ {len(patents):,} patents loaded")
    
    embeddings = np.load('data/embeddings/patent_embeddings.npy')
    with open('data/embeddings/patent_ids.json', 'r') as f:
        patent_ids = json.load(f)
    print(f"✓ {len(embeddings):,} embeddings loaded")
    
    st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    print(f"✓ PatentSBERTa loaded")
    
    return patents, embeddings, patent_ids, st_model


def find_similar(query_emb, embeddings, patent_ids, patents, exclude_id=None, top_k=10):
    """Find similar patents by cosine similarity."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    all_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = np.dot(all_norms, query_norm)
    
    top_idx = np.argsort(sims)[::-1]
    
    results = []
    for idx in top_idx:
        pid = patent_ids[idx]
        if pid == exclude_id:
            continue
        p = patents.get(pid, {})
        results.append({
            'patent_id': pid,
            'similarity': float(sims[idx]),
            'title': p.get('title', 'N/A'),
            'abstract': p.get('abstract', 'N/A'),
            'year': p.get('year', 'N/A'),
            'claims': p.get('claims', [])
        })
        if len(results) >= top_k:
            break
    
    return results


def fetch_patentsview_evidence(patent_ids: list, max_patents: int = 3) -> list:
    """Fetch evidence from PatentsView API."""
    print(f"\nFetching USPTO evidence from PatentsView API...")
    
    api = PatentsViewAPI()
    evidence = []
    
    for pid in patent_ids[:max_patents]:
        try:
            details = api.get_patent_details(pid)
            if details:
                evidence.append(details)
                print(f"  ✓ Patent {pid}: {details.get('num_claims', '?')} claims")
        except Exception as e:
            print(f"  ✗ Patent {pid}: {e}")
    
    return evidence


def print_report(report: NoveltyReport):
    """Print formatted novelty report."""
    
    print("\n" + "=" * 60)
    print("NOVELTY ASSESSMENT REPORT")
    print("=" * 60)
    
    # Score bar
    bar_len = 40
    filled = int(report.novelty_score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    # Color coding
    if report.assessment == "NOVEL":
        emoji = "🟢"
    elif report.assessment == "MODERATELY_NOVEL":
        emoji = "🟡"
    elif report.assessment == "LOW_NOVELTY":
        emoji = "🟠"
    else:
        emoji = "🔴"
    
    print(f"\n{emoji} VERDICT: {report.assessment}")
    print(f"\n📊 NOVELTY SCORE: {report.novelty_score:.2f}/1.00")
    print(f"   [{bar}]")
    
    print(f"\n📝 SUMMARY:")
    print(f"   {report.summary}")
    
    print(f"\n📚 PRIOR ART CITATIONS:")
    for i, cite in enumerate(report.prior_art_citations[:5], 1):
        relevance_icon = "🔴" if cite['relevance'] == "High" else "🟡"
        print(f"   {i}. {relevance_icon} Patent {cite['patent_id']} - {cite['similarity']:.1%} similar")
        print(f"      {cite['title'][:60]}...")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {report.recommendation}")
    
    print(f"\n{'=' * 60}")
    print("DETAILED ANALYSIS")
    print("=" * 60)
    print(report.full_explanation)


def assess_patent(
    query_patent: dict,
    patents: dict,
    embeddings: np.ndarray,
    patent_ids: list,
    st_model,
    use_full_model: bool = False,
    use_patentsview: bool = True
) -> NoveltyReport:
    """
    Full novelty assessment pipeline.
    """
    print("\n" + "=" * 60)
    print("PATENT NOVELTY ASSESSMENT")
    print("=" * 60)
    
    print(f"\n📄 Query Patent: {query_patent.get('title', 'Custom')[:50]}...")
    
    # 1. Generate embedding
    print("\n[1/4] Generating semantic embedding...")
    query_text = query_patent.get('abstract', '')[:500]
    query_emb = st_model.encode(query_text)
    
    # 2. Find similar patents
    print("[2/4] Finding similar prior art...")
    query_id = query_patent.get('patent_id', '')
    similar = find_similar(query_emb, embeddings, patent_ids, patents, exclude_id=query_id)
    
    # 3. Compute novelty score
    print("[3/4] Computing novelty score...")
    max_sim = similar[0]['similarity'] if similar else 0
    novelty_score = 1 - max_sim
    print(f"       Score: {novelty_score:.2f} (max similarity: {max_sim:.2f})")
    
    # 4. Fetch PatentsView evidence
    patentsview_data = None
    if use_patentsview:
        print("[4/4] Fetching USPTO evidence...")
        try:
            pids = [s['patent_id'] for s in similar[:3]]
            patentsview_data = fetch_patentsview_evidence(pids)
        except Exception as e:
            print(f"       ⚠ PatentsView error: {e}")
            patentsview_data = None
    else:
        print("[4/4] Skipping PatentsView (disabled)")
    
    # 5. Generate explanation
    print("\n" + "-" * 40)
    print("GENERATING EXPLANATION")
    print("-" * 40)
    
    explainer = get_explainer(use_full_model=use_full_model)
    
    if use_full_model:
        print("Loading Phi-3 model (this may take a few minutes)...")
    else:
        print("Using template-based explanation (fast mode)")
        print("Tip: Use --full-model flag for AI-generated explanations")
    
    report = explainer.generate_explanation(
        query_patent=query_patent,
        similar_patents=similar,
        novelty_score=novelty_score,
        patentsview_evidence=patentsview_data
    )
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Patent Novelty Assessment with Phi-3")
    parser.add_argument("--full-model", action="store_true", 
                       help="Use full Phi-3 model (requires GPU, slower)")
    parser.add_argument("--no-patentsview", action="store_true",
                       help="Skip PatentsView API calls")
    parser.add_argument("--patent-id", type=str, default=None,
                       help="Assess existing patent by ID")
    args = parser.parse_args()
    
    # Load resources
    patents, embeddings, patent_ids, st_model = load_resources()
    
    print("\n" + "=" * 60)
    print("PHI-3 PATENT NOVELTY EXPLAINER")
    print("=" * 60)
    print(f"Mode: {'Full Phi-3 Model' if args.full_model else 'Template-based (fast)'}")
    print(f"PatentsView API: {'Enabled' if not args.no_patentsview else 'Disabled'}")
    
    if args.patent_id:
        # Assess existing patent
        query = patents.get(args.patent_id)
        if not query:
            print(f"Patent {args.patent_id} not found!")
            return
    else:
        # Demo with custom patent
        print("\n[DEMO] Assessing custom ML patent application...")
        
        query = {
            "patent_id": "CUSTOM_DEMO",
            "title": "Automated Patent Prior Art Search Using Transformer Neural Networks",
            "abstract": """
            A computer-implemented system and method for automatically identifying relevant 
            prior art for patent applications using deep learning. The system encodes patent 
            claims and abstracts into dense vector embeddings using a transformer-based neural 
            network trained on patent corpora. A similarity search module compares query patents 
            against a database of millions of existing patents using approximate nearest neighbor 
            algorithms. The system outputs a ranked list of relevant prior art documents along 
            with novelty scores and natural language explanations identifying specific claim 
            overlaps. This enables patent examiners and applicants to efficiently assess 
            patentability and identify potential blocking patents.
            """,
            "claims": [
                {"text": "A computer-implemented method for patent novelty assessment comprising: receiving a query patent document; encoding the document into a semantic embedding using a transformer neural network; comparing the embedding against a prior art database; and generating a novelty score and explanation."},
                {"text": "The method of claim 1, wherein the transformer neural network is pre-trained on a corpus of patent documents."},
                {"text": "The method of claim 1, further comprising generating natural language explanations using a large language model."},
            ],
            "year": 2025
        }
    
    # Run assessment
    report = assess_patent(
        query_patent=query,
        patents=patents,
        embeddings=embeddings,
        patent_ids=patent_ids,
        st_model=st_model,
        use_full_model=args.full_model,
        use_patentsview=not args.no_patentsview
    )
    
    # Print results
    print_report(report)
    
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    
    if not args.full_model:
        print("\n💡 For AI-generated explanations, run:")
        print("   python scripts/demo_phi3_explainability.py --full-model")
        print("\n   Requirements:")
        print("   - GPU with 8GB+ VRAM, OR")
        print("   - Mac with 16GB+ RAM (Apple Silicon)")


if __name__ == "__main__":
    main()


