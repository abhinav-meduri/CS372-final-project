"""
Demo: Patent Novelty Inference

Run this to see how the system assesses novelty for a patent.
"""

import json
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer


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
    
    # Load model
    with open('models/mlp_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ MLP model loaded")
    
    # Load sentence transformer for new patents
    st_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    print(f"  ✓ PatentSBERTa loaded")
    
    return patents, embeddings, patent_ids, pid_to_idx, model, scaler, st_model


def find_similar_patents(query_embedding, all_embeddings, patent_ids, top_k=10):
    """Find most similar patents using cosine similarity."""
    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    all_norms = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(all_norms, query_norm)
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'patent_id': patent_ids[idx],
            'similarity': float(similarities[idx])
        })
    
    return results


def assess_novelty(query_patent, similar_patents, patents):
    """Generate a novelty assessment."""
    
    # Simple novelty score based on similarity to closest prior art
    max_similarity = similar_patents[0]['similarity']
    
    # Novelty score: inverse of max similarity
    # High similarity → Low novelty, Low similarity → High novelty
    novelty_score = 1 - max_similarity
    
    # Interpretation
    if novelty_score > 0.7:
        assessment = "HIGHLY NOVEL - No closely related prior art found"
    elif novelty_score > 0.5:
        assessment = "MODERATELY NOVEL - Some related patents exist but significant differences"
    elif novelty_score > 0.3:
        assessment = "LOW NOVELTY - Similar patents exist in the database"
    else:
        assessment = "NOT NOVEL - Very similar prior art exists"
    
    return novelty_score, assessment


def demo_with_existing_patent(patent_id: str, patents, embeddings, patent_ids, pid_to_idx, model, scaler, st_model):
    """Demo using an existing patent as query."""
    
    print(f"\n{'='*60}")
    print(f"NOVELTY ASSESSMENT FOR PATENT: {patent_id}")
    print(f"{'='*60}")
    
    query_patent = patents.get(patent_id)
    if not query_patent:
        print(f"Patent {patent_id} not found!")
        return
    
    print(f"\nTitle: {query_patent.get('title', 'N/A')[:100]}...")
    print(f"Year: {query_patent.get('year', 'N/A')}")
    print(f"Abstract: {query_patent.get('abstract', 'N/A')[:200]}...")
    
    # Get embedding
    if patent_id in pid_to_idx:
        query_embedding = embeddings[pid_to_idx[patent_id]]
    else:
        # Generate new embedding
        text = query_patent.get('abstract', '')[:500]
        query_embedding = st_model.encode(text)
    
    # Find similar patents (excluding self)
    similar = find_similar_patents(query_embedding, embeddings, patent_ids, top_k=11)
    similar = [s for s in similar if s['patent_id'] != patent_id][:10]
    
    # Assess novelty
    novelty_score, assessment = assess_novelty(query_patent, similar, patents)
    
    print(f"\n{'='*60}")
    print(f"NOVELTY SCORE: {novelty_score:.2f} / 1.00")
    print(f"ASSESSMENT: {assessment}")
    print(f"{'='*60}")
    
    print(f"\nTOP 5 MOST SIMILAR PRIOR ART:")
    print("-" * 60)
    for i, s in enumerate(similar[:5], 1):
        prior_patent = patents.get(s['patent_id'], {})
        print(f"\n{i}. Patent {s['patent_id']} (Similarity: {s['similarity']:.3f})")
        print(f"   Title: {prior_patent.get('title', 'N/A')[:80]}...")
        print(f"   Year: {prior_patent.get('year', 'N/A')}")
    
    return novelty_score, similar


def demo_with_custom_text(title: str, abstract: str, patents, embeddings, patent_ids, st_model):
    """Demo with custom patent text."""
    
    print(f"\n{'='*60}")
    print(f"NOVELTY ASSESSMENT FOR CUSTOM PATENT")
    print(f"{'='*60}")
    
    print(f"\nTitle: {title}")
    print(f"Abstract: {abstract[:200]}...")
    
    # Generate embedding
    text = abstract[:500]
    query_embedding = st_model.encode(text)
    
    # Find similar patents
    similar = find_similar_patents(query_embedding, embeddings, patent_ids, top_k=10)
    
    # Assess novelty
    novelty_score, assessment = assess_novelty({'title': title, 'abstract': abstract}, similar, patents)
    
    print(f"\n{'='*60}")
    print(f"NOVELTY SCORE: {novelty_score:.2f} / 1.00")
    print(f"ASSESSMENT: {assessment}")
    print(f"{'='*60}")
    
    print(f"\nTOP 5 MOST SIMILAR PRIOR ART:")
    print("-" * 60)
    for i, s in enumerate(similar[:5], 1):
        prior_patent = patents.get(s['patent_id'], {})
        print(f"\n{i}. Patent {s['patent_id']} (Similarity: {s['similarity']:.3f})")
        print(f"   Title: {prior_patent.get('title', 'N/A')[:80]}...")
        print(f"   Year: {prior_patent.get('year', 'N/A')}")
    
    return novelty_score, similar


def main():
    # Load resources
    patents, embeddings, patent_ids, pid_to_idx, model, scaler, st_model = load_resources()
    
    print("\n" + "="*60)
    print("PATENT NOVELTY ASSESSMENT DEMO")
    print("="*60)
    
    # Demo 1: Use a random existing patent
    print("\n[DEMO 1] Testing with an existing patent from 2023...")
    sample_patents_2023 = [pid for pid, p in patents.items() if p.get('year') == 2023][:5]
    if sample_patents_2023:
        demo_with_existing_patent(sample_patents_2023[0], patents, embeddings, patent_ids, pid_to_idx, model, scaler, st_model)
    
    # Demo 2: Test with custom text (a hypothetical AI patent)
    print("\n" + "="*60)
    print("[DEMO 2] Testing with CUSTOM patent text...")
    
    custom_title = "Machine Learning System for Real-Time Patent Novelty Assessment"
    custom_abstract = """
    A computer-implemented method and system for assessing the novelty of patent applications 
    using machine learning techniques. The system receives a patent application including claims 
    and abstract text, generates semantic embeddings using a transformer-based neural network, 
    compares the embeddings against a database of prior art, and produces a novelty score 
    indicating the likelihood that the patent application describes a novel invention. 
    The system further generates natural language explanations identifying specific prior art 
    documents and claim elements that may affect patentability.
    """
    
    demo_with_custom_text(custom_title, custom_abstract, patents, embeddings, patent_ids, st_model)
    
    # Demo 3: Test with something clearly different (a cooking recipe as patent)
    print("\n" + "="*60)
    print("[DEMO 3] Testing with UNRELATED text (should be 'novel')...")
    
    weird_title = "Chocolate Cake Baking Method"
    weird_abstract = """
    A method for baking a chocolate cake comprising the steps of: mixing flour, sugar, 
    cocoa powder, and eggs in a bowl; adding milk and vanilla extract; pouring the batter 
    into a greased pan; and baking at 350 degrees Fahrenheit for 30 minutes. The resulting 
    cake has a moist texture and rich chocolate flavor.
    """
    
    demo_with_custom_text(weird_title, weird_abstract, patents, embeddings, patent_ids, st_model)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo test your own patent, modify this script or wait for the Streamlit UI!")


if __name__ == "__main__":
    main()


