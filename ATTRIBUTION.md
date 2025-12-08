# Attribution

This document details all external resources, datasets, libraries, and AI assistance used in developing the Patent Novelty Assessment System.

## Datasets

### PatentsView
- **Source:** [PatentsView](https://patentsview.org/)
- **License:** Public Domain
- **Usage:** Patent claims and brief summary text data from 2021-2025
- **Files Used:**
  - `g_claims_{year}.tsv` - Patent claims data
  - `g_brf_sum_text_{year}.tsv` - Brief summary text data

## Pre-trained Models

### PatentSBERTa
- **Source:** [AI-Growth-Lab/PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
- **Paper:** [Patent-SBERTA: A Sentence BERT Model for Patent Embeddings](https://arxiv.org/abs/2103.11933)
- **License:** Apache 2.0
- **Usage:** Generating semantic embeddings for patent text

## Libraries and Frameworks

### Machine Learning
| Library | Version | License | Usage |
|---------|---------|---------|-------|
| PyTorch | 2.0+ | BSD-3 | Deep learning backend |
| scikit-learn | 1.3+ | BSD-3 | MLP classifier, preprocessing |
| sentence-transformers | 2.2+ | Apache 2.0 | Loading PatentSBERTa |
| FAISS | 1.7+ | MIT | Vector similarity search |

### Information Retrieval
| Library | Version | License | Usage |
|---------|---------|---------|-------|
| Pyserini | 0.21+ | Apache 2.0 | BM25 indexing and search |
| rank-bm25 | 0.2+ | Apache 2.0 | BM25 scoring |

### NLP
| Library | Version | License | Usage |
|---------|---------|---------|-------|
| spaCy | 3.6+ | MIT | Text preprocessing |
| NLTK | 3.8+ | Apache 2.0 | Tokenization, stopwords |

### LLM Integration
| Service/Model | Usage | License |
|--------------|-------|---------|
| Ollama | Local LLM runtime for Phi-3 | MIT |
| Phi-3 (Microsoft) | Patent explanation generation | MIT |
| OpenAI GPT-4 | Alternative explanation generation (not used in production) | Commercial |
| Anthropic Claude | Alternative explanation generation (not used in production) | Commercial |

### Web Development
| Library | Version | License | Usage |
|---------|---------|---------|-------|
| Streamlit | 1.28+ | Apache 2.0 | Web interface |
| Plotly | 5.17+ | MIT | Interactive visualizations |

### Data Processing
| Library | Version | License | Usage |
|---------|---------|---------|-------|
| pandas | 2.0+ | BSD-3 | Data manipulation |
| NumPy | 1.24+ | BSD-3 | Numerical operations |

## Code References

### BM25 Implementation
- Based on Pyserini documentation and examples
- Reference: [Pyserini GitHub](https://github.com/castorini/pyserini)

### FAISS Indexing
- Based on FAISS official documentation
- Reference: [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)

### MLP Architecture
- Standard scikit-learn MLPClassifier implementation
- Hyperparameters informed by cross-validation

## AI Assistance

### Claude (Anthropic)
- **Usage:** 
  - Project planning and architecture design
  - Code generation assistance
  - Documentation drafting
  - Debugging support
- **Extent:** AI-assisted development with human review and modification

### Online Search APIs
| Service | Usage | License |
|---------|-------|---------|
| SerpAPI | Google Patents search integration | Commercial (free tier available) |
| Google Patents | Patent database search | Public |

### Specific AI-Generated Components
1. Project structure and file organization
2. Initial boilerplate code for data processing pipeline
3. Documentation templates (README, SETUP, this file)
4. Prompt engineering for LLM explainability component

**Note:** All AI-generated code was reviewed, tested, and modified as needed to ensure correctness and alignment with project requirements.

## Academic References

1. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.

2. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.

3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

4. Srebrovic, R., & Yonamine, J. (2021). Patent-SBERTA: A Sentence BERT Model for Patent Embeddings. *NAACL*.

## Acknowledgments

- CS 372 course staff for project guidance and feedback
- PatentsView team for making patent data publicly accessible
- Hugging Face community for model hosting infrastructure
- Open source maintainers of all libraries used

---

*Last Updated: November 2025*

