# Patent Novelty Assessment System - Complete Technical Walkthrough

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Training](#model-training)
4. [Inference Pipeline](#inference-pipeline)
5. [Web Application](#web-application)
6. [Key Files Deep Dive](#key-files-deep-dive)

---

# System Architecture Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT (Patent Text)                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  INPUT HANDLER (input_handler.py)                │
│  - Parse text/JSON/CSV                                           │
│  - Validate format                                               │
│  - Extract title, abstract, claims                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              PATENT ANALYZER (patent_analyzer.py)                │
│                      Orchestrates Pipeline                       │
└─────────────────┬───────────────────────────┬───────────────────┘
                  ↓                           ↓
     ┌────────────────────────┐   ┌──────────────────────────┐
     │   LOCAL SEARCH         │   │   ONLINE SEARCH          │
     │   PatentSBERTa         │   │   LLM Keywords           │
     │   200K Patents         │   │   SerpAPI                │
     │   Cosine Similarity    │   │   Google Patents         │
     └────────┬───────────────┘   └────────┬─────────────────┘
              │                             │
              └──────────┬──────────────────┘
                         ↓
        ┌────────────────────────────────┐
        │  FEATURE EXTRACTION            │
        │  10 features per patent pair   │
        │  (feature_extract.py)          │
        └────────────┬───────────────────┘
                     ↓
        ┌────────────────────────────────┐
        │  ML SCORING                    │
        │  PyTorch Neural Network        │
        │  (pytorch_classifier.py)       │
        └────────────┬───────────────────┘
                     ↓
        ┌────────────────────────────────┐
        │  LLM EXPLANATION               │
        │  Phi-3 via Ollama              │
        │  (phi3_explainer.py)           │
        └────────────┬───────────────────┘
                     ↓
        ┌────────────────────────────────┐
        │  RESULTS                       │
        │  - Novelty Score               │
        │  - Similar Patents             │
        │  - AI Explanation              │
        └────────────────────────────────┘
```

## Technology Stack

**Embeddings & NLP:**
- `sentence-transformers`: PatentSBERTa model
- `transformers`: Underlying BERT architecture
- `sklearn.feature_extraction.text`: TF-IDF vectorization

**Machine Learning:**
- `torch`: PyTorch neural network
- `numpy`: Array operations
- `sklearn`: Preprocessing, metrics

**Vector Search:**
- Pure NumPy cosine similarity (no FAISS in final version)
- Memory-mapped embeddings for efficiency

**LLM:**
- Ollama: Local LLM server
- Phi-3: 3.8B parameter model

**Web Framework:**
- `streamlit`: Web UI
- `requests`: API calls

**Data Storage:**
- JSONL: Patent database
- NPY: NumPy arrays for embeddings
- Pickle: Model checkpoints

---

# Data Pipeline

## 1. Data Collection (`scripts/data/collection/`)

### USPTO Patent Download

**File: `download_uspto_patents.py`**

Downloads patents from USPTO bulk data servers for years 2021-2025.

```python
def download_patents(year: int, output_dir: str):
    """
    Downloads USPTO patent XML files for a given year.
    
    Process:
    1. Construct USPTO bulk data URL
    2. Download compressed archive
    3. Extract XML files
    4. Parse patent data
    5. Save as JSONL
    
    Output: Raw patent XMLs organized by year
    """
```

**Key Method:**
```python
def parse_patent_xml(xml_content: str) -> dict:
    """
    Extracts structured data from USPTO XML.
    
    Returns:
    {
        'patent_id': 'US12345678',
        'title': str,
        'abstract': str,
        'claims': List[str],
        'publication_date': str,
        'assignee': str,
        'inventors': List[str],
        'cpc_codes': List[str]
    }
    """
```

## 2. Data Sampling (`scripts/data/preprocessing/sample_patents.py`)

**Purpose:** Create a manageable 200K patent subset from millions of patents.

**Sampling Strategy:**
```python
def stratified_sampling(
    years=[2021, 2022, 2023, 2024, 2025],
    samples_per_year=40000,
    random_seed=42
):
    """
    Ensures balanced representation across years.
    
    Algorithm:
    1. Group patents by year
    2. Random sample 40K from each year
    3. Combine into 200K total
    
    Why stratified:
    - Avoids temporal bias
    - Ensures recent patents included
    - Maintains distribution of patent types
    """
```

**Output:**
- `data/sampled/patents_sampled.jsonl` (200,000 lines, ~3.8 GB)
- `data/sampled/sampling_metadata.json` (statistics)

## 3. Embedding Generation (`scripts/data/preprocessing/generate_embeddings.py`)

**Purpose:** Convert 200K patents into 768-dimensional semantic embeddings using PatentSBERTa.

### Core Algorithm

```python
class EmbeddingGenerator:
    def __init__(self):
        # Load PatentSBERTa - fine-tuned on 1.2M patents
        self.model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
        self.model.to('mps')  # Use Apple GPU if available
        
    def get_patent_text(self, patent: dict) -> str:
        """
        Extracts text for embedding.
        
        Priority order:
        1. Abstract (best balance of info/length)
        2. Summary
        3. First claim
        
        Truncate to 500 chars to fit model context window.
        """
        if patent.get("abstract"):
            return patent["abstract"][:500]
        if patent.get("summary"):
            return patent["summary"][:500]
        claims = patent.get("claims", [])
        if claims:
            first_claim = claims[0]
            if isinstance(first_claim, dict):
                return (first_claim.get("text") or "")[:500]
            if isinstance(first_claim, str):
                return first_claim[:500]
        return ""
    
    def generate_embeddings(self, patents: List[dict], batch_size=32):
        """
        Batch embedding generation for efficiency.
        
        Process:
        1. Extract text from each patent
        2. Batch encode (32 patents at a time)
        3. Save every 10K patents (checkpointing)
        4. Return 200K x 768 embedding matrix
        
        Time: ~11 hours for 200K patents on Apple M1
        """
        texts = [self.get_patent_text(p) for p in patents]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # Will normalize during search
        )
        return embeddings  # Shape: (200000, 768)
```

**Mathematical Details:**

PatentSBERTa uses BERT architecture:
```
Input: Patent text (max 512 tokens)
  ↓
BERT Encoder: 12 transformer layers
  ↓
[CLS] Token Pooling: Extract sentence representation
  ↓
Output: 768-dimensional dense vector

Embedding properties:
- Semantic similarity preserved
- Domain-specific (trained on patents)
- Normalized ||v|| for cosine similarity
```

**Outputs:**
- `data/embeddings/patent_embeddings.npy`: Shape (200000, 768), 586 MB
- `data/embeddings/patent_ids.json`: List of 200K patent IDs
- `data/embeddings/embedding_metadata.json`: Generation stats

## 4. Citation Pair Extraction (`scripts/training/extract_citation_pairs.py`)

**Purpose:** Create training data for the similarity model using actual patent citations.

### Citation Graph Analysis

**Key Insight:** If patent A cites patent B, they are definitely related. This gives us labeled positive examples.

```python
class CitationPairExtractor:
    def __init__(self, patent_database, citation_graph):
        self.patents = patent_database
        self.citations = citation_graph  # A → [B, C, D] mapping
        
    def extract_positive_pairs(self) -> List[Tuple[str, str]]:
        """
        Extract citation pairs as positive examples.
        
        Returns:
        [
            ('US11234567', 'US10123456'),  # 11234567 cites 10123456
            ('US11234568', 'US10123457'),
            ...
        ]
        
        Only keep pairs where BOTH patents are in our 200K database.
        
        Result: 28,557 positive pairs
        """
        positive_pairs = []
        for citing_patent, cited_patents in self.citations.items():
            if citing_patent not in self.our_patent_ids:
                continue
            for cited_patent in cited_patents:
                if cited_patent in self.our_patent_ids:
                    positive_pairs.append((citing_patent, cited_patent))
        return positive_pairs
    
    def generate_negative_pairs(
        self, 
        positive_pairs: List[Tuple],
        ratio: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Create negative pairs by random sampling.
        
        Strategy:
        - For each positive pair, create `ratio` negative pairs
        - Negative = random patent that is NOT cited
        - Ensures class balance
        
        Algorithm:
        1. For positive pair (A, B)
        2. Sample random patent C where C != B and A does not cite C
        3. Create negative pair (A, C)
        
        Result: 28,557 negative pairs (1:1 ratio)
        
        Note: This is "positive-unlabeled" learning
        - Positives are confirmed (citations)
        - Negatives are ASSUMED (no citation doesn't mean unrelated)
        """
        negative_pairs = []
        all_patents = list(self.our_patent_ids)
        
        for citing, cited in positive_pairs:
            # Get all patents cited by `citing`
            cited_by_citing = set(self.citations.get(citing, []))
            
            # Sample random patent not in that set
            attempts = 0
            while attempts < 100:  # Avoid infinite loop
                candidate = random.choice(all_patents)
                if candidate != citing and candidate not in cited_by_citing:
                    negative_pairs.append((citing, candidate))
                    break
                attempts += 1
        
        return negative_pairs
    
    def create_train_val_test_split(
        self, 
        positive_pairs, 
        negative_pairs,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    ):
        """
        Stratified split maintaining class balance.
        
        Total: 57,114 pairs (28,557 positive + 28,557 negative)
        
        Split:
        - Train: 39,979 pairs (70%)
        - Val: 8,567 pairs (15%)
        - Test: 8,568 pairs (15%)
        
        Each split has 50% positive, 50% negative.
        """
```

**Outputs:**
- `data/training/train_pairs.json`: 39,979 pairs
- `data/training/val_pairs.json`: 8,567 pairs
- `data/training/test_pairs.json`: 8,568 pairs
- `data/training/dataset_stats.json`: Class distribution

## 5. Feature Computation (`scripts/data/preprocessing/compute_features.py`)

**Purpose:** Extract 10 engineered features for each patent pair.

### Feature Engineering Pipeline

```python
class FeatureComputer:
    def __init__(self, embeddings, patents, embedder):
        self.embeddings = embeddings  # 200K x 768
        self.patents = patents
        self.embedder = embedder  # PatentSBERTa model
        
    def compute_features_for_pair(
        self, 
        patent_a_id: str, 
        patent_b_id: str
    ) -> np.ndarray:
        """
        Computes 10 features for a patent pair.
        
        Returns: Array of shape (10,)
        """
        features = []
        
        patent_a = self.patents[patent_a_id]
        patent_b = self.patents[patent_b_id]
        
        # Feature 1: PatentSBERTa Cosine Similarity
        emb_a = self.embeddings[self.patent_id_to_index[patent_a_id]]
        emb_b = self.embeddings[self.patent_id_to_index[patent_b_id]]
        cosine_sim = self._cosine_similarity(emb_a, emb_b)
        features.append(cosine_sim)
        
        # Feature 2: TF-IDF Cosine Similarity
        text_a = patent_a.get('abstract', '')
        text_b = patent_b.get('abstract', '')
        tfidf_sim = self._tfidf_similarity(text_a, text_b)
        features.append(tfidf_sim)
        
        # Feature 3: Jaccard Similarity (word overlap)
        jaccard_sim = self._jaccard_similarity(text_a, text_b)
        features.append(jaccard_sim)
        
        # Feature 4: Claim Count Ratio
        claims_a = len(patent_a.get('claims', []))
        claims_b = len(patent_b.get('claims', []))
        claim_ratio = min(claims_a, claims_b) / (max(claims_a, claims_b) + 1e-8)
        features.append(claim_ratio)
        
        # Feature 5: Abstract Length Ratio
        len_a = len(text_a.split())
        len_b = len(text_b.split())
        length_ratio = min(len_a, len_b) / (max(len_a, len_b) + 1e-8)
        features.append(length_ratio)
        
        # Feature 6: Year Difference (normalized)
        year_a = patent_a.get('year', 2023)
        year_b = patent_b.get('year', 2023)
        year_diff = 1.0 / (1.0 + abs(year_a - year_b))
        features.append(year_diff)
        
        # Feature 7: Assignee Match (binary)
        assignee_match = float(
            patent_a.get('assignee', '') == patent_b.get('assignee', '')
            and patent_a.get('assignee', '') != ''
        )
        features.append(assignee_match)
        
        # Feature 8: CPC Code Overlap (Jaccard)
        cpc_a = set(patent_a.get('cpc_codes', []))
        cpc_b = set(patent_b.get('cpc_codes', []))
        cpc_jaccard = len(cpc_a & cpc_b) / (len(cpc_a | cpc_b) + 1e-8)
        features.append(cpc_jaccard)
        
        # Feature 9: Max Claim Embedding Similarity
        max_claim_sim = self._max_claim_similarity(patent_a, patent_b)
        features.append(max_claim_sim)
        
        # Feature 10: Title Similarity
        title_a = patent_a.get('title', '')
        title_b = patent_b.get('title', '')
        title_sim = self._sentence_similarity(title_a, title_b)
        features.append(title_sim)
        
        return np.array(features, dtype=np.float32)
    
    def _cosine_similarity(self, a, b):
        """Cosine similarity: dot(a, b) / (||a|| * ||b||)"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def _tfidf_similarity(self, text_a, text_b):
        """Traditional TF-IDF cosine similarity."""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            vectors = vectorizer.fit_transform([text_a, text_b])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            return 0.0
    
    def _jaccard_similarity(self, text_a, text_b):
        """Word-level Jaccard: |A ∩ B| / |A ∪ B|"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / (len(union) + 1e-8)
    
    def _max_claim_similarity(self, patent_a, patent_b):
        """
        Computes maximum similarity across all claim pairs.
        
        Algorithm:
        1. Embed all claims from patent A
        2. Embed all claims from patent B
        3. Compute pairwise similarities
        4. Return max similarity
        
        Captures: "Even if most claims differ, is there ONE very similar claim?"
        """
        claims_a = patent_a.get('claims', [])
        claims_b = patent_b.get('claims', [])
        
        if not claims_a or not claims_b:
            return 0.0
        
        # Get claim text
        texts_a = [c.get('text', c) if isinstance(c, dict) else c for c in claims_a]
        texts_b = [c.get('text', c) if isinstance(c, dict) else c for c in claims_b]
        
        # Embed claims
        embs_a = self.embedder.encode(texts_a[:10])  # Limit to first 10
        embs_b = self.embedder.encode(texts_b[:10])
        
        # Pairwise similarities
        max_sim = 0.0
        for emb_a in embs_a:
            for emb_b in embs_b:
                sim = self._cosine_similarity(emb_a, emb_b)
                max_sim = max(max_sim, sim)
        
        return max_sim
    
    def batch_compute_features(self, pairs: List[Tuple], batch_size=1000):
        """
        Efficiently compute features for all pairs.
        
        Process:
        1. Group pairs into batches
        2. Compute features in parallel where possible
        3. Save checkpoints every 10K pairs
        
        Output: Shape (N, 10) where N = number of pairs
        """
```

**Outputs:**
- `data/features/train_features_v2.X.npy`: (39,979, 10)
- `data/features/train_features_v2.y.npy`: (39,979,) labels
- `data/features/val_features_v2.X.npy`: (8,567, 10)
- `data/features/val_features_v2.y.npy`: (8,567,)
- `data/features/test_features_v2.X.npy`: (8,568, 10)
- `data/features/test_features_v2.y.npy`: (8,568,)
- `data/features/feature_names_v2.json`: List of feature names

---

# Model Training

## PyTorch Neural Network (`src/app/pytorch_classifier.py`)

### Architecture Design

```python
class ResidualBlock(nn.Module):
    """
    Residual block with skip connections.
    
    Architecture:
        Input
          ↓
        Linear → BatchNorm → ReLU → Dropout
          ↓
        Linear → BatchNorm → ReLU → Dropout
          ↓
        + Skip Connection (identity or projection)
          ↓
        Output
    
    Why residual blocks:
    - Prevents vanishing gradients
    - Allows deeper networks
    - Improves training stability
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        dropout_rate: float,
        bn_momentum: float = 0.1
    ):
        super().__init__()
        
        # First transformation
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second transformation
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Skip connection
        if in_features != out_features:
            # Project input to match output dimensions
            self.skip_connection = nn.Linear(in_features, out_features)
            self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        else:
            # Identity mapping
            self.skip_connection = nn.Identity()
            self.skip_bn = nn.Identity()
    
    def forward(self, x):
        # Main path
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        # Skip path
        skip = self.skip_connection(x)
        skip = self.skip_bn(skip)
        
        # Combine
        return out + skip  # Element-wise addition


class PatentNoveltyNet(nn.Module):
    """
    Custom neural network for patent similarity prediction.
    
    Input: 10 features
    Hidden: [256, 128] with residual connections
    Output: 1 (similarity score via sigmoid)
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        bn_momentum: float = 0.1
    ):
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0], momentum=bn_momentum)
        self.input_dropout = nn.Dropout(dropout)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.residual_blocks.append(
                ResidualBlock(in_dim, out_dim, dropout, bn_momentum)
            )
            in_dim = out_dim
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        # Input transformation
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        x = self.output_layer(x)
        x = torch.sigmoid(x)  # Squash to [0, 1]
        
        return x
```

### Training Process

```python
class PyTorchPatentClassifier:
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        device: str = None
    ):
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Apple Silicon
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Model
        self.model = PatentNoveltyNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the model.
        
        Process:
        1. Normalize features using StandardScaler
        2. Create DataLoaders for batching
        3. Initialize optimizer and scheduler
        4. Training loop with early stopping
        5. Save best model based on validation loss
        """
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Create DataLoaders
        train_loader, val_loader = self._create_dataloaders(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,  # Reduce LR by 50%
            patience=5   # After 5 epochs without improvement
        )
        
        # Loss function
        criterion = nn.BCELoss()  # Binary Cross-Entropy
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                        
                        val_losses.append(loss.item())
                        all_preds.extend(predictions.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Compute metrics
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                binary_preds = (all_preds > 0.5).astype(int)
                
                metrics = {
                    'accuracy': accuracy_score(all_labels, binary_preds),
                    'f1': f1_score(all_labels, binary_preds),
                    'roc_auc': roc_auc_score(all_labels, all_preds)
                }
                history['val_metrics'].append(metrics)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f}, "
                    f"val_acc={metrics['accuracy']:.4f}, "
                    f"val_roc_auc={metrics['roc_auc']:.4f}"
                )
        
        # Load best model
        self._load_checkpoint()
        
        return history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict similarity probabilities.
        
        Returns: Array of shape (N,) with values in [0, 1]
        """
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
```

### Hyperparameter Tuning (`scripts/evaluation/tuning/nn_tuning.py`)

```python
def hyperparameter_search():
    """
    Grid search over hyperparameter space.
    
    Search space:
    - hidden_dims: [[256, 128], [512, 256], [128, 64]]
    - dropout: [0.1, 0.2, 0.3]
    - learning_rate: [0.001, 0.002, 0.005]
    - weight_decay: [1e-4, 1e-5, 1e-6]
    - batch_size: [32, 64]
    
    Evaluation:
    - 3-fold cross-validation
    - Metric: ROC-AUC (most important for ranking)
    - Total configurations: 54
    - Time: ~5.4 hours
    
    Best configuration:
    - hidden_dims: [256, 128]
    - dropout: 0.3
    - learning_rate: 0.002
    - weight_decay: 1e-5
    - batch_size: 64
    - CV ROC-AUC: 97.17%
    """
    param_grid = {
        'hidden_dims': [[256, 128], [512, 256], [128, 64]],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.002, 0.005],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'batch_size': [32, 64]
    }
    
    results = []
    for params in ParameterGrid(param_grid):
        cv_scores = cross_validate_model(params, X_train, y_train, n_folds=3)
        results.append({
            'params': params,
            'mean_roc_auc': np.mean(cv_scores),
            'std_roc_auc': np.std(cv_scores)
        })
    
    best = max(results, key=lambda x: x['mean_roc_auc'])
    return best
```

---

# Inference Pipeline

## PatentAnalyzer (`src/app/patent_analyzer.py`)

### Main Orchestration Class

```python
class PatentAnalyzer:
    """
    Main analyzer orchestrating the entire pipeline.
    
    Components:
    1. PatentSBERTa embedder
    2. Local search (cosine similarity)
    3. Online search (SerpAPI)
    4. PyTorch classifier
    5. Phi-3 explainer
    """
    
    def __init__(
        self,
        patents_path: str = 'data/sampled/patents_sampled.jsonl',
        embeddings_path: str = 'data/embeddings/patent_embeddings.npy',
        patent_ids_path: str = 'data/embeddings/patent_ids.json',
        use_online_search: bool = True,
        use_llm_keywords: bool = True,
        serpapi_key: str = None
    ):
        self.patents_path = patents_path
        self.embeddings_path = embeddings_path
        self.patent_ids_path = patent_ids_path
        
        # Components (loaded on-demand)
        self.embedder = None
        self.embeddings = None
        self.patent_ids = None
        self.patents = {}
        self.classifier = None
        self.feature_extractor = None
        self.explainer = None
        self.online_searcher = None
        self.keyword_extractor = None
        
        # Configuration
        self.use_online_search = use_online_search
        self.use_llm_keywords = use_llm_keywords
        self.serpapi_key = serpapi_key
    
    def load(self, status_callback=None):
        """
        Load all components.
        
        Lazy loading pattern:
        - Only load what's needed
        - Load heavy components last
        - Report progress via callback
        """
        if status_callback:
            status_callback("Loading embeddings...")
        
        # Load embeddings (memory-mapped for efficiency)
        self.embeddings = np.load(self.embeddings_path, mmap_mode='r')
        
        # Load patent IDs
        with open(self.patent_ids_path, 'r') as f:
            self.patent_ids = json.load(f)
        
        if status_callback:
            status_callback(f"Loaded {len(self.patent_ids)} embeddings")
        
        # Load PatentSBERTa
        if status_callback:
            status_callback("Loading PatentSBERTa model...")
        from src.embeddings.patent_sberta import PatentEmbedder
        self.embedder = PatentEmbedder()
        
        # Load online search
        if self.use_online_search:
            if status_callback:
                status_callback("Initializing online search...")
            from data.api.online_search import GooglePatentsSearch
            self.online_searcher = GooglePatentsSearch(serpapi_key=self.serpapi_key)
        
        # Load LLM keyword extractor
        if self.use_llm_keywords:
            if status_callback:
                status_callback("Initializing LLM keyword extractor...")
            from data.api.online_search import LLMKeywordExtractor
            self.keyword_extractor = LLMKeywordExtractor()
        
        # Load Phi-3 explainer
        if status_callback:
            status_callback("Initializing Phi-3 explainer...")
        from src.app.phi3_explainer import Phi3OllamaExplainer
        self.explainer = Phi3OllamaExplainer()
        
        # Load PyTorch classifier
        if status_callback:
            status_callback("Loading PyTorch model...")
        self.classifier = PyTorchPatentClassifier()
        self.classifier.load('models/pytorch_nn')
        
        # Load feature extractor
        from src.features.feature_extract import FeatureExtractor
        self.feature_extractor = FeatureExtractor(
            embeddings=self.embeddings,
            patent_ids=self.patent_ids,
            embedder=self.embedder
        )
        
        if status_callback:
            status_callback("Ready!")
    
    def analyze(
        self, 
        query_text: str, 
        top_k: int = 15,
        status_callback=None
    ) -> 'NoveltyReport':
        """
        Complete analysis pipeline.
        
        Steps:
        1. Generate query embedding
        2. Local search (cosine similarity)
        3. Online search (LLM keywords + SerpAPI)
        4. Merge results
        5. Extract features
        6. Score with PyTorch model
        7. Generate LLM explanation
        8. Return NoveltyReport
        
        Time: 60-90 seconds
        """
        
        # Step 1: Generate embedding
        if status_callback:
            status_callback("🔄 Generating embeddings...")
        
        query_embedding = self.embedder.encode(query_text)
        
        # Step 2: Local search
        if status_callback:
            status_callback("🔍 Searching local database (200K patents)...")
        
        local_results = self._find_similar(
            query_embedding, 
            top_k=top_k,
            status_callback=status_callback
        )
        
        # Step 3: Online search
        online_results = []
        if self.use_online_search and self.online_searcher:
            if status_callback:
                status_callback("🌐 Extracting search keywords with LLM...")
            
            # Generate keywords
            keywords = []
            if self.use_llm_keywords and self.keyword_extractor:
                keywords = self.keyword_extractor.extract_keywords(query_text)
                if status_callback:
                    status_callback(f"Generated {len(keywords)} search terms: {keywords[:3]}...")
            else:
                # Fallback: extract from text
                keywords = self._extract_keywords_heuristic(query_text)
            
            if status_callback:
                status_callback("🌐 Searching Google Patents online...")
            
            online_results = self.online_searcher.search_multiple_terms(
                keywords,
                max_per_term=10
            )
            
            if status_callback:
                status_callback(f"Online search found {len(online_results)} unique patents")
        
        # Step 4: Merge results
        all_candidates = self._merge_results(local_results, online_results)
        
        if status_callback:
            status_callback(f"⚖️ Scoring {len(all_candidates)} candidates with PyTorch model...")
        
        # Step 5: Extract features and score
        scored_patents = []
        for candidate in all_candidates:
            # Extract features
            features = self.feature_extractor.extract_features(
                query_text,
                candidate
            )
            
            # Score with model
            similarity_score = self.classifier.predict_proba(
                features.reshape(1, -1)
            )[0]
            
            candidate['model_similarity'] = float(similarity_score)
            candidate['model_novelty'] = 1 - similarity_score
            scored_patents.append(candidate)
        
        # Step 6: Rank by score
        scored_patents.sort(key=lambda x: x['model_similarity'], reverse=True)
        top_scored = scored_patents[:20]  # Keep top 20 for explanation
        
        # Compute novelty score
        mean_similarity = np.mean([p['model_similarity'] for p in top_scored])
        novelty_score = 1 - mean_similarity
        rank_percentile = (1 / len(top_scored)) * 100 if top_scored else 50.0
        
        if status_callback:
            status_callback(
                f"Ranking-based assessment: scored {len(top_scored)} patents, "
                f"mean similarity={mean_similarity:.3f}, "
                f"novelty={novelty_score:.3f}, "
                f"percentile={rank_percentile:.1f}%"
            )
        
        # Step 7: Generate explanation
        if status_callback:
            status_callback("🤖 Generating explanation with Phi-3...")
        
        explanation = self.explainer.generate_explanation(
            query_patent={'text': query_text},
            similar_patents=top_scored[:10],
            novelty_score=novelty_score
        )
        
        # Step 8: Create report
        report = NoveltyReport(
            query_text=query_text,
            novelty_score=novelty_score,
            rank_percentile=rank_percentile,
            similar_patents=top_scored,
            explanation=explanation,
            search_metadata={
                'local_results': len(local_results),
                'online_results': len(online_results),
                'total_candidates': len(all_candidates),
                'top_k_scored': len(top_scored)
            }
        )
        
        return report
    
    def _find_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        status_callback=None
    ) -> List[Dict]:
        """
        Find similar patents using cosine similarity.
        
        Algorithm:
        1. Normalize query and database embeddings
        2. Compute dot products (cosine similarity)
        3. Get top-k indices
        4. Load patent metadata
        5. Return results with similarity scores
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        all_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Cosine similarity
        similarities = np.dot(all_norms, query_norm)
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Load patents
        results = []
        for idx in top_indices:
            pid = self.patent_ids[idx]
            patent_data = self._load_patent(str(pid)) or {}
            results.append({
                'patent_id': pid,
                'similarity': float(similarities[idx]),
                'title': patent_data.get('title', 'N/A'),
                'abstract': patent_data.get('abstract', 'N/A'),
                'year': patent_data.get('year', 'N/A'),
                'claims': patent_data.get('claims', []),
                'source': 'local'
            })
        
        return results
    
    def _load_patent(self, patent_id: str) -> Optional[Dict]:
        """
        Load patent from JSONL database.
        
        Lazy loading:
        - Only load when needed
        - Cache in memory
        - Read single line from JSONL (efficient)
        """
        if patent_id in self.patents:
            return self.patents[patent_id]
        
        # Search JSONL file
        with open(self.patents_path, 'r') as f:
            for line in f:
                patent = json.loads(line)
                if patent.get('patent_id') == patent_id or patent.get('id') == patent_id:
                    self.patents[patent_id] = patent
                    return patent
        
        return None
```

## Online Search (`data/api/online_search.py`)

### LLM Keyword Extraction

```python
class LLMKeywordExtractor:
    """
    Uses Phi-3 to generate intelligent search terms.
    
    Advantage over heuristics:
    - Understands context
    - Generates synonyms
    - Creates boolean queries
    - Domain-aware
    """
    
    def __init__(self, ollama_endpoint="http://localhost:11434"):
        self.endpoint = f"{ollama_endpoint}/api/generate"
        self.model = "phi3"
    
    def extract_keywords(self, patent_text: str, num_terms: int = 5) -> List[str]:
        """
        Generate search terms using LLM.
        
        Prompt engineering:
        - Ask for Google Patents syntax
        - Request boolean operators
        - Limit to num_terms
        """
        prompt = f"""You are a patent search expert. Given this patent text, generate {num_terms} search terms for Google Patents.

Patent text:
{patent_text[:1000]}

Generate {num_terms} diverse search terms using Google Patents syntax (AND, OR). Each term should target a different aspect of the invention. Output only the search terms, one per line.

Search terms:"""
        
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Some creativity
                    "num_predict": 200
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            text = response.json()['response']
            # Parse lines
            terms = [line.strip() for line in text.split('\n') if line.strip()]
            terms = [t for t in terms if len(t) > 10]  # Filter short ones
            return terms[:num_terms]
        
        return []


class GooglePatentsSearch:
    """
    Search Google Patents via SerpAPI.
    
    Why SerpAPI:
    - Google Patents API deprecated
    - SerpAPI provides structured results
    - Handles pagination and rate limiting
    """
    
    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY')
        self.use_serpapi = SERPAPI_AVAILABLE and bool(self.serpapi_key)
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search single query.
        
        Returns:
        [
            {
                'patent_id': 'US12345678',
                'title': str,
                'abstract': str,  # Snippet only
                'year': int,
                'url': str,
                'source': 'online'
            }
        ]
        """
        if not self.use_serpapi:
            return []
        
        params = {
            "engine": "google_patents",
            "q": query,
            "num": num_results,
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        patents = []
        for result in results.get('organic_results', []):
            patents.append({
                'patent_id': result.get('patent_id', 'N/A'),
                'title': result.get('title', 'N/A'),
                'abstract': result.get('snippet', 'N/A'),
                'year': self._parse_year(result.get('publication_date')),
                'url': result.get('link', ''),
                'source': 'online'
            })
        
        return patents
    
    def search_multiple_terms(
        self, 
        terms: List[str],
        max_per_term: int = 10
    ) -> List[Dict]:
        """
        Search multiple terms and deduplicate.
        
        Process:
        1. Search each term
        2. Collect all results
        3. Deduplicate by patent_id
        4. Return unique patents
        """
        seen_ids = set()
        all_patents = []
        
        for i, term in enumerate(terms, 1):
            logger.info(f"[{i}/{len(terms)}] Searching term: '{term[:50]}...'")
            
            results = self.search(term, num_results=max_per_term)
            
            for patent in results:
                if patent['patent_id'] not in seen_ids:
                    all_patents.append(patent)
                    seen_ids.add(patent['patent_id'])
        
        logger.info(f"Total online patents found: {len(all_patents)} across {len(terms)} terms")
        
        return all_patents
```

## Phi-3 Explainer (`src/app/phi3_explainer.py`)

### LLM-Based Explanation Generation

```python
class Phi3OllamaExplainer:
    """
    Generates human-readable novelty explanations using Phi-3.
    
    Why Phi-3:
    - 3.8B params (good quality/size trade-off)
    - Runs locally (privacy)
    - Fast inference with Ollama
    - Domain-adaptable through prompting
    """
    
    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434",
        model_name: str = "phi3",
        max_new_tokens: int = 1200
    ):
        self.api_endpoint = f"{ollama_endpoint}/api/generate"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
    
    def generate_explanation(
        self,
        query_patent: Dict,
        similar_patents: List[Dict],
        novelty_score: float
    ) -> str:
        """
        Generate detailed novelty assessment.
        
        Prompt structure:
        1. Role definition (patent examiner)
        2. Context (query patent + similar patents)
        3. Task (assess novelty, cite prior art)
        4. Format (structured report)
        
        Output sections:
        - Executive Summary
        - Technical Overlap Analysis
        - Novelty Concerns
        - Recommendation
        """
        
        # Build prompt
        prompt = self._build_prompt(query_patent, similar_patents, novelty_score)
        
        # Call Ollama
        response = requests.post(
            self.api_endpoint,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_new_tokens,
                    "temperature": 0.4,  # Lower for more factual
                    "top_p": 0.9
                }
            },
            timeout=180  # 3 minutes max
        )
        
        if response.status_code == 200:
            explanation = response.json()['response']
            return explanation
        else:
            return "Error generating explanation."
    
    def _build_prompt(
        self, 
        query_patent: Dict, 
        similar_patents: List[Dict],
        novelty_score: float
    ) -> str:
        """
        Constructs the LLM prompt.
        
        Prompt engineering principles:
        - Clear role definition
        - Specific task instructions
        - Structured output format
        - Examples (few-shot if needed)
        - Constraints (cite sources, be specific)
        """
        
        # Determine novelty level
        if novelty_score > 0.7:
            assessment = "HIGH NOVELTY"
        elif novelty_score > 0.4:
            assessment = "MODERATE NOVELTY"
        else:
            assessment = "LOW NOVELTY"
        
        # Format similar patents
        prior_art_text = ""
        for i, patent in enumerate(similar_patents[:5], 1):
            similarity = patent.get('model_similarity', patent.get('similarity', 0))
            prior_art_text += f"""
Patent {i}: {patent.get('patent_id', 'N/A')} (Similarity: {similarity*100:.1f}%)
Title: {patent.get('title', 'N/A')}
Abstract: {patent.get('abstract', 'N/A')[:300]}...
Year: {patent.get('year', 'N/A')}

"""
        
        prompt = f"""You are a patent examiner analyzing a patent application for novelty.

PATENT APPLICATION:
{query_patent.get('text', '')[:1500]}

TOP SIMILAR PRIOR ART:
{prior_art_text}

NOVELTY SCORE: {novelty_score:.3f} ({assessment})
Scale: 0.0 = Not Novel (high similarity) | 1.0 = Highly Novel (low similarity)

Your task: Provide a detailed novelty assessment following this structure:

## EXECUTIVE SUMMARY
Brief determination of novelty level and key reasoning (2-3 sentences).

## TECHNICAL OVERLAP ANALYSIS
For each of the top 3 most similar patents:
- Patent ID and similarity score
- Specific overlapping concepts
- Key quotes from prior art that overlap with the application
- Which claims are affected

## NOVELTY CONCERNS
List specific claims or features that face novelty challenges based on prior art.

## RECOMMENDATION
Based on novelty score and prior art analysis:
- Overall patentability assessment
- Suggestions for improving novelty (if applicable)
- Key differentiators to emphasize (if any)

Be specific. Cite patent IDs and similarity scores. Quote prior art when relevant.

Assessment:"""
        
        return prompt
```

## Feature Extraction (`src/features/feature_extract.py`)

### Runtime Feature Computation

```python
class FeatureExtractor:
    """
    Extracts 10 features for query-candidate pairs during inference.
    
    Difference from training:
    - Training: Batch processing, offline
    - Inference: Real-time, per-pair
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        patent_ids: List[str],
        embedder: 'PatentEmbedder'
    ):
        self.embeddings = embeddings
        self.patent_ids = patent_ids
        self.embedder = embedder
        
        # Create ID to index mapping
        self.id_to_idx = {pid: i for i, pid in enumerate(patent_ids)}
    
    def extract_features(
        self,
        query_text: str,
        candidate_patent: Dict
    ) -> np.ndarray:
        """
        Extract 10 features for a single pair.
        
        Returns: Shape (10,) array
        """
        features = []
        
        # Get candidate embedding
        candidate_id = candidate_patent.get('patent_id')
        if candidate_id in self.id_to_idx:
            candidate_emb = self.embeddings[self.id_to_idx[candidate_id]]
        else:
            # Online patent - generate embedding
            candidate_text = candidate_patent.get('abstract', '')
            candidate_emb = self.embedder.encode(candidate_text)
        
        # Generate query embedding
        query_emb = self.embedder.encode(query_text)
        
        # Feature 1: Embedding cosine similarity
        emb_sim = self._cosine_sim(query_emb, candidate_emb)
        features.append(emb_sim)
        
        # Feature 2: TF-IDF similarity
        tfidf_sim = self._tfidf_sim(query_text, candidate_patent.get('abstract', ''))
        features.append(tfidf_sim)
        
        # Feature 3: Jaccard similarity
        jaccard_sim = self._jaccard_sim(query_text, candidate_patent.get('abstract', ''))
        features.append(jaccard_sim)
        
        # Features 4-10: Metadata features
        # (Same as training - omitted for brevity)
        # ...
        
        return np.array(features, dtype=np.float32)
```

---

# Web Application

## Streamlit App (`app.py`)

### Main Application Structure

```python
import streamlit as st
from src.app.patent_analyzer import PatentAnalyzer

st.set_page_config(
    page_title="Patent Novelty Assessment",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_analyzer(serpapi_key: str = None, use_online: bool = True, use_keywords: bool = True):
    """
    Load analyzer with caching.
    
    Caching strategy:
    - Load once per session
    - Persist in memory
    - Reload only if config changes
    """
    analyzer = PatentAnalyzer(
        use_online_search=use_online,
        use_llm_keywords=use_keywords,
        serpapi_key=serpapi_key
    )
    analyzer.load()
    return analyzer

def main():
    st.title("📄 Patent Novelty Assessment System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        serpapi_key = st.text_input(
            "SerpAPI Key (for online search)",
            value=st.session_state.get('serpapi_key', ''),
            type="password",
            help="Enter your SerpAPI key to enable online patent search"
        )
        
        if serpapi_key != st.session_state.get('serpapi_key', ''):
            st.session_state['serpapi_key'] = serpapi_key
            if serpapi_key:
                os.environ['SERPAPI_KEY'] = serpapi_key
                st.success(f"SerpAPI key configured ({len(serpapi_key)} characters)")
            else:
                st.info("SerpAPI key removed")
        
        # Search settings
        use_online = st.checkbox(
            "Enable Online Search",
            value=True,
            help="Search Google Patents via SerpAPI"
        )
        
        use_keywords = st.checkbox(
            "Use LLM for Keyword Extraction",
            value=True,
            help="Use Phi-3 to generate intelligent search terms"
        )
        
        num_results = st.slider(
            "Number of results",
            min_value=5,
            max_value=30,
            value=15,
            help="Top-K similar patents to retrieve"
        )
    
    # Main tabs
    tab1, tab2 = st.tabs(["Novelty Assessment", "Prior Art Search"])
    
    with tab1:
        novelty_assessment_tab(
            serpapi_key=serpapi_key,
            use_online=use_online,
            use_keywords=use_keywords,
            num_results=num_results
        )
    
    with tab2:
        prior_art_search_tab(
            serpapi_key=serpapi_key,
            num_results=num_results
        )

def novelty_assessment_tab(serpapi_key, use_online, use_keywords, num_results):
    """
    Main novelty assessment interface.
    
    Workflow:
    1. User enters patent text
    2. Click "Analyze Patent Novelty"
    3. Show real-time progress
    4. Display results:
       - Novelty score
       - Similar patents table
       - AI explanation
       - Download options
    """
    
    st.header("Assess Patent Novelty")
    
    # Input area
    query_text = st.text_area(
        "Enter patent details (Title, Abstract, Claims):",
        height=300,
        placeholder="Paste your patent text here..."
    )
    
    if st.button("🔍 Analyze Patent Novelty", type="primary"):
        if not query_text.strip():
            st.error("Please enter patent text.")
            return
        
        # Load analyzer
        with st.spinner("Loading models..."):
            analyzer = load_analyzer(serpapi_key, use_online, use_keywords)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_status(msg):
            status_text.text(msg)
        
        # Run analysis
        try:
            result = analyzer.analyze(
                query_text,
                top_k=num_results,
                status_callback=update_status
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            display_results(result)
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            logger.exception(e)

def display_results(result):
    """
    Display novelty assessment results.
    
    Layout:
    - Score card (prominent)
    - Metrics row
    - Similar patents table
    - AI explanation (expandable)
    - Download buttons
    """
    
    # Novelty score
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        score = result.novelty_score
        
        # Color coding
        if score > 0.7:
            color = "green"
            interpretation = "High Novelty - Likely Patentable"
        elif score > 0.4:
            color = "orange"
            interpretation = "Moderate Novelty - Further Review Needed"
        else:
            color = "red"
            interpretation = "Low Novelty - Potential Prior Art Found"
        
        st.metric(
            label="Novelty Score",
            value=f"{score:.3f}",
            delta=interpretation
        )
    
    with col2:
        st.metric(
            label="Similar Patents Found",
            value=len(result.similar_patents)
        )
    
    with col3:
        percentile = result.rank_percentile
        st.metric(
            label="Rank Percentile",
            value=f"{percentile:.1f}%"
        )
    
    # Similar patents table
    st.subheader("📊 Similar Patents")
    
    df_data = []
    for p in result.similar_patents[:15]:
        df_data.append({
            'Patent ID': p.get('patent_id', 'N/A'),
            'Title': p.get('title', 'N/A')[:60] + '...',
            'Similarity': f"{p.get('model_similarity', 0):.3f}",
            'Year': p.get('year', 'N/A'),
            'Source': p.get('source', 'N/A')
        })
    
    st.dataframe(df_data, use_container_width=True)
    
    # AI Explanation
    st.subheader("🤖 AI-Generated Novelty Explanation")
    
    with st.expander("View Detailed Explanation", expanded=True):
        st.markdown(result.explanation)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        txt_report = generate_text_report(result)
        st.download_button(
            label="📄 Download Text Report",
            data=txt_report,
            file_name="novelty_report.txt",
            mime="text/plain"
        )
    
    with col2:
        json_report = generate_json_report(result)
        st.download_button(
            label="📋 Download JSON Report",
            data=json_report,
            file_name="novelty_report.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
```

---

# Key Files Deep Dive

## 1. Input Handler (`src/app/input_handler.py`)

**Purpose:** Parse and validate different input formats.

**Key Methods:**

```python
class InputHandler:
    @staticmethod
    def parse_text_input(text: str) -> Dict:
        """
        Parse free-form text into structured format.
        
        Handles:
        - "Title: ...\nAbstract: ..."
        - JSON-like formatting
        - Plain text (entire text as abstract)
        """
        
    @staticmethod
    def parse_json_input(json_data: Union[str, Dict]) -> Dict:
        """
        Parse JSON patent format.
        
        Expected format:
        {
            "title": str,
            "abstract": str,
            "claims": List[str] or List[Dict]
        }
        """
    
    @staticmethod
    def parse_csv_upload(csv_file) -> List[Dict]:
        """
        Parse CSV file with multiple patents.
        
        Columns: patent_id, title, abstract, claims
        """
```

## 2. Patent Embedder (`src/embeddings/patent_sberta.py`)

**Purpose:** Wrapper around PatentSBERTa model.

**Key Methods:**

```python
class PatentEmbedder:
    def __init__(self, model_name="AI-Growth-Lab/PatentSBERTa"):
        self.model = SentenceTransformer(model_name)
        
        # Auto device selection
        if torch.cuda.is_available():
            self.model.to('cuda')
        elif torch.backends.mps.is_available():
            self.model.to('mps')
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Array of shape (N, 768) where N = number of texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        
        return embeddings
```

## 3. Claim Embeddings (`src/features/claim_embeddings.py`)

**Purpose:** Generate embeddings for individual patent claims.

**Why separate from abstract:**
- Claims are legally binding
- More specific than abstract
- Useful for fine-grained comparison

**Key Method:**

```python
def compute_claim_similarity(claims_a: List[str], claims_b: List[str]) -> float:
    """
    Maximum similarity across all claim pairs.
    
    Process:
    1. Embed all claims from both patents
    2. Compute pairwise similarities
    3. Return maximum (most similar pair)
    
    Captures: "Do any claims overlap significantly?"
    """
    embedder = PatentEmbedder()
    
    embs_a = embedder.encode(claims_a)
    embs_b = embedder.encode(claims_b)
    
    max_sim = 0.0
    for emb_a in embs_a:
        for emb_b in embs_b:
            sim = cosine_similarity(
                emb_a.reshape(1, -1),
                emb_b.reshape(1, -1)
            )[0][0]
            max_sim = max(max_sim, sim)
    
    return max_sim
```

---

# Complete Workflow Example

## End-to-End Patent Analysis

```
User enters:
"A method for wireless power transfer using magnetic resonance..."

↓

[app.py]
- Parses input via InputHandler
- Loads PatentAnalyzer (cached)
- Calls analyzer.analyze()

↓

[patent_analyzer.py]
1. embedder.encode(query_text)
   → Returns (768,) embedding

2. _find_similar(embedding, top_k=15)
   → Cosine similarity across 200K
   → Returns 15 local patents

3. keyword_extractor.extract_keywords(query_text)
   → Phi-3 generates: [
       "wireless power transfer AND magnetic resonance",
       "inductive charging system OR resonant coupling",
       ...
     ]

4. online_searcher.search_multiple_terms(keywords)
   → SerpAPI queries Google Patents
   → Returns ~50 online patents

5. _merge_results(local, online)
   → Deduplicates
   → Returns ~60 unique candidates

6. For each candidate:
     features = feature_extractor.extract_features(query, candidate)
     score = classifier.predict_proba(features)
     candidate['model_similarity'] = score

7. Sort by score, keep top 20

8. Compute novelty_score = 1 - mean(top_20_scores)

9. explainer.generate_explanation(query, top_20, novelty_score)
   → Phi-3 generates detailed report

10. Return NoveltyReport object

↓

[app.py]
- Displays novelty score
- Shows similar patents table
- Renders AI explanation
- Provides download buttons

↓

User sees:
Novelty Score: 0.234 (Low Novelty)
Similar Patents: 18 found
Explanation: "Based on analysis of prior art..."
```

---

# Performance Characteristics

## Time Breakdown

| Stage | Time | Bottleneck |
|-------|------|------------|
| Input parsing | <1s | - |
| Query embedding | 2-3s | PatentSBERTa inference |
| Local search | 1-2s | NumPy operations (optimized) |
| LLM keywords | 10-15s | Phi-3 generation |
| Online search | 15-25s | Network I/O to SerpAPI |
| Feature extraction | 2-3s | Per-patent computation |
| Model scoring | 1-2s | PyTorch inference (batched) |
| LLM explanation | 30-45s | Phi-3 generation (long text) |
| **Total** | **60-90s** | |

## Memory Usage

- PatentSBERTa model: ~400 MB
- Embeddings (mmap): ~586 MB (virtual, not loaded fully)
- PyTorch model: ~20 MB
- Phi-3 (Ollama): ~2.3 GB (separate process)
- Patent cache: Variable (grows with usage)

**Total peak: ~3-4 GB**

## Accuracy

- Training accuracy: 91.73%
- Validation ROC-AUC: 97.17%
- Test ROC-AUC: 97.20%
- Retrieval Recall@10: 99.91%
- Mean Reciprocal Rank: 99.96%

---

# Key Design Decisions

## 1. Why NumPy Cosine Instead of FAISS?

**Decision:** Use pure NumPy for vector search.

**Rationale:**
- 200K vectors is manageable for NumPy
- Avoids FAISS dependency and complexity
- NumPy dot product is highly optimized
- Memory-mapping keeps RAM usage low
- Simplifies deployment

**Trade-off:** Slower than FAISS for >1M vectors, but fast enough for our scale.

## 2. Why Hybrid (Local + Online)?

**Decision:** Combine local database with online search.

**Rationale:**
- Local: Fast, semantic, high-quality embeddings
- Online: Comprehensive coverage, historical patents
- Together: Best of both worlds

**Trade-off:** Adds 20-30s to inference time.

## 3. Why PyTorch Over sklearn?

**Decision:** Custom PyTorch network instead of sklearn models.

**Rationale:**
- Better accuracy (97.2% vs 94.5% ROC-AUC)
- Residual blocks handle feature interactions
- Batch normalization improves generalization
- GPU acceleration (MPS/CUDA)
- Extensible architecture

**Trade-off:** More code complexity, longer training time.

## 4. Why Local LLM (Phi-3) Not Cloud API?

**Decision:** Run Phi-3 locally via Ollama.

**Rationale:**
- Privacy: Patent data never leaves user's machine
- Cost: No per-query API fees
- Reliability: No dependency on external services
- Control: Can fine-tune prompts and settings

**Trade-off:** Slower inference (30-45s vs 5-10s for GPT-4).

## 5. Why Positive-Unlabeled Learning?

**Decision:** Train on citation pairs (positive) + random pairs (unlabeled).

**Rationale:**
- Citations are confirmed relationships
- Cannot definitively label all negatives
- Random non-citations are likely negative
- Aligns with patent examiner workflow

**Trade-off:** Some false negatives in training data, but model still achieves 97% ROC-AUC.

---

# Conclusion

This technical walkthrough covers:

1. **Data Pipeline**: Collection → Sampling → Embedding → Feature Engineering
2. **Model Training**: PyTorch architecture → Hyperparameter tuning → 97% ROC-AUC
3. **Inference Pipeline**: Local search → Online search → ML scoring → LLM explanation
4. **Web Application**: Streamlit UI → Real-time processing → Downloadable reports

**Key Strengths:**
- Hybrid RAG combining local and online search
- Supervised ML on real citation data
- Local LLM for privacy and interpretability
- 60-90s end-to-end for comprehensive assessment

**Technology Stack:**
- PatentSBERTa (embeddings)
- PyTorch (ML scoring)
- Phi-3 (explanations)
- SerpAPI (online search)
- Streamlit (UI)
- NumPy (vector operations)

This system demonstrates a complete machine learning pipeline from data collection through deployment, with strong emphasis on interpretability and real-world usability.

