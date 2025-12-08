# Setup Guide - Patent Novelty Assessment System

Complete end-to-end setup instructions for the Patent Novelty Assessment System.

## Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 16GB recommended (8GB minimum)
- **Disk Space**: ~50GB for data and models
- **Operating System**: macOS, Linux, or Windows

## Step 1: Clone Repository

```bash
git clone <repository-url>
cd CS372-final-project
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

## Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch 2.0+ (for neural network model)
- sentence-transformers (for PatentSBERTa embeddings)
- FAISS (for vector similarity search)
- Streamlit (for web interface)
- google-search-results (for SerpAPI online search)

## Step 4: Install Ollama (for LLM Explanations)

The system uses Ollama to run Phi-3 locally for generating patent explanations.

### macOS
```bash
brew install ollama
brew services start ollama
ollama pull phi3
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3
```

### Windows
Download and install from: https://ollama.ai/download

Then run:
```bash
ollama pull phi3
```

**Verify installation:**
```bash
ollama list
# Should show: phi3
```

## Step 5: Download Required Data Files

The application requires pre-processed data files. These are too large for GitHub and must be downloaded separately.

### Required Files

1. **Patent Database** (~3.8GB)
   - File: `data/sampled/patents_sampled.jsonl`
   - Contains: 200,000 USPTO patents (2021-2025)
   - Location: `data/sampled/`

2. **Embeddings** (~1.5GB)
   - File: `data/embeddings/patent_embeddings.npy`
   - Contains: Pre-computed PatentSBERTa embeddings
   - Location: `data/embeddings/`

3. **Patent ID Mapping**
   - File: `data/embeddings/patent_ids.json`
   - Contains: Mapping between patent IDs and embedding indices
   - Location: `data/embeddings/`

4. **Trained Model Files**
   - `models/pytorch_nn/pytorch_model.pt` - Trained PyTorch model (10 features)
   - `models/pytorch_nn/scaler_pytorch.pkl` - Feature scaler
   - `models/pytorch_nn/training_history_pytorch.json` - Training metadata
   - Location: `models/pytorch_nn/`

5. **Feature Names**
   - `data/features/feature_names_v2.json` - Feature names (included in repo)
   - Location: `data/features/`

### Download Options

**Option A: Download from Shared Location**
- Contact repository maintainer for download links
- Files may be hosted on Google Drive, Dropbox, or similar

**Option B: Generate from Raw Data**
If you have access to PatentsView raw data files:

```bash
# 1. Download raw PatentsView data (2021-2025)
# Place TSV files in data/{year}/ directories

# 2. Process raw data
python scripts/data/preprocessing/process_patents.py

# 3. Generate embeddings
python scripts/data/preprocessing/generate_embeddings.py

# 4. Sample patents
python scripts/data/preprocessing/sample_patents.py

# 5. Train model (if needed)
python scripts/training/train_pytorch.py
```

**Option C: Use Existing Data**
If you have access to a shared data directory, copy files to the appropriate locations.

## Step 6: Verify Data Files

Check that all required files exist:

```bash
# Check patent database
ls -lh data/sampled/patents_sampled.jsonl

# Check embeddings
ls -lh data/embeddings/patent_embeddings.npy
ls -lh data/embeddings/patent_ids.json

# Check model files
ls -lh models/pytorch_nn/pytorch_model.pt
ls -lh models/pytorch_nn/scaler_pytorch.pkl

# Check feature names
ls -lh data/features/feature_names_v2.json
```

## Step 7: Configure API Keys (Optional)

For online patent search via Google Patents, you need a SerpAPI key.

1. **Get SerpAPI Key:**
   - Sign up at https://serpapi.com/
   - Get your free API key from the dashboard

2. **Set Environment Variable:**
   ```bash
   # macOS/Linux
   export SERPAPI_KEY=your_key_here
   
   # Windows (PowerShell)
   $env:SERPAPI_KEY="your_key_here"
   
   # Or add to ~/.bashrc or ~/.zshrc for persistence
   echo 'export SERPAPI_KEY=your_key_here' >> ~/.zshrc
   ```

3. **Or Set in Streamlit UI:**
   - The key can be entered directly in the Streamlit sidebar
   - No need to set environment variable if using UI input

## Step 8: Verify Installation

### Test Ollama Connection

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models including phi3
```

### Test Python Imports

```bash
python -c "import torch; import streamlit; import faiss; print('All imports successful')"
```

### Test Model Loading

```bash
python -c "
from src.models.pytorch_classifier import PyTorchPatentClassifier
model = PyTorchPatentClassifier()
model.load('models/pytorch_nn')
print('Model loaded successfully')
"
```

## Step 9: Run the Application

### Start Streamlit App

```bash
streamlit run app.py
```

The application will:
1. Load the patent database and embeddings
2. Load the trained PyTorch model
3. Initialize the Phi-3 explainer (via Ollama)
4. Start the web server

**Access the app:**
- Local URL: http://localhost:8501
- Network URL: Displayed in terminal

### First Run

On first run, the application will:
- Load 200,000 patent embeddings (~30 seconds)
- Load PatentSBERTa model (~10 seconds)
- Initialize Phi-3 explainer (~5 seconds)
- Load PyTorch model (~2 seconds)

**Total startup time:** ~45-60 seconds

## Step 10: Test the System

### Test Novelty Assessment

1. Open the Streamlit app in your browser
2. Go to "Novelty Assessment & Prior Art Search" tab
3. Enter a patent description in the text area
4. Click "Analyze Novelty"
5. Verify that:
   - Novelty score is displayed
   - Similar patents are found
   - Explanation is generated

### Test Online Search (if SerpAPI key configured)

1. Enter your SerpAPI key in the sidebar
2. Enable "Enable Online Search" checkbox
3. Run an analysis
4. Verify that patents from online search appear (marked with [Online])

## Troubleshooting

### Ollama Not Running

**Error:** "Cannot connect to Ollama"

**Solution:**
```bash
# Start Ollama service
brew services start ollama  # macOS
# OR
systemctl start ollama      # Linux

# Verify
ollama list
```

### Model Files Missing

**Error:** "Model file not found"

**Solution:**
- Ensure all model files are in `models/pytorch_nn/`
- Check file permissions
- Verify file sizes match expected values

### Embeddings Not Found

**Error:** "Embeddings file not found"

**Solution:**
- Ensure `data/embeddings/patent_embeddings.npy` exists
- Check file size (~1.5GB)
- Verify `data/embeddings/patent_ids.json` exists

### Memory Issues

**Error:** "Out of memory" or slow performance

**Solution:**
- Close other applications
- Reduce batch size in model configuration
- Use CPU-only FAISS if GPU memory is limited
- Consider using smaller sample of patents

### SerpAPI Errors

**Error:** "SerpAPI key not configured"

**Solution:**
- Verify API key is correct
- Check API key has remaining credits
- Ensure key is set in environment or UI

## Next Steps

- Read `README.md` for system overview and features
- Check `docs/PROJECT_DOCUMENTATION.md` for technical details
- Review `notebooks/pipeline.ipynb` for example usage
- See `ATTRIBUTION.md` for data and model sources

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review documentation in `docs/`
3. Check GitHub issues
4. Contact repository maintainer

---

*Last Updated: December 2025*

