# Setup Instructions

Follow these steps to run the Patent Novelty Assessment System locally from start to finish.

## Prerequisites

### System Requirements
- Python 3.9 or higher (Python 3.10+ recommended)
- 16 GB RAM recommended (8 GB minimum)
- ~15 GB free disk space for data, embeddings, and model files
- macOS, Linux, or Windows
- Internet connection for initial setup (downloading dependencies and models)

### Required Accounts/Keys
- **SerpAPI key** for online patent search via Google Patents
  - Sign up at https://serpapi.com/ (free tier available with 100 searches/month)
  - Required for hybrid retrieval with online search capability

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/abhinavmeduri/CS372-final-project.git
cd CS372-final-project
```

### Step 2: Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch and scikit-learn for ML models
- sentence-transformers for PatentSBERTa embeddings
- Streamlit for the web interface
- google-search-results for SerpAPI integration
- Other utilities (numpy, pandas, matplotlib, etc.)

### Step 4: Install and Configure Ollama (for Phi-3 LLM)

Ollama is required to run the Phi-3 local LLM for generating patent novelty explanations.

**macOS:**
```bash
brew install ollama
brew services start ollama
ollama pull phi3
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull phi3
```

**Windows:**
- Download from https://ollama.ai/download
- Install and run the Ollama application
- Open terminal and run: `ollama pull phi3`

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```
You should see a JSON response listing available models including `phi3`.

## Data Setup

The application requires several large data files that are not included in the repository due to size constraints. These files are essential for the application to run.

### Download Required Files from Box

**Box Link:** https://duke.box.com/s/4y6mjf1965d15gnltnkqnk0dkedbttqh

**Note:** This is a Duke Box link. You may need Duke network credentials to access it. If you don't have access, contact the repository maintainer.

**Package Contents:**
- Patent embeddings and metadata (~7 GB total)
- Trained model files (~2 MB)
- Package size: ~4-5 GB compressed, ~7-8 GB uncompressed

### What's Included in the Box Download

The Box package contains the following files:

**Required Files (Essential for app to run):**
1. `patent_embeddings.npy` (~3.2 GB) - 200K pre-computed patent embeddings
2. `patent_ids.json` (~15 MB) - Patent ID to embedding index mappings
3. `patents_sampled.jsonl` (~3.8 GB) - Patent metadata database (200K patents)
4. `pytorch_model.pt` (~2 MB) - Trained PyTorch neural network classifier
5. `scaler_pytorch.pkl` (~20 KB) - Feature normalization scaler

**Optional Files (Included for reference/documentation):**
6. `training_history_pytorch.json` (~5 KB) - Training curves and metrics (not needed for inference)
7. `embedding_metadata.json` - Metadata about the embedding generation process
8. `mlp_model.pkl`, `scaler.pkl`, `mlp_metrics.json` - Baseline MLP model files (for comparison)

### Installation Instructions

After downloading the Box package, follow these steps to set up the data files:

**Step 1: Download and Extract**
```bash
# Download the file from Box (it will be named something like CS372-patent-data.zip)
# Extract the archive
unzip CS372-patent-data.zip
# This will create a folder with all the required files
```

**Step 2: Place Files in Correct Directories**

Navigate to your cloned repository and create the necessary directories if they don't exist:

```bash
cd CS372-final-project

# Create directories if needed
mkdir -p data/embeddings
mkdir -p data/sampled
mkdir -p models/pytorch_nn
```

**Step 3: Move Files to Appropriate Locations**

From the extracted Box download folder, move files to their correct locations:

```bash
# Move embedding files
mv path/to/extracted/patent_embeddings.npy data/embeddings/
mv path/to/extracted/patent_ids.json data/embeddings/

# Move patent database
mv path/to/extracted/patents_sampled.jsonl data/sampled/

# Move model files (REQUIRED)
mv path/to/extracted/pytorch_model.pt models/pytorch_nn/
mv path/to/extracted/scaler_pytorch.pkl models/pytorch_nn/

# Optional: Move training metadata (not required for app)
mv path/to/extracted/training_history_pytorch.json models/pytorch_nn/ 2>/dev/null || true

# Optional: Move embedding metadata if included
mv path/to/extracted/embedding_metadata.json data/embeddings/ 2>/dev/null || true

# Optional: Move MLP baseline files if included
mkdir -p models/mlp
mv path/to/extracted/mlp_model.pkl models/mlp/ 2>/dev/null || true
mv path/to/extracted/scaler.pkl models/mlp/ 2>/dev/null || true
mv path/to/extracted/mlp_metrics.json models/mlp/ 2>/dev/null || true
```

**Alternative: If Box download preserves directory structure**

If the Box download already contains the files in a `data/` and `models/` folder structure:

```bash
# Simply copy the entire data and models folders
cp -r path/to/extracted/data/* data/
cp -r path/to/extracted/models/* models/
```

### Verify File Structure

After moving the files, your directory structure should look like this:

```
CS372-final-project/
├── data/
│   ├── embeddings/
│   │   ├── patent_embeddings.npy      # REQUIRED: 200K patent embeddings (~3.2 GB)
│   │   ├── patent_ids.json            # REQUIRED: Patent ID mapping (~15 MB)
│   │   └── embedding_metadata.json    # (optional) Embedding metadata
│   ├── sampled/
│   │   └── patents_sampled.jsonl      # REQUIRED: 200K patent metadata (~3.8 GB)
│   └── features/
│       └── feature_names_v2.json      # Feature names (already in repo)
├── models/
│   ├── pytorch_nn/
│   │   ├── pytorch_model.pt           # REQUIRED: Trained classifier (~2 MB)
│   │   ├── scaler_pytorch.pkl         # REQUIRED: Feature scaler (~20 KB)
│   │   └── training_history_pytorch.json  # (optional) Training metadata
│   └── mlp/                           # (optional) Baseline model
│       ├── mlp_model.pkl
│       ├── scaler.pkl
│       └── mlp_metrics.json
└── ...
```

### Verification Checklist

Run these commands to verify all **required** files are in place:

```bash
# Check embeddings (REQUIRED)
ls -lh data/embeddings/patent_embeddings.npy
ls -lh data/embeddings/patent_ids.json

# Check patent database (REQUIRED)
ls -lh data/sampled/patents_sampled.jsonl

# Check model files (REQUIRED)
ls -lh models/pytorch_nn/pytorch_model.pt
ls -lh models/pytorch_nn/scaler_pytorch.pkl
```

**Expected output:**
- `patent_embeddings.npy` should be ~3.2 GB
- `patent_ids.json` should be ~15 MB
- `patents_sampled.jsonl` should be ~3.8 GB
- `pytorch_model.pt` should be ~2 MB
- `scaler_pytorch.pkl` should be ~20 KB

**Critical files checklist (required for app to run):**
- [ ] `data/embeddings/patent_embeddings.npy` (200K patent vectors, ~3.2 GB)
- [ ] `data/embeddings/patent_ids.json` (ID mappings, ~15 MB)
- [ ] `data/sampled/patents_sampled.jsonl` (patent database, ~3.8 GB)
- [ ] `models/pytorch_nn/pytorch_model.pt` (trained classifier, ~2 MB)
- [ ] `models/pytorch_nn/scaler_pytorch.pkl` (feature scaler, ~20 KB)

**If any files are missing or sizes don't match, re-download from Box or contact the repository maintainer.**

### Verify Data Integrity (Optional)

To verify the data files loaded correctly, run this Python script:

```python
import numpy as np
import json
from pathlib import Path

# Check embeddings
embeddings = np.load('data/embeddings/patent_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")  # Should be (200000, 768)

# Check patent IDs
with open('data/embeddings/patent_ids.json', 'r') as f:
    patent_ids = json.load(f)
print(f"Patent IDs count: {len(patent_ids)}")  # Should be 200000

# Check patent database
import json
count = 0
with open('data/sampled/patents_sampled.jsonl', 'r') as f:
    for line in f:
        count += 1
print(f"Patent database entries: {count}")  # Should be 200000

# Check model files
import torch
model_state = torch.load('models/pytorch_nn/pytorch_model.pt', map_location='cpu')
print(f"Model loaded successfully")

print("\nAll data files verified and ready!")
```

## Environment Configuration

### Set SerpAPI Key

The application requires a SerpAPI key for online patent search functionality.

**Option 1: Export in shell session (temporary)**
```bash
export SERPAPI_KEY=your_serpapi_key_here
```

**Option 2: Create a `.env` file (persistent)**
```bash
echo "SERPAPI_KEY=your_serpapi_key_here" > .env
```

**Option 3: Export alternative variable name**
```bash
export SERPAPI_API_KEY=your_serpapi_key_here
```

The application will check for both `SERPAPI_KEY` and `SERPAPI_API_KEY`.

## Running the Application

### Start the Streamlit Web App

From the project root directory:

```bash
streamlit run app.py --server.port=8501
```

The application will start and automatically open in your browser at:
- **URL:** http://localhost:8501

**If port 8501 is already in use:**
```bash
streamlit run app.py --server.port=8503
```

### Using the Application

1. **Enter a patent query** in the text area (title, abstract, or claims)
2. **Configure settings** in the sidebar:
   - Enable/disable online search (requires SerpAPI key)
   - Enable/disable LLM keyword extraction
   - Adjust number of results to return
3. **Click "Analyze Patent Novelty"**
4. **Wait for analysis to complete** (typically 30-90 seconds):
   - Loading patent data from disk
   - Performing local semantic search (200K patents)
   - Conducting online search via SerpAPI (5 search terms)
   - Scoring candidates with PyTorch model
   - Generating AI explanation with Phi-3
5. **Review results:**
   - Similar patents with similarity scores
   - Patent metadata (publication date, assignee, CPC codes)
   - AI-generated novelty assessment and explanation

**Note:** First query may take longer (60-120 seconds) as the system loads embeddings into memory and initializes models. Subsequent queries will be faster due to caching.

### Running the Pipeline Notebook (Optional)

To execute the full pipeline demonstration notebook:

```bash
SERPAPI_KEY=your_serpapi_key_here \
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb \
  --output pipeline.exec.ipynb --output-dir notebooks
```

Or run it interactively:
```bash
jupyter notebook notebooks/pipeline.ipynb
```

## Verification and Testing

### Quick Smoke Test

1. **Start the Streamlit app** as described above
2. **Enter a sample patent query:**
   ```
   A method for wireless power transfer using magnetic resonance coupling between transmitter and receiver coils with automatic frequency tuning for maximum efficiency.
   ```
3. **Enable online search** (ensure SERPAPI_KEY is set)
4. **Click "Analyze Patent Novelty"**
5. **Verify the following:**
   - [ ] Similar patents are retrieved and displayed
   - [ ] Similarity scores are shown for each patent
   - [ ] Online search results appear (if enabled)
   - [ ] AI-generated novelty explanation is displayed
   - [ ] No error messages in terminal or browser console

### Verify Ollama and Phi-3

Check that Ollama is running and Phi-3 is available:

```bash
# Check Ollama service
curl http://localhost:11434/api/tags

# Test Phi-3 generation
curl http://localhost:11434/api/generate -d '{
  "model": "phi3",
  "prompt": "Explain what a patent is in one sentence.",
  "stream": false
}'
```

You should receive a JSON response with generated text.

### Verify Data Files

Verify embeddings and patent database are loaded correctly:

```python
# Run in Python or IPython
import numpy as np
import json

# Check embeddings
embeddings = np.load('data/embeddings/patent_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")  # Should be (200000, 768)

# Check patent IDs
with open('data/embeddings/patent_ids.json', 'r') as f:
    patent_ids = json.load(f)
print(f"Number of patent IDs: {len(patent_ids)}")  # Should be 200000
```

## Troubleshooting

### Common Issues and Solutions

**1. Port already in use**
```bash
# Find and kill process using the port
lsof -i :8501
kill -9 <PID>

# Or use a different port
streamlit run app.py --server.port=8503
```

**2. Ollama not running**
```bash
# macOS
brew services start ollama

# Linux
ollama serve &

# Verify
curl http://localhost:11434/api/tags
```

**3. Phi-3 model not found**
```bash
ollama pull phi3
ollama list  # Verify phi3 is in the list
```

**4. Missing Python packages**
```bash
pip install --upgrade -r requirements.txt

# Specific packages:
pip install sentence-transformers
pip install google-search-results
pip install torch
```

**5. SerpAPI errors (online search not working)**
- Verify your API key is set correctly: `echo $SERPAPI_KEY`
- Check you haven't exceeded free tier limits (100 searches/month)
- Ensure `google-search-results` package is installed

**6. File not found errors (embeddings/data)**
- Verify all files were extracted from Box download
- Check file paths match the expected structure
- Run the verification script above

**7. Out of memory errors**
- Close other applications to free RAM
- Reduce batch size in configuration if training models
- Use a machine with at least 8 GB RAM (16 GB recommended)

**8. PyTorch/CUDA errors**
- The application uses CPU by default (FAISS-CPU)
- If you have GPU: install `faiss-gpu` and `torch` with CUDA support
- For Apple Silicon: PyTorch will use Metal Performance Shaders automatically

## First Run Notes

- **Initial load time:** First Streamlit app launch may take 30-60 seconds to load embeddings into memory
- **First LLM call:** First Phi-3 generation may take 10-20 seconds as Ollama loads the model
- **Subsequent runs:** Much faster due to caching (embeddings memory-mapped, Ollama KV cache active)
- **Online search:** First SerpAPI call will verify your API key

## Additional Resources

- **Project README:** `README.md` for architecture and methodology details
- **Attribution:** `ATTRIBUTION.md` for data sources and model credits
- **Requirements:** `requirements.txt` for full dependency list
- **Notebooks:** `notebooks/pipeline.ipynb` for end-to-end pipeline demonstration

## Getting Help

If you encounter issues not covered here:
1. Check the terminal/console output for detailed error messages
2. Verify all prerequisites and data files are correctly set up
3. Review the troubleshooting section above
4. Check that all services (Ollama, Streamlit) are running

