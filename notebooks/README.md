# Notebooks

This folder contains Jupyter notebooks for analysis and demonstration.

## Available Notebooks

### `end_to_end_pipeline.ipynb`

**Purpose**: Demonstrates the complete production pipeline

**What it does:**
- Loads all models and data
- Processes example patent application
- Runs complete novelty assessment pipeline
- Shows step-by-step breakdown of each component

**Usage:**
```bash
jupyter notebook notebooks/end_to_end_pipeline.ipynb
```

**Pipeline Steps:**
1. Initialize Patent Analyzer
2. Process input (title, abstract, claims)
3. Retrieve similar patents (FAISS)
4. Extract 13 features for each pair
5. Classify using PyTorch Neural Network
6. Generate novelty score
7. Create LLM explanation

**Outputs:**
- Console output showing each step
- Final novelty assessment with explanation

## Requirements

All notebooks require:
- Jupyter notebook or JupyterLab
- Project dependencies (see `requirements.txt`)
- Trained models in `models/` directory
- Preprocessed data in `data/` directory

## Running Notebooks

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open notebook** from the Jupyter interface

4. **Run all cells** sequentially (Cell → Run All)

## Notes

- Notebooks use relative paths from project root
- Some cells may take several minutes to run
- Ensure all models are trained before running notebooks

