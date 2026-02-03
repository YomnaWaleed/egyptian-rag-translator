# 🚀 Quick Start Guide

Get up and running with the Egyptian RAG Translator in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- Ollama Cloud API key

## Installation

### Step 1: Clone and Install

```bash
cd egyptian-rag-translator
pip install -r requirements.txt
```

### Step 2: Configure API Key

Edit `.env` file:

```bash
# Add your Ollama API key
OLLAMA_API_KEY=your_api_key_here
```

### Step 3: Run Setup

This will download the dataset, build indexes, and prepare everything:

```bash
python setup.py
```

⏱️ **Expected time**: 15-30 minutes (depending on your machine)

The setup script will:
1. ✅ Load TLA dataset from HuggingFace
2. ✅ Clean and normalize transliterations
3. ✅ Extract lemmas
4. ✅ Create train/test split
5. ✅ Generate embeddings (BGE-M3)
6. ✅ Build Qdrant vector database
7. ✅ Build BM25 sparse index

## Usage

### Basic Translation

```bash
python main.py "ḥtp dj njswt"
```

**Expected output:**
```
🏛️ Egyptian:  ḥtp dj njswt
🇩🇪 German:    Ein Opfer, das der König gibt
🇬🇧 English:   An offering which the king gives
```

### Quiet Mode (No Details)

```bash
python main.py "ḥtp dj njswt" --no-details
```

### Python API

```python
from src.pipeline.rag_pipeline import RAGPipeline

# Initialize (only once)
pipeline = RAGPipeline(in_memory=False)

# Translate
result = pipeline.translate("ḥtp dj njswt", show_details=True)

print(f"German:  {result['german']}")
print(f"English: {result['english']}")
```

## Example Queries

Try these Egyptian phrases:

```bash
# Offering formula
python main.py "ḥtp dj njswt"

# King's name
python main.py "nsw-bjt khnty"

# Greeting
python main.py "jj.k m ḥtp"
```

## Customization

### Change Models

Edit `src/config/settings.py`:

```python
# Use different LLM
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"

# Use different embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"
```

### Adjust Search Settings

```python
# Number of retrieved examples
TOP_K_RESULTS = 30

# Hybrid search weight
HYBRID_SEARCH_ALPHA = 0.5
```

## Troubleshooting

### "OLLAMA_API_KEY not set"
- Make sure `.env` file exists and contains your API key
- Check that the key is valid

### "Collection not found"
- Run `python setup.py` to build the database

### Slow performance
- First run loads models (slow)
- Subsequent runs are much faster
- Consider using GPU for embeddings

### Out of memory
- Reduce `BATCH_SIZE` in settings
- Use smaller embedding model
- Process data in chunks

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore individual components in `src/`
- Test with your own Egyptian texts
- Experiment with different LLM models

## Support

For issues and questions:
- Check the [README.md](README.md)
- Review source code documentation
- Open an issue on GitHub

---

**Happy translating!** 🏛️→🇩🇪→🇬🇧