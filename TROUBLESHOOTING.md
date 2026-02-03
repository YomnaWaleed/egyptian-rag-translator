# 🔧 Troubleshooting Guide

Common issues and solutions for the Egyptian RAG Translator.

## Common Errors

### ❌ KeyError: 'embedding'

**Error message:**
```
KeyError: 'embedding'
```

**Cause:** The script is trying to load embeddings, but they haven't been generated yet or the wrong file is being loaded.

**Solution:**

1. Make sure you've run the embedding generation:
   ```bash
   python -m src.embeddings.embedder
   ```

2. If running individual scripts, use the correct order:
   ```bash
   # Correct order
   python -m src.preprocessing.loader
   python -m src.preprocessing.normalization
   python -m src.preprocessing.lemmatization
   python -m src.preprocessing.split
   python -m src.embeddings.embedder        # Generate embeddings FIRST
   python -m src.retrieval.qdrant_store     # Then build database
   python -m src.retrieval.bm25_index       # Then build BM25
   ```

3. Or use the setup script which handles everything:
   ```bash
   python setup.py
   ```

### ❌ FileNotFoundError: train_embeddings.npy

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data/embeddings/train_embeddings.npy'
```

**Cause:** Embeddings haven't been generated yet.

**Solution:**
```bash
# Generate embeddings
python -m src.embeddings.embedder
```

This will create:
- `./data/embeddings/train_embeddings.npy` (embeddings)
- `./data/processed/tla_train_with_embeddings.csv` (data + embeddings)

### ❌ OLLAMA_API_KEY not set

**Error message:**
```
ValueError: OLLAMA_API_KEY not set in environment variables
```

**Cause:** Missing or incorrect API key configuration.

**Solution:**

1. Check your `.env` file exists:
   ```bash
   ls -la .env
   ```

2. Add your API key to `.env`:
   ```bash
   OLLAMA_API_KEY=your_actual_api_key_here
   ```

3. Make sure there are no spaces around the `=`:
   ```bash
   # Correct ✅
   OLLAMA_API_KEY=sk-abc123

   # Wrong ❌
   OLLAMA_API_KEY = sk-abc123
   ```

### ❌ Collection not found

**Error message:**
```
qdrant_client.exceptions.UnexpectedResponse: Collection `egyptian_transliterations` not found
```

**Cause:** Qdrant database hasn't been built yet.

**Solution:**
```bash
# Build the Qdrant database
python -m src.retrieval.qdrant_store
```

Or run the complete setup:
```bash
python setup.py
```

### ❌ BM25 index not found

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data/processed/bm25_corpus.pkl'
```

**Cause:** BM25 index hasn't been built yet.

**Solution:**
```bash
# Build BM25 index
python -m src.retrieval.bm25_index
```

### ❌ ImportError: sys.meta_path is None

**Error message:**
```
ImportError: sys.meta_path is None, Python is likely shutting down
```

**Cause:** This is a harmless warning during Python shutdown when using Qdrant. It doesn't affect functionality.

**Solution:** You can safely ignore this warning. It appears when the program exits.

### ❌ Out of Memory (OOM)

**Error message:**
```
RuntimeError: CUDA out of memory
```
or
```
MemoryError
```

**Cause:** Not enough RAM or GPU memory for processing large batches.

**Solution:**

1. Reduce batch size in `.env`:
   ```bash
   BATCH_SIZE=16  # Default is 32
   ```

2. Or edit `src/config/settings.py`:
   ```python
   BATCH_SIZE = 16  # Reduce from 32
   ```

3. If using GPU, try CPU-only:
   ```python
   import torch
   torch.set_default_device('cpu')
   ```

## Workflow Issues

### Issue: Scripts must be run in order

**Problem:** Running scripts out of order causes dependency errors.

**Solution:** Use the correct order:

```bash
# Option 1: Use setup script (recommended)
python setup.py

# Option 2: Run individual scripts in order
python -m src.preprocessing.loader           # Step 1
python -m src.preprocessing.normalization    # Step 2
python -m src.preprocessing.lemmatization    # Step 3
python -m src.preprocessing.split            # Step 4
python -m src.embeddings.embedder            # Step 5 ⚠️ CRITICAL
python -m src.retrieval.qdrant_store         # Step 6 (needs Step 5)
python -m src.retrieval.bm25_index           # Step 7
```

### Issue: Data files not found

**Problem:** Scripts can't find CSV files.

**Solution:**

1. Check data directories exist:
   ```bash
   ls -la data/raw/
   ls -la data/processed/
   ls -la data/embeddings/
   ```

2. Verify files are created at each step:
   ```bash
   # After loader
   ls -la data/raw/tla_original.csv

   # After split
   ls -la data/processed/tla_train.csv
   ls -la data/processed/tla_test.csv

   # After embeddings
   ls -la data/embeddings/train_embeddings.npy
   ```

## Performance Issues

### Slow embedding generation

**Symptoms:** Embedding generation takes >1 hour

**Solutions:**

1. **Use GPU if available:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

2. **Increase batch size** (if you have enough RAM):
   ```bash
   BATCH_SIZE=64  # Increase from 32
   ```

3. **Use smaller model** (faster but less accurate):
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   VECTOR_DIM = 384  # Update accordingly
   ```

### Slow translation

**Symptoms:** Each translation takes >30 seconds

**Solutions:**

1. **Reduce TOP_K_RESULTS** in `.env`:
   ```bash
   TOP_K_RESULTS=25  # Reduce from 55
   ```

2. **Use faster LLM model:**
   ```bash
   LLM_MODEL=qwen3-next:32b-cloud  # Smaller model
   ```

## Data Quality Issues

### Issue: Translations seem incorrect

**Possible causes:**
1. Not enough retrieved examples
2. Wrong LLM model
3. Poor embedding quality

**Solutions:**

1. **Increase retrieved examples:**
   ```bash
   TOP_K_RESULTS=100  # Increase from 55
   ```

2. **Try different LLM:**
   ```bash
   LLM_MODEL=qwen3-vl:235b-instruct-cloud  # Best quality
   ```

3. **Regenerate embeddings** with better model:
   ```bash
   EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
   ```

## Getting Help

### Debug mode

Enable detailed logging:

```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check versions

```bash
pip list | grep -E "qdrant|transformers|torch|pandas"
```

### Verify setup

```bash
python -m src.utils  # Should show train/test counts and embedding dimension
```

### Reset everything

If all else fails, start fresh:

```bash
# Delete all generated data
rm -rf data/raw/*.csv
rm -rf data/processed/*.csv
rm -rf data/embeddings/*.npy
rm -rf qdrant_db/

# Run setup again
python setup.py
```

## Still Having Issues?

1. Check the [README.md](README.md) for detailed documentation
2. Review the [QUICKSTART.md](QUICKSTART.md) guide
3. Examine error messages carefully
4. Check file paths and permissions
5. Verify Python version (3.8+ required)

---

**Most Common Solution:** Run `python setup.py` to regenerate everything in the correct order! 🎯