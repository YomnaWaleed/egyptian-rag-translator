# 🏛️ Egyptian RAG Translator

A Retrieval-Augmented Generation (RAG) system for translating Earlier Egyptian transliterations to English via German, using hybrid search (dense + sparse) and LLM-powered translation.

## 📋 Features

- **Hybrid Retrieval**: Combines dense vector search (Qdrant) with sparse BM25 search using Reciprocal Rank Fusion (RRF)
- **LLM Translation**: Uses Ollama Cloud API for Egyptian→German translation with linguistic context
- **Neural Translation**: MarianMT for German→English translation
- **Comprehensive Preprocessing**: Normalization, lemmatization, and data cleaning for Egyptian texts
- **Modular Architecture**: Clean separation of concerns with reusable components

## 🏗️ Architecture

```
Egyptian Text → Normalization → Embedding → Hybrid Search (Qdrant + BM25)
                                                ↓
                                        Retrieved Examples
                                                ↓
                                        LLM Translation (→ German)
                                                ↓
                                        Neural Translation (→ English)
```

## 📁 Project Structure

```
egyptian-rag-translator/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (not in git)
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned and processed data
│   └── embeddings/               # Pre-computed embeddings
│
├── qdrant_db/                    # Vector database storage
│
└── src/
│   ├── config/
│   │   └── settings.py           # Configuration and constants
│   │
│   ├── preprocessing/
│   │   ├── loader.py             # Dataset loading
│   │   ├── normalization.py      # Transliteration normalization
│   │   ├── lemmatization.py      # Lemma extraction
│   │   └── split.py              # Train/test splitting
│   │
│   ├── embeddings/
│   │   └── embedder.py           # Embedding generation
│   │
│   ├── retrieval/
│   │   ├── qdrant_store.py       # Vector database operations
│   │   ├── bm25_index.py         # BM25 sparse search
│   │   └── hybrid_search.py      # Hybrid retrieval (RRF)
│   │
│   ├── llm/
│   │   ├── ollama_client.py      # Ollama API client
│   │   └── prompt_templates.py   # LLM prompts
│   │
│   ├── translation/
│   │   ├── egyptian_to_german.py # Egyptian→German translation
│   │   └── german_to_english.py  # German→English translation
│   │
│   └── pipeline/
│       └── rag_pipeline.py       # Complete translation pipeline
│
│
└── ui/
    └── 
    └── 

```

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YomnaWaleed/egyptian-rag-translator.git
cd egyptian-rag-translator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env` and add your Ollama API key:

```bash
cp .env.example .env
# Edit .env and add your OLLAMA_API_KEY
```

Required environment variables:
- `OLLAMA_API_KEY`: Your Ollama Cloud API key

### 4. Prepare Data

Run the preprocessing pipeline:

```bash
# Load and clean dataset
python -m src.preprocessing.loader

# Normalize transliterations
python -m src.preprocessing.normalization

# Extract lemmas
python -m src.preprocessing.lemmatization

# Create train/test split
python -m src.preprocessing.split
```

### 5. Build Indexes

```bash
# Generate embeddings
python -m src.embeddings.embedder

# Build Qdrant vector database
python -m src.retrieval.qdrant_store

# Build BM25 index
python -m src.retrieval.bm25_index
```

## 💻 Usage

### Complete Pipeline

```python
from src.pipeline.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(in_memory=False)

# Translate Egyptian text
query = "ḥtp dj njswt"
result = pipeline.translate(query, show_details=True)

if result['success']:
    print(f"Egyptian:  {result['query_original']}")
    print(f"German:    {result['german']}")
    print(f"English:   {result['english']}")
```

### Individual Components

#### Normalization
```python
from src.preprocessing.normalization import normalize_transliteration

text = "ḥtp dj njswt"
normalized = normalize_transliteration(text)
print(normalized)  # "htp dj njswt"
```

#### Embedding Generation
```python
from src.embeddings.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()
embedding = embedder.generate_single("htp dj njswt")
```

#### Hybrid Search
```python
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.bm25_index import BM25Index

# Load components
qdrant_store = QdrantStore(in_memory=False)
bm25_index = BM25Index()
bm25_index.load_index()

# Create searcher
searcher = HybridSearcher(qdrant_store, bm25_index)

# Search
results = searcher.search(query_text, query_embedding, top_k=10)
```

#### Translation
```python
from src.translation.egyptian_to_german import translate_egyptian_to_german
from src.translation.german_to_english import translate_german_to_english

# Egyptian → German
german, _ = translate_egyptian_to_german(query_original, query_normalized, examples)

# German → English
english = translate_german_to_english(german)
```

## 🔧 Configuration

Key settings in `src/config/settings.py`:

```python
# Models
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Search
TOP_K_RESULTS = 30
HYBRID_SEARCH_ALPHA = 0.5

# Training
TRAIN_SPLIT = 0.99
VECTOR_DIM = 1024
BATCH_SIZE = 32
```

## 📊 Dataset

Uses the TLA (Thesaurus Linguae Aegyptiae) dataset:
- **Source**: `thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium`
- **Language**: Earlier Egyptian (Old Egyptian & Early Middle Egyptian)
- **Content**: Transliterations, translations, lemmas, POS tags, glossing

## 🔍 How It Works

### 1. Preprocessing
- Load TLA dataset from HuggingFace
- Clean data (remove duplicates, missing values)
- Normalize transliterations (Egyptian→Latin)
- Extract lemmas and linguistic features

### 2. Indexing
- Generate embeddings using BGE-M3
- Store in Qdrant vector database
- Build BM25 index for keyword search

### 3. Retrieval (Hybrid Search)
- **Dense Search**: Semantic similarity via embeddings
- **Sparse Search**: Keyword matching via BM25
- **Fusion**: Combine using Reciprocal Rank Fusion (RRF)

### 4. Translation
- **Egyptian→German**: LLM with retrieved examples as context
- **German→English**: Neural machine translation (MarianMT)

## 📈 Performance

The system uses:
- **Reciprocal Rank Fusion** for optimal retrieval
- **Top-K=55** retrieved examples for rich linguistic context
- **BGE-M3** embeddings (1024-dim) for semantic understanding
- **Large LLM** (Qwen 3 VL 235B) for accurate translation

## 🛠️ Development

### Running Tests

```bash
# Test individual components
python -m src.preprocessing.normalization
python -m src.embeddings.embedder
python -m src.retrieval.hybrid_search
python -m src.pipeline.rag_pipeline
```

### Adding New Features

The modular design makes it easy to:
- Swap embedding models (edit `src/embeddings/embedder.py`)
- Change LLM providers (edit `src/llm/ollama_client.py`)
- Modify prompts (edit `src/llm/prompt_templates.py`)
- Add new search methods (create new file in `src/retrieval/`)

## 📝 Citation

If you use this system, please cite:

```bibtex
@software{egyptian_rag_translator,
  title={Egyptian RAG Translator},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/egyptian-rag-translator}
}
```

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Your Contact Information]

## 🙏 Acknowledgments

- **TLA Dataset**: Thesaurus Linguae Aegyptiae
- **Embedding Model**: BAAI/bge-m3
- **Translation Model**: Helsinki-NLP/opus-mt-de-en
- **LLM**: Ollama Cloud (Qwen models)