# 🏛️ Egyptian RAG Translator - Developer Documentation

Complete technical documentation for the Egyptian RAG Translation system development, architecture, and evaluation.

## 📑 Table of Contents

- [System Overview](#system-overview)
- [Architecture Details](#architecture-details)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Data Pipeline](#data-pipeline)
- [RAG System Components](#rag-system-components)
- [Evaluation Framework](#evaluation-framework)
- [Performance Metrics](#performance-metrics)
- [Development Workflow](#development-workflow)
- [Testing & Debugging](#testing--debugging)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## System Overview

### What Was Built

A complete Retrieval-Augmented Generation (RAG) system for translating Earlier Egyptian transliterations to English, with comprehensive evaluation against LLM-only baseline.

### Technology Stack

- **Language Models:**
  - LLM: `qwen3-vl:235b-instruct-cloud` (Ollama Cloud API)
  - Embedding: `BAAI/bge-m3` (1024-dim, multilingual)
  - Translation: `Helsinki-NLP/opus-mt-de-en` (MarianMT)

- **Vector Database:** Qdrant (persistent storage)
- **Sparse Search:** BM25 (Okapi variant)
- **Hybrid Search:** Reciprocal Rank Fusion (RRF)

- **Frameworks:**
  - HuggingFace Transformers & Datasets
  - Sentence Transformers
  - Qdrant Client
  - rank-bm25

### Key Performance Results

**Configuration:**
- LLM Model: `qwen3-vl:235b-instruct-cloud`
- Embedding Model: `BAAI/bge-m3` (1024-dim)
- Top-K: 30 (optimized)
- Test Set: 91 samples

**RAG vs LLM-Only Comparison:**

| Metric | RAG System | LLM-Only | Improvement |
|--------|------------|----------|-------------|
| BLEU | 23.70% | 3.22% | **+20.48%** |
| ROUGE-1 | 53.93% | 22.08% | **+31.85%** |
| ROUGE-2 | 36.53% | 5.51% | **+31.02%** |
| ROUGE-L | 52.31% | 19.77% | **+32.54%** |
| METEOR | 39.32% | 12.83% | **+26.49%** |
| chrF | 45.35% | 17.34% | **+28.01%** |
| Exact Match | 9.89% | 0.00% | **+9.89%** |
| Word Overlap | 43.36% | 18.43% | **+24.93%** |

**Retrieval Metrics (RAG only):**
- Recall@1: 2.20%
- Recall@10: 2.20%
- MRR: 2.20%
- Avg Retrieval Score: 2.90%

---

## Architecture Details

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Egyptian Text                        │
│                     Example: "ḥtp dj njswt"                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. NORMALIZATION                              │
│  • Remove brackets & combining marks                             │
│  • Map Egyptian chars → Latin (ḥ→h, ḏ→dj, etc.)                 │
│  • Remove suffix pronouns (=f, =k, =s, etc.)                    │
│  Output: "htp dj njswt"                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. EMBEDDING GENERATION                       │
│  Model: BAAI/bge-m3 (1024-dim)                                  │
│  Output: Dense vector [0.009, 0.012, -0.031, ...]              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. HYBRID SEARCH                              │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Dense Search    │         │  Sparse Search   │             │
│  │  (Qdrant)        │         │  (BM25)          │             │
│  │  Cosine similarity│         │  Keyword match   │             │
│  └─────────┬────────┘         └─────────┬────────┘             │
│            │                             │                       │
│            └──────────┬──────────────────┘                       │
│                       │                                          │
│                       ▼                                          │
│            Reciprocal Rank Fusion (RRF)                         │
│            Top-K=30 most relevant examples                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                4. LLM TRANSLATION (Egyptian → German)            │
│  • Build prompt with 30 retrieved examples                      │
│  • Include: transliteration, lemmas, POS, glossing, German      │
│  • LLM: qwen3-vl:235b-instruct-cloud                           │
│  • Output: "Ein Opfer, das der König gibt."                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                5. NEURAL TRANSLATION (German → English)          │
│  Model: Helsinki-NLP/opus-mt-de-en (MarianMT)                  │
│  Output: "A sacrifice given by the King."                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT                                │
│  🏛️ Egyptian:  ḥtp dj njswt                                     │
│  🇩🇪 German:    Ein Opfer, das der König gibt.                  │
│  🇬🇧 English:   A sacrifice given by the King.                  │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Search Algorithm (RRF)

**Reciprocal Rank Fusion Formula:**
```
RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]

where:
  d = document
  k = 60 (constant)
  rank_i(d) = rank of document d in ranking i
  i ∈ {dense_ranking, sparse_ranking}
```

**Process:**
1. Dense search (Qdrant): Get top 60 by cosine similarity
2. Sparse search (BM25): Get top 60 by keyword matching
3. For each document in either ranking:
   - Calculate RRF score from both rankings
   - If only in one ranking, use that rank
4. Sort by final RRF score, return top-K

---

## Project Structure

```
egyptian-rag-translator/
│
├── README.md                      # User-facing documentation
├── DEVELOPER.md                   # This file - technical docs
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (not in git)
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
│
├── setup.py                      # Smart setup script
├── main.py                       # CLI interface
│
├── data/
│   ├── raw/
│   │   └── tla_original.csv      # Original cleaned dataset
│   │
│   ├── processed/
│   │   ├── tla_train.csv         # Training set (99%)
│   │   ├── tla_test.csv          # Test set (1%)
│   │   ├── bm25_corpus.pkl       # BM25 index
│   │   │
│   │   └── (Evaluation outputs - optional)
│   │       ├── rag_evaluation_results.csv
│   │       ├── rag_evaluation_metrics.csv
│   │       ├── llm_only_evaluation_results.csv
│   │       ├── llm_only_evaluation_metrics.csv
│   │       └── comparison_summary.csv
│   │
│   └── embeddings/
│       └── train_embeddings.npy  # Pre-computed embeddings (8997 × 1024)
│
├── qdrant_db/                    # Persistent vector database
│   ├── collection/
│   ├── meta.json
│   └── storage.sqlite
│
├── src/
│   ├── config/
│   │   └── settings.py           # All configuration constants
│   │
│   ├── preprocessing/
│   │   ├── loader.py             # HuggingFace dataset loading
│   │   ├── normalization.py      # Egyptian → Latin normalization
│   │   ├── lemmatization.py      # Extract lemmas from dataset
│   │   └── split.py              # Train/test splitting
│   │
│   ├── embeddings/
│   │   └── embedder.py           # BGE-M3 embedding generation
│   │
│   ├── retrieval/
│   │   ├── qdrant_store.py       # Qdrant database operations
│   │   ├── bm25_index.py         # BM25 index building
│   │   └── hybrid_search.py      # RRF hybrid search
│   │
│   ├── llm/
│   │   ├── ollama_client.py      # Ollama Cloud API wrapper
│   │   └── prompt_templates.py   # Prompt engineering
│   │
│   ├── translation/
│   │   ├── egyptian_to_german.py # LLM-based translation
│   │   └── german_to_english.py  # MarianMT translation
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py       # Complete end-to-end pipeline
│   │
│   └── evaluation/               # Optional evaluation framework
│       ├── metrics.py            # BLEU, ROUGE, METEOR, chrF, Recall@K
│       └── evaluate_rag.py       # RAG vs LLM-only comparison
│
├── scripts/
│   └── run_evaluation.py         # Standalone evaluation runner
│
└── notebooks/
    ├── 01_data_exploration.ipynb # Dataset exploration
    ├── 02_embedding_debug.ipynb  # Embedding visualization
    └── 03_evaluation.ipynb       # Interactive evaluation
```

---

## Development Setup

### Initial Setup from Scratch

```bash
# 1. Create project directory
mkdir egyptian-rag-translator
cd egyptian-rag-translator

# 2. Initialize environment
uv init

# 3. Create virtual environment
uv venv

# 4. Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 5. Install dependencies
uv pip install -r requirements.txt

# 6. Create .env file
echo "OLLAMA_API_KEY=your_key_here" > .env

# 7. Run setup
python setup.py
```

### Environment Variables

Create `.env` file:

```bash
# Required
OLLAMA_API_KEY=your_ollama_api_key

# Optional - Override defaults
OLLAMA_API_URL=https://ollama.com/api/chat
LLM_MODEL=qwen3-vl:235b-instruct-cloud
EMBEDDING_MODEL=BAAI/bge-m3
TOP_K_RESULTS=30
HYBRID_SEARCH_ALPHA=0.5

# Paths
QDRANT_PATH=./qdrant_db
DATA_RAW_PATH=./data/raw
DATA_PROCESSED_PATH=./data/processed
DATA_EMBEDDINGS_PATH=./data/embeddings

# Training
TRAIN_SPLIT=0.99
VECTOR_DIM=1024
BATCH_SIZE=32
```

---

## Data Pipeline

### Dataset: TLA (Thesaurus Linguae Aegyptiae)

**Source:** `thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium`

**Original Fields:**
- `hieroglyphs`: Egyptian hieroglyphic representation
- `transliteration`: Latin transliteration (e.g., "ḥtp dj njswt")
- `lemmatization`: Word|Lemma pairs (e.g., "123|ḥtp 456|dj 789|njswt")
- `UPOS`: Part-of-speech tags
- `glossing`: Word-by-word glosses
- `translation`: German translation (expert-curated)
- `dateNotBefore`, `dateNotAfter`: Dating information

**Statistics:**
- Original: 12,773 records
- After cleaning: 9,088 unique transliterations
- Train set: 8,997 (99%)
- Test set: 91 (1%)

### Preprocessing Steps

#### 1. Loading (`src/preprocessing/loader.py`)

```python
from datasets import load_dataset

dataset = load_dataset(
    "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium",
    split="train"
)
```

**Cleaning:**
- Remove columns: `hieroglyphs`, `dateNotBefore`, `dateNotAfter`
- Drop rows with missing `transliteration` or `translation`
- Remove duplicates (keep first occurrence)
- Result: 9,088 unique records

#### 2. Normalization (`src/preprocessing/normalization.py`)

**Egyptian Character Mapping:**
```python
EGYPTIAN_CHAR_MAP = {
    'ꜣ': 'a',      # vulture (aleph)
    'ꞽ': 'i',      # reed (yodh)
    'ḥ': 'h',      # wick
    'ḫ': 'kh',     # placenta
    'š': 'sh',     # pool
    'ḳ': 'q',      # hill
    'ṯ': 'tj',     # rope
    'ḏ': 'dj',     # cobra
    # ... more mappings
}
```

**Process:**
1. Remove brackets `()` but keep content
2. Normalize Unicode (NFC form)
3. Remove combining marks
4. Lowercase
5. Map special Egyptian characters
6. Remove suffix pronouns (`=f`, `=k`, `=s`, etc.)
7. Clean extra spaces and dots

**Example:**
```
Input:  "ḥtp (dj) njswt=f"
Output: "htp dj njswt"
```

#### 3. Lemmatization (`src/preprocessing/lemmatization.py`)

Extract lemmas from `lemmatization` field:

```python
# Input: "123|ḥtp 456|dj 789|njswt"
# Output: ["ḥtp", "dj", "njswt"]
```

Skip suffixes starting with `=`.

#### 4. Splitting (`src/preprocessing/split.py`)

- Shuffle with seed=42 (reproducibility)
- Split 99% train / 1% test
- Save separately

---

## RAG System Components

### 1. Embedding Generation (`src/embeddings/embedder.py`)

**Model:** `BAAI/bge-m3`
- Multilingual (supports Egyptian-derived text)
- Dimension: 1024
- Normalized vectors (cosine similarity ready)

**Process:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=32
)
```

**Performance:**
- 8,997 texts
- Batch size: 32
- Time: ~30 minutes
- Output: 8997 × 1024 numpy array

### 2. Vector Database (`src/retrieval/qdrant_store.py`)

**Qdrant Configuration:**
```python
VectorParams(
    size=1024,
    distance=Distance.COSINE
)
```

**Payload Structure:**
```python
{
    "transliteration_original": "ḥtp dj njswt",
    "transliteration_normalized": "htp dj njswt",
    "lemmas": ["ḥtp", "dj", "njswt"],
    "UPOS": "NOUN VERB NOUN",
    "glossing": "offering give king",
    "translation_de": "Ein Opfer, das der König gibt"
}
```

**Storage:**
- Persistent: `./qdrant_db/`
- Collection: `egyptian_transliterations`
- Points: 8,997

### 3. BM25 Index (`src/retrieval/bm25_index.py`)

**Algorithm:** Okapi BM25
- Tokenization: Simple space-split
- No stemming (preserves Egyptian forms)
- Saved as pickle: `bm25_corpus.pkl`

**Formula:**
```
BM25(Q,d) = Σ IDF(qi) × (f(qi,d) × (k1 + 1)) / 
                         (f(qi,d) + k1 × (1 - b + b × |d|/avgdl))

where:
  Q = query
  d = document
  qi = query term i
  f(qi,d) = term frequency in document
  k1 = 1.5 (default)
  b = 0.75 (default)
  avgdl = average document length
```

### 4. Hybrid Search (`src/retrieval/hybrid_search.py`)

**Reciprocal Rank Fusion (RRF):**
```python
def hybrid_search(query_text, query_embedding, top_k=30):
    # Dense search
    dense_results = qdrant.query_points(
        query=query_embedding,
        limit=top_k * 2
    )
    
    # Sparse search
    bm25_scores = bm25.get_scores(query_text.split())
    top_bm25_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
    
    # Reciprocal Rank Fusion
    combined_scores = {}
    
    for rank, result in enumerate(dense_results):
        doc_id = result.id
        combined_scores[doc_id] = {
            'rrf_score': 1 / (rank + 60),
            'dense_score': result.score,
            'sparse_score': 0
        }
    
    for rank, idx in enumerate(top_bm25_indices):
        if idx in combined_scores:
            combined_scores[idx]['rrf_score'] += 1 / (rank + 60)
            combined_scores[idx]['sparse_score'] = bm25_scores[idx]
        else:
            # Add new document
            combined_scores[idx] = {...}
    
    # Sort by RRF score
    return sorted(combined_scores.items(), 
                  key=lambda x: x[1]['rrf_score'], 
                  reverse=True)[:top_k]
```

**Why k=60?**
Standard RRF constant, balances rank contribution.

### 5. LLM Translation (`src/llm/ollama_client.py`, `src/llm/prompt_templates.py`)

**Prompt Engineering:**

```python
system_prompt = """You are a senior linguist specializing in Earlier Egyptian 
(Old Egyptian & Early Middle Egyptian), with strong expertise in morphology, 
syntax, and historical semantics.

Your task is to translate an Earlier Egyptian transliteration into German
using retrieved linguistic examples ONLY as structural and semantic guidance."""

user_prompt = f"""
=====================================
QUERY TO TRANSLATE
=====================================
Normalized Transliteration:
{query_normalized}

=====================================
RETRIEVED DATABASE EXAMPLES
=====================================
Example 1:
- Original: ḥtp dj njswt
- Normalized: htp dj njswt
- Lemmas: ḥtp, dj, njswt
- POS Tags: NOUN VERB NOUN
- Glossing: offering give king
- German: Ein Opfer, das der König gibt
---
[... 29 more examples ...]

=====================================
INSTRUCTIONS
=====================================
1. Linguistic Analysis
   - Identify grammatical category of each word
   - Detect verb tense/aspect, suffix pronouns, syntactic order

2. Morphological Alignment
   - Compare suffixes, verb forms, particles with examples
   - Use lemma meanings as semantic hints

3. Translation Construction
   - Produce fluent and historically plausible German translation
   - Adapt word order to correct German syntax
   - Prefer linguistically conservative interpretations

4. Uncertainty Handling
   - Choose most likely reading if multiple possible
   - Mention ambiguity only if materially affects meaning

=====================================
STRICT RULES
=====================================
- DO NOT copy any German translation from examples
- DO NOT mention example numbers or quote them
- DO NOT add explanations unless uncertainty exists
- DO NOT hallucinate missing words
- Base output strictly on Earlier Egyptian grammar

=====================================
OUTPUT FORMAT (STRICT)
=====================================
German Translation: <one clear German sentence>
Confidence: High | Medium | Low
Notes: <only if confidence is Medium or Low>
"""
```

**API Call:**
```python
response = requests.post(
    "https://ollama.com/api/chat",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    },
    json={
        "model": "qwen3-vl:235b-instruct-cloud",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
)
```

### 6. German→English Translation (`src/translation/german_to_english.py`)

**Model:** `Helsinki-NLP/opus-mt-de-en` (MarianMT)

```python
from transformers import MarianMTModel, MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")

inputs = tokenizer(german_text, return_tensors="pt")
outputs = model.generate(**inputs)
english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Why MarianMT?**
- Fast inference
- Good German→English quality
- No API required (runs locally)

---

## Evaluation Framework

### Metrics Implementation (`src/evaluation/metrics.py`)

#### Translation Quality Metrics

**1. BLEU (Bilingual Evaluation Understudy)**
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, hypothesis):
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothing = SmoothingFunction()
    return sentence_bleu(
        [ref_tokens], 
        hyp_tokens,
        smoothing_function=smoothing.method1
    ) * 100
```
- Measures n-gram precision
- Range: 0-100%
- Higher is better

**2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, hypothesis)
```
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- Range: 0-100%

**3. METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
```python
from nltk.translate.meteor_score import meteor_score

score = meteor_score([ref_tokens], hyp_tokens) * 100
```
- Considers synonymy and stemming
- Range: 0-100%

**4. chrF (Character n-gram F-score)**
```python
from sacrebleu.metrics import CHRF

chrf = CHRF()
score = chrf.sentence_score(hypothesis, [reference])
```
- Character-level matching
- Good for morphologically rich languages
- Range: 0-100%

**5. Exact Match**
```python
def calculate_exact_match(reference, hypothesis):
    return 100.0 if ref.strip().lower() == hyp.strip().lower() else 0.0
```

**6. Word Overlap**
```python
def calculate_word_overlap(reference, hypothesis):
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    overlap = len(ref_words.intersection(hyp_words))
    return (overlap / len(ref_words)) * 100
```

#### Retrieval Quality Metrics

**1. Recall@K**
```python
def calculate_recall_at_k(reference_german, retrieved_examples, k_values=[1,3,5,10,20]):
    recalls = {}
    for k in k_values:
        found = False
        for i, example in enumerate(retrieved_examples[:k]):
            if reference_german.lower() == example['payload']['translation_de'].lower():
                found = True
                break
        recalls[f'recall@{k}'] = 100.0 if found else 0.0
    return recalls
```
- Checks if ground truth appears in top K results
- Binary: 100% if found, 0% otherwise

**2. MRR (Mean Reciprocal Rank)**
```python
def calculate_mrr(reference_german, retrieved_examples):
    for i, example in enumerate(retrieved_examples):
        if reference_german.lower() == example['payload']['translation_de'].lower():
            return (1.0 / (i + 1)) * 100
    return 0.0
```
- Reciprocal of rank of first relevant result
- Higher rank = higher score

**3. Average Retrieval Score**
```python
def calculate_average_retrieval_score(retrieved_examples, top_k=10):
    scores = [example['rrf_score'] for example in retrieved_examples[:top_k]]
    return np.mean(scores) * 100
```

### Evaluation Pipeline (`src/evaluation/evaluate_rag.py`)

**RAG Evaluator:**
```python
class RAGEvaluator:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline or RAGPipeline()
        self.de_to_en_translator = GermanToEnglishTranslator()
    
    def evaluate_test_set(self, test_df, max_samples=None):
        results = []
        for idx, row in test_df.iterrows():
            # Translate using RAG
            result = self.pipeline.translate(row['transliteration'])
            
            # Get reference English
            ref_english = self.de_to_en_translator.translate(row['translation'])
            
            # Store results
            results.append({
                'reference_english': ref_english,
                'predicted_english': result['english'],
                'reference_german': row['translation'],
                'top_matches': result['top_matches']
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        return results, metrics
```

**LLM-Only Evaluator:**
```python
class LLMOnlyEvaluator:
    def translate_without_rag(self, query_original):
        # Direct LLM call without retrieved examples
        prompt = f"""Translate this Earlier Egyptian to German:
        {query_original}
        
        Output only the German translation."""
        
        response = ollama_client.generate(prompt)
        return response
```

### Running Evaluation

**Standalone Script:**
```bash
python scripts/run_evaluation.py
```

**Python API:**
```python
from src.evaluation.evaluate_rag import RAGEvaluator, LLMOnlyEvaluator, compare_rag_vs_llm

# Evaluate RAG
rag_evaluator = RAGEvaluator()
rag_results, rag_metrics, _ = rag_evaluator.evaluate_test_set(df_test)

# Evaluate LLM-only
llm_evaluator = LLMOnlyEvaluator()
llm_results, llm_metrics, _ = llm_evaluator.evaluate_test_set(df_test)

# Compare
comparison = compare_rag_vs_llm(rag_metrics, llm_metrics)
```

---

## Performance Metrics

### Complete Evaluation Results

**Test Configuration:**
- Test set: 91 samples (1% of dataset)
- LLM: qwen3-vl:235b-instruct-cloud
- Embedding: BAAI/bge-m3 (1024-dim)
- Top-K: 30 retrieved examples

### Translation Quality

| Metric | RAG System | LLM-Only | Difference | Improvement |
|--------|------------|----------|------------|-------------|
| **BLEU** | 23.70% | 3.22% | +20.48% | **+636%** |
| **ROUGE-1** | 53.93% | 22.08% | +31.85% | **+144%** |
| **ROUGE-2** | 36.53% | 5.51% | +31.02% | **+563%** |
| **ROUGE-L** | 52.31% | 19.77% | +32.54% | **+165%** |
| **METEOR** | 39.32% | 12.83% | +26.49% | **+206%** |
| **chrF** | 45.35% | 17.34% | +28.01% | **+162%** |
| **Exact Match** | 9.89% | 0.00% | +9.89% | **∞** |
| **Word Overlap** | 43.36% | 18.43% | +24.93% | **+135%** |

### Retrieval Quality (RAG only)

| Metric | Score | Description |
|--------|-------|-------------|
| **Recall@1** | 2.20% | Exact match in top result |
| **Recall@3** | 2.20% | Exact match in top 3 |
| **Recall@5** | 2.20% | Exact match in top 5 |
| **Recall@10** | 2.20% | Exact match in top 10 |
| **Recall@20** | 2.20% | Exact match in top 20 |
| **MRR** | 2.20% | Mean Reciprocal Rank |
| **Avg Retrieval Score** | 2.90% | Average RRF score |

### Analysis

**Why RAG Dominates:**

1. **Linguistic Context (20-32% improvement):**
   - Retrieved examples provide grammatical patterns
   - Morphological alignment with similar structures
   - Suffix pronoun matching

2. **Semantic Grounding:**
   - Prevents hallucination
   - Ensures historically plausible translations
   - Vocabulary consistency

3. **Expert Knowledge Transfer:**
   - 9,000 expert-curated translations
   - Covers diverse linguistic constructions
   - Domain-specific terminology

**Retrieval Quality Insights:**

- Low Recall@K (2.20%) indicates exact matches are rare
- This is expected: Egyptian translations are diverse
- Even without exact matches, similar examples provide value
- RRF score (2.90%) shows moderate retrieval relevance

**Why Not Higher Recall?**

- Egyptian is highly context-dependent
- Same transliteration can have different translations
- Test set is small (91 samples)
- Exact match is a very strict criterion

---

## Development Workflow

### Daily Development

```bash
# 1. Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Pull latest changes
git pull origin main

# 3. Install new dependencies (if any)
uv pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/

# 5. Make changes...

# 6. Test specific component
python -m src.preprocessing.normalization
python -m src.retrieval.hybrid_search

# 7. Commit
git add .
git commit -m "Description of changes"
git push origin main
```

### Adding New Features

**Example: Add new embedding model**

1. Edit `src/embeddings/embedder.py`:
```python
class EmbeddingGenerator:
    def __init__(self, model_name="new-embedding-model"):
        self.model = SentenceTransformer(model_name)
```

2. Update `src/config/settings.py`:
```python
EMBEDDING_MODEL = "new-embedding-model"
VECTOR_DIM = 768  # Update if different
```

3. Regenerate embeddings:
```bash
python -m src.embeddings.embedder
```

4. Rebuild Qdrant:
```bash
python -m src.retrieval.qdrant_store
```

**Example: Add new LLM provider**

1. Create `src/llm/new_provider_client.py`:
```python
class NewProviderClient:
    def generate(self, system_prompt, user_prompt):
        # API call implementation
        pass
```

2. Update `src/translation/egyptian_to_german.py`:
```python
from src.llm.new_provider_client import NewProviderClient

class EgyptianToGermanTranslator:
    def __init__(self):
        self.llm_client = NewProviderClient()
```

### Debugging Tips

**1. Check embeddings:**
```python
import numpy as np

embeddings = np.load('./data/embeddings/train_embeddings.npy')
print(f"Shape: {embeddings.shape}")
print(f"Sample: {embeddings[0][:10]}")
```

**2. Test search:**
```python
from src.retrieval.hybrid_search import HybridSearcher

searcher = HybridSearcher(qdrant_store, bm25_index)
results = searcher.search("htp dj njswt", embedding, top_k=5)

for r in results:
    print(f"Score: {r['rrf_score']:.4f}")
    print(f"Text: {r['payload']['transliteration_normalized']}")
```

**3. Verify Qdrant:**
```python
from src.retrieval.qdrant_store import QdrantStore

qdrant = QdrantStore()
count = qdrant.verify_database()
print(f"Total points: {count}")
```

---

## Testing & Debugging

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_normalization.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Integration Tests

```python
# Test complete pipeline
from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.translate("ḥtp dj njswt", show_details=True)

assert result['success'] == True
assert len(result['english']) > 0
assert len(result['german']) > 0
```

### Performance Profiling

```python
import time

# Time translation
start = time.time()
result = pipeline.translate("ḥtp dj njswt")
end = time.time()

print(f"Translation took: {end - start:.2f}s")
```

### Common Issues

**Issue: "OLLAMA_API_KEY not found"**
- Solution: Check `.env` file exists in project root
- Verify `OLLAMA_API_KEY=your_key` is set

**Issue: "Qdrant collection not found"**
- Solution: Run `python setup.py` to rebuild database
- Or: Delete `./qdrant_db/` and re-run setup

**Issue: "BM25 index not found"**
- Solution: Run `python -m src.retrieval.bm25_index`

**Issue: "Embeddings dimension mismatch"**
- Solution: Delete `./data/embeddings/` and regenerate
- Check `VECTOR_DIM` in settings matches model

---

## Deployment

### Production Checklist

- [ ] Environment variables secured (not in git)
- [ ] Dependencies frozen (`pip freeze > requirements.txt`)
- [ ] Database backed up (`./qdrant_db/`)
- [ ] Embeddings pre-computed (`./data/embeddings/`)
- [ ] API rate limits configured
- [ ] Error logging enabled
- [ ] Monitoring setup

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Pre-load models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

CMD ["python", "main.py"]
```

### API Deployment (Optional)

```python
# api.py
from fastapi import FastAPI
from src.pipeline.rag_pipeline import RAGPipeline

app = FastAPI()
pipeline = RAGPipeline()

@app.post("/translate")
def translate(text: str):
    result = pipeline.translate(text, show_details=False)
    return {
        "egyptian": result['query_original'],
        "german": result['german'],
        "english": result['english'],
        "success": result['success']
    }
```

---

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Max line length: 100 characters

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "Add new feature"

# 3. Push to remote
git push origin feature/new-feature

# 4. Create Pull Request on GitHub
```

### Documentation

- Update README.md for user-facing changes
- Update DEVELOPER.md for technical changes
- Add docstrings to new functions
- Include usage examples

---

## Appendix

### Configuration Reference

**All settings in `src/config/settings.py`:**

```python
# API
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = "https://ollama.com/api/chat"

# Models
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Search
TOP_K_RESULTS = 30
HYBRID_SEARCH_ALPHA = 0.5

# Database
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "egyptian_transliterations"

# Data
DATA_RAW_PATH = "./data/raw"
DATA_PROCESSED_PATH = "./data/processed"
DATA_EMBEDDINGS_PATH = "./data/embeddings"

# Training
TRAIN_SPLIT = 0.99
VECTOR_DIM = 1024
BATCH_SIZE = 32

# Egyptian normalization
EGYPTIAN_CHAR_MAP = {...}
SUFFIXES_TO_REMOVE = [...]

# Dataset
DATASET_NAME = "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium"
COLUMNS_TO_DROP = ["hieroglyphs", "dateNotBefore", "dateNotAfter"]
```

### Dependencies

**Core:**
- Python 3.8+
- pandas>=2.2.0
- numpy>=1.26.0
- tqdm>=4.66.0

**ML/NLP:**
- transformers>=4.38.0
- sentence-transformers
- torch>=2.2.0
- datasets>=2.18.0

**Search:**
- qdrant-client>=1.7.0
- rank-bm25>=0.2.2

**Evaluation:**
- nltk==3.9.2
- rouge-score==0.1.2
- sacrebleu==2.6.0

**API:**
- httpx>=0.25.2
- requests
- python-dotenv>=1.0.0

### References

- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [TLA Dataset](https://thesaurus-linguae-aegyptiae.de/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

**Last Updated:** February 2025  
**Version:** 1.0.0  
**Maintainer:** [Your Name]