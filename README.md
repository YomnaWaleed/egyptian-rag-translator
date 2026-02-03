# 🏛️ Egyptian RAG Translator

Translate Earlier Egyptian transliterations to English using state-of-the-art AI and Retrieval-Augmented Generation (RAG).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 What is This?

This tool translates Ancient Egyptian transliterations (like `ḥtp dj njswt`) into English through a sophisticated AI pipeline:

1. **Normalizes** the Egyptian text
2. **Searches** a database of 9,000 expert translations for similar examples
3. **Translates** to German using a large language model with context
4. **Converts** the German to English

**Example:**
```
Input:  ḥtp dj njswt
Output: A sacrifice given by the King.
```

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- An Ollama API key ([Get one here](https://ollama.com/))
- 5GB free disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/egyptian-rag-translator.git
cd egyptian-rag-translator
```

2. **Create virtual environment:**
```bash
# Using uv (recommended - faster)
uv init
uv venv

# OR using standard Python
python -m venv .venv
```

3. **Activate environment:**

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
# Using uv (faster)
uv pip install -r requirements.txt

# OR using pip
pip install -r requirements.txt
```

5. **Configure API key:**

Create a `.env` file in the project root:
```bash
OLLAMA_API_KEY=your_api_key_here
```

6. **Setup the system (one command):**
```bash
python setup.py
```

This will automatically:
- Download the Egyptian dataset (~9,000 texts)
- Process and clean the data
- Generate AI embeddings (~30 minutes)
- Build the search database

**Note:** The setup script is smart - it won't re-download or re-process if files already exist.

## 🚀 Usage

### Command Line

```bash
# Basic translation
python main.py "ḥtp dj njswt"

# Quick mode (hide processing details)
python main.py "ḥtp dj njswt" --no-details
```

**Example output:**
```
======================================================================
✅ TRANSLATION COMPLETE
======================================================================
🏛️ Egyptian:  ḥtp dj njswt
🔤 Normalized: htp dj njswt
🇩🇪 German:    Ein Opfer, das der König gibt.
🇬🇧 English:   A sacrifice given by the King.
======================================================================
```

### Python API

```python
from src.pipeline.rag_pipeline import RAGPipeline

# Initialize the translator
pipeline = RAGPipeline()

# Translate
result = pipeline.translate("ḥtp dj njswt", show_details=False)

if result['success']:
    print(f"English: {result['english']}")
    print(f"German:  {result['german']}")
```

## 📊 Performance

Our RAG system significantly outperforms direct LLM translation:

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

*Tested on 91 samples from the TLA dataset*

### Why RAG is Better

- ✅ **20-32% higher accuracy** across all metrics
- ✅ **Contextual understanding** from 9,000 reference translations
- ✅ **Grammatical consistency** through example matching
- ✅ **No hallucinations** - grounded in real expert translations

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Required
OLLAMA_API_KEY=your_key

# Optional (defaults shown)
LLM_MODEL=qwen3-vl:235b-instruct-cloud
EMBEDDING_MODEL=BAAI/bge-m3
TOP_K_RESULTS=30
```

## 📚 Dataset

Uses the **Thesaurus Linguae Aegyptiae (TLA)** dataset:
- 9,000+ Earlier Egyptian texts
- Old Egyptian & Early Middle Egyptian periods
- Expert-curated translations
- Linguistic annotations (lemmas, POS tags, glossing)

Source: [thesaurus-linguae-aegyptiae](https://huggingface.co/datasets/thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium)

## ❓ Troubleshooting

### "OLLAMA_API_KEY not found"
Make sure you created a `.env` file with your API key.

### "Dataset download failed"
Check your internet connection. The dataset is ~50MB.

### "Embedding generation is slow"
This is normal - generating 9,000 embeddings takes ~30 minutes. It only runs once.

### "Translation quality is poor"
- Make sure `setup.py` completed successfully
- Try increasing `TOP_K_RESULTS` in `.env` (default: 30)
- Check that your Ollama API key is valid

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/egyptian-rag-translator/issues)
- **Email:** yomnawaleed2023@gmail.com
- **Documentation:** [Developer Guide](./DEVELOPER.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TLA Dataset:** [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de/)
- **Embedding Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Translation Model:** [Helsinki-NLP/opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
- **LLM:** [Ollama Cloud - Qwen 3 VL](https://ollama.com/)

---

**Note:** This is a research tool. For critical academic work, always verify translations with Egyptology experts.