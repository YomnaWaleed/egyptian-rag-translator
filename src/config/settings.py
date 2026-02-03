# src/ config/ settings.py
"""
Configuration settings for Egyptian RAG Translation System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://ollama.com/api/chat")

# Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-vl:235b-instruct-cloud")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Database Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "egyptian_transliterations")

# Search Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "55"))
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

# Data Paths
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "./data/raw")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "./data/processed")
DATA_EMBEDDINGS_PATH = os.getenv("DATA_EMBEDDINGS_PATH", "./data/embeddings")

# Training Configuration
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.99"))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Egyptian character mapping (uniliteral signs)
EGYPTIAN_CHAR_MAP = {
    # Traditional → Normalized
    "ꜣ": "a",  # vulture (aleph)
    "ꞽ": "i",  # reed (yodh)
    "y": "y",  # double yodh
    "ꜥ": "a",  # arm (ayin)
    "w": "w",  # quail
    "b": "b",  # leg
    "p": "p",  # stool
    "f": "f",  # viper
    "m": "m",  # owl
    "n": "n",  # water
    "r": "r",  # mouth
    "h": "h",  # shelter
    "ḥ": "h",  # wick
    "ḫ": "kh",  # placenta
    "ẖ": "kh",  # belly
    "s": "s",  # cloth
    "š": "sh",  # pool
    "ḳ": "q",  # hill
    "q": "q",  # hill
    "k": "k",  # basket
    "g": "g",  # stand
    "t": "t",  # bun
    "ṯ": "tj",  # rope
    "d": "d",  # hand
    "ḏ": "dj",  # cobra
    # Additional special characters
    "ṭ": "t",
    "ḍ": "d",
    "ṣ": "s",
    "ẓ": "z",
}

# Suffixes to remove (pronouns and particles)
SUFFIXES_TO_REMOVE = [
    "=f",  # his/him
    "=k",  # your/you (masc)
    "=ṯ",  # your/you (fem)
    "=s",  # her/it
    "=sn",  # their/them
    "=ꞽ",  # my/me
    "=n",  # our/us
    "=tn",  # your/you (pl)
    "=fꞽ",  # variant
]

# Dataset Configuration
DATASET_NAME = "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium"
COLUMNS_TO_DROP = ["hieroglyphs", "dateNotBefore", "dateNotAfter"]


def validate_config():
    """Validate that required configuration is present"""
    if not OLLAMA_API_KEY:
        raise ValueError("OLLAMA_API_KEY not set in environment variables")

    # Create directories if they don't exist
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    os.makedirs(DATA_EMBEDDINGS_PATH, exist_ok=True)
    os.makedirs(QDRANT_PATH, exist_ok=True)

    print("✅ Configuration validated successfully")
