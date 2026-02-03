# src/retrieval/bm25_index.py
"""
BM25 sparse search index
"""
import pickle
from rank_bm25 import BM25Okapi
from src.config import settings


class BM25Index:
    """BM25 index for sparse search"""

    def __init__(self):
        """Initialize BM25 index"""
        self.bm25 = None
        self.corpus_texts = None

    def build_index(self, texts):
        """
        Build BM25 index from corpus

        Args:
            texts: List of text strings
        """
        print("\n" + "=" * 70)
        print("🔍 Building BM25 index for sparse search")
        print("=" * 70)

        self.corpus_texts = texts
        tokenized_corpus = [text.split() for text in texts]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✅ BM25 index built!")
        print(f"   Documents indexed: {len(tokenized_corpus)}")

    def search(self, query_text, top_k=10):
        """
        Search using BM25

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            tuple: (top_indices, top_scores)
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        query_tokens = query_text.split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np

        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def save_index(self, filename="bm25_corpus.pkl"):
        """
        Save BM25 index to file

        Args:
            filename: Output filename
        """
        output_path = f"{settings.DATA_PROCESSED_PATH}/{filename}"

        data = {"bm25": self.bm25, "corpus_texts": self.corpus_texts}

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"\n💾 BM25 index saved to: {output_path}")

    def load_index(self, filename="bm25_corpus.pkl"):
        """
        Load BM25 index from file

        Args:
            filename: Input filename
        """
        input_path = f"{settings.DATA_PROCESSED_PATH}/{filename}"

        with open(input_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.corpus_texts = data["corpus_texts"]

        print(f"📥 BM25 index loaded from: {input_path}")
        print(f"   Documents indexed: {len(self.corpus_texts)}")


def build_bm25_from_dataset(df, column="transliteration_normalized"):
    """
    Build BM25 index from dataset

    Args:
        df: Input DataFrame
        column: Column to index

    Returns:
        BM25Index: Built index
    """
    bm25_index = BM25Index()

    corpus_texts = df[column].tolist()
    bm25_index.build_index(corpus_texts)

    # Save index
    bm25_index.save_index()

    return bm25_index


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Load training data
    df_train = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_train.csv")

    # Build BM25 index
    bm25_index = build_bm25_from_dataset(df_train)

    # Test search
    query = df_train.iloc[0]["transliteration_normalized"]
    top_indices, top_scores = bm25_index.search(query, top_k=5)

    print(f"\n🔍 Test search for: {query}")
    print(f"\n📋 Top 5 BM25 results:")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(
            f"   {i}. Score: {score:.4f} - {df_train.iloc[idx]['transliteration_normalized']}"
        )
