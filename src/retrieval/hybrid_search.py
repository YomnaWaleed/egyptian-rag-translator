# src/ retrieval/hybrid_search.py
"""
Hybrid search combining dense (vector) and sparse (BM25) retrieval
"""
import numpy as np
from src.config import settings


class HybridSearcher:
    """Hybrid search using Dense + Sparse retrieval"""

    def __init__(self, qdrant_store, bm25_index):
        """
        Initialize hybrid searcher

        Args:
            qdrant_store: QdrantStore instance
            bm25_index: BM25Index instance
        """
        self.qdrant_store = qdrant_store
        self.bm25_index = bm25_index
        self.qdrant_client = qdrant_store.get_client()
        self.collection_name = qdrant_store.collection_name

    def search(self, query_text, query_embedding, top_k=10, alpha=0.5):
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF)

        Args:
            query_text: Normalized transliteration query
            query_embedding: Embedding vector of query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for sparse) - NOT USED in RRF

        Returns:
            List of search results with scores
        """
        # 1. Dense Search (Vector Similarity)
        dense_results = self.qdrant_client.query_points(
            collection_name=self.collection_name, query=query_embedding, limit=top_k * 2
        ).points

        # 2. Sparse Search (BM25)
        query_tokens = query_text.split()
        bm25_scores = self.bm25_index.bm25.get_scores(query_tokens)

        # Get top BM25 indices
        top_bm25_indices = np.argsort(bm25_scores)[-top_k * 2 :][::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        combined_scores = {}

        # Add dense scores
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            rrf_score = 1 / (rank + 60)  # RRF formula
            combined_scores[doc_id] = {
                "rrf_score": rrf_score,
                "dense_score": result.score,
                "sparse_score": 0,
                "payload": result.payload,
            }

        # Add sparse scores
        for rank, idx in enumerate(top_bm25_indices):
            if idx in combined_scores:
                combined_scores[idx]["rrf_score"] += 1 / (rank + 60)
                combined_scores[idx]["sparse_score"] = bm25_scores[idx]
            else:
                # Retrieve payload from Qdrant
                point = self.qdrant_client.retrieve(
                    collection_name=self.collection_name, ids=[int(idx)]
                )
                if point:
                    combined_scores[idx] = {
                        "rrf_score": 1 / (rank + 60),
                        "dense_score": 0,
                        "sparse_score": bm25_scores[idx],
                        "payload": point[0].payload,
                    }

        # 4. Sort by combined RRF score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:top_k]

        # 5. Format results
        final_results = []
        for doc_id, scores in sorted_results:
            final_results.append(
                {
                    "id": doc_id,
                    "rrf_score": scores["rrf_score"],
                    "dense_score": scores["dense_score"],
                    "sparse_score": scores["sparse_score"],
                    "payload": scores["payload"],
                }
            )

        return final_results


def create_hybrid_searcher(qdrant_store, bm25_index):
    """
    Create a hybrid searcher instance

    Args:
        qdrant_store: QdrantStore instance
        bm25_index: BM25Index instance

    Returns:
        HybridSearcher: Configured hybrid searcher
    """
    print("✅ Hybrid search function ready!")
    return HybridSearcher(qdrant_store, bm25_index)


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings
    from src.retrieval.qdrant_store import QdrantStore
    from src.retrieval.bm25_index import BM25Index
    from src.embeddings.embedder import EmbeddingGenerator

    # Load data
    df_train = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_train.csv")

    # Initialize components
    qdrant_store = QdrantStore(in_memory=False)

    bm25_index = BM25Index()
    bm25_index.load_index()

    embedder = EmbeddingGenerator()

    # Create hybrid searcher
    hybrid_searcher = create_hybrid_searcher(qdrant_store, bm25_index)

    # Test search
    query_text = df_train.iloc[0]["transliteration_normalized"]
    query_embedding = embedder.generate_single(query_text)

    results = hybrid_searcher.search(query_text, query_embedding, top_k=5)

    print(f"\n🔍 Test hybrid search for: {query_text}")
    print(f"\n📋 Top 5 results:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. RRF Score: {result['rrf_score']:.4f}")
        print(
            f"      Dense: {result['dense_score']:.4f} | Sparse: {result['sparse_score']:.4f}"
        )
        print(f"      {result['payload']['transliteration_normalized']}")
