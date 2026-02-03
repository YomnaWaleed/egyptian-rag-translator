# src/ pipeline/rag_pipeline.py
"""
Complete RAG pipeline for Egyptian translation
"""
from src.config import settings
from src.preprocessing.normalization import normalize_transliteration
from src.embeddings.embedder import EmbeddingGenerator
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_search import HybridSearcher
from src.translation.egyptian_to_german import EgyptianToGermanTranslator
from src.translation.german_to_english import GermanToEnglishTranslator


class RAGPipeline:
    """Complete RAG pipeline for Egyptian→German→English translation"""

    def __init__(self, qdrant_path=None, bm25_path=None, in_memory=False):
        """
        Initialize RAG pipeline

        Args:
            qdrant_path: Path to Qdrant database
            bm25_path: Path to BM25 index file
            in_memory: Use in-memory Qdrant
        """
        print("\n" + "=" * 70)
        print("🚀 Initializing RAG Pipeline")
        print("=" * 70)

        # Initialize components
        print("\n1️⃣ Loading embedding model...")
        self.embedder = EmbeddingGenerator()

        print("\n2️⃣ Connecting to Qdrant...")
        self.qdrant_store = QdrantStore(path=qdrant_path, in_memory=in_memory)

        print("\n3️⃣ Loading BM25 index...")
        self.bm25_index = BM25Index()
        bm25_file = bm25_path or "bm25_corpus.pkl"
        self.bm25_index.load_index(bm25_file)

        print("\n4️⃣ Creating hybrid searcher...")
        self.hybrid_searcher = HybridSearcher(self.qdrant_store, self.bm25_index)

        print("\n5️⃣ Initializing translators...")
        self.egy_to_de_translator = EgyptianToGermanTranslator()
        self.de_to_en_translator = GermanToEnglishTranslator()

        print("\n" + "=" * 70)
        print("✅ RAG Pipeline initialized successfully!")
        print("=" * 70)

    def translate(self, query_original, show_details=True):
        """
        Complete translation pipeline: Egyptian → German → English

        Args:
            query_original: Original Egyptian transliteration
            show_details: Print intermediate steps

        Returns:
            dict: Translation results
        """
        if show_details:
            print("\n" + "=" * 70)
            print(f"📝 TRANSLATING: {query_original}")
            print("=" * 70)

        # Step 1: Normalize query
        query_normalized = normalize_transliteration(query_original)

        if show_details:
            print(f"\n1️⃣ Normalization:")
            print(f"   Original:   {query_original}")
            print(f"   Normalized: {query_normalized}")

        # Step 2: Generate embedding
        query_embedding = self.embedder.generate_single(query_normalized)

        if show_details:
            print(f"\n2️⃣ Embedding generated (dim={len(query_embedding)})")

        # Step 3: Hybrid search
        if show_details:
            print(f"\n3️⃣ Hybrid search (Dense + BM25)...")

        search_results = self.hybrid_searcher.search(
            query_text=query_normalized,
            query_embedding=query_embedding,
            top_k=settings.TOP_K_RESULTS,
        )

        if show_details:
            print(f"   ✅ Found {len(search_results)} results")
            print(f"\n   📊 Top 3 matches:")
            for i, result in enumerate(search_results[:3], 1):
                print(f"\n   {i}. RRF Score: {result['rrf_score']:.4f}")
                print(
                    f"      Transliteration: {result['payload']['transliteration_normalized']}"
                )
                print(f"      German: {result['payload']['translation_de'][:50]}...")

        # Step 4: LLM Translation (Egyptian → German)
        if show_details:
            print(f"\n4️⃣ LLM Translation (Egyptian → German)...")

        german_translation, llm_full_output = self.egy_to_de_translator.translate(
            query_original=query_original,
            query_normalized=query_normalized,
            retrieved_examples=search_results,
        )

        if not german_translation:
            return {"success": False, "error": "LLM translation failed"}

        if show_details:
            print(f"   🇩🇪 German: {german_translation}")

        # Step 5: German → English
        if show_details:
            print(f"\n5️⃣ Translation (German → English)...")

        english_translation = self.de_to_en_translator.translate(german_translation)

        if not english_translation:
            return {"success": False, "error": "German→English translation failed"}

        # Final result
        if show_details:
            print("\n" + "=" * 70)
            print("✅ TRANSLATION COMPLETE")
            print("=" * 70)
            print(f"🏛️ Egyptian:  {query_original}")
            print(f"🔤 Normalized: {query_normalized}")
            print(f"🇩🇪 German:    {german_translation}")
            print(f"🇬🇧 English:   {english_translation}")
            print("=" * 70 + "\n")

        return {
            "success": True,
            "query_original": query_original,
            "query_normalized": query_normalized,
            "german": german_translation,
            "english": english_translation,
            "llm_output": llm_full_output,
            "top_matches": search_results[:3],
        }


def create_pipeline(qdrant_path=None, bm25_path=None, in_memory=False):
    """
    Create a RAG pipeline instance

    Args:
        qdrant_path: Path to Qdrant database
        bm25_path: Path to BM25 index
        in_memory: Use in-memory Qdrant

    Returns:
        RAGPipeline: Initialized pipeline
    """
    return RAGPipeline(qdrant_path, bm25_path, in_memory)


if __name__ == "__main__":
    # Test pipeline
    pipeline = create_pipeline(in_memory=False)

    # Test translation
    query = "ḥtp dj njswt"
    result = pipeline.translate(query, show_details=True)

    if result["success"]:
        print("\n✅ Translation successful!")
    else:
        print(f"\n❌ Translation failed: {result.get('error')}")
