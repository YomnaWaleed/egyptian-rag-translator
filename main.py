# -*- coding: utf-8 -*-
"""
Main script to run the complete Egyptian RAG Translator pipeline
"""
import argparse
from src.pipeline.rag_pipeline import RAGPipeline
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Translate Earlier Egyptian transliterations to English"
    )
    parser.add_argument("query", type=str, help="Egyptian transliteration to translate")
    parser.add_argument(
        "--no-details", action="store_true", help="Hide detailed processing steps"
    )
    parser.add_argument(
        "--in-memory", action="store_true", help="Use in-memory Qdrant (for testing)"
    )
    parser.add_argument(
        "--qdrant-path", type=str, default=None, help="Path to Qdrant database"
    )
    parser.add_argument(
        "--bm25-path", type=str, default=None, help="Path to BM25 index file"
    )

    args = parser.parse_args()

    # Initialize pipeline
    print("\n🚀 Initializing Egyptian RAG Translator...")
    pipeline = RAGPipeline(
        qdrant_path=args.qdrant_path, bm25_path=args.bm25_path, in_memory=args.in_memory
    )

    # Translate
    result = pipeline.translate(args.query, show_details=not args.no_details)

    # Print result
    if result["success"]:
        if args.no_details:
            print(f"\n{'='*70}")
            print("✅ TRANSLATION RESULT")
            print("=" * 70)
            print(f"🏛️ Egyptian:  {result['query_original']}")
            print(f"🇩🇪 German:    {result['german']}")
            print(f"🇬🇧 English:   {result['english']}")
            print("=" * 70 + "\n")
    else:
        print(f"\n❌ Translation failed: {result.get('error')}")


if __name__ == "__main__":
    main()
