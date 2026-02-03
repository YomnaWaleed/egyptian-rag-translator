#!/usr/bin/env python3
"""
Setup script to prepare the entire Egyptian RAG Translator system
"""
import sys
from src.config import settings


def main():
    print("\n" + "=" * 70)
    print("🏛️ Egyptian RAG Translator - Setup Script")
    print("=" * 70)

    try:
        # Validate configuration
        print("\n1️⃣ Validating configuration...")
        settings.validate_config()

        # Step 1: Load and clean dataset
        print("\n2️⃣ Loading and cleaning dataset...")
        from src.preprocessing.loader import load_tla_dataset, clean_dataset

        df = load_tla_dataset()
        df_clean = clean_dataset(df)

        # Save raw data
        raw_path = f"{settings.DATA_RAW_PATH}/tla_original.csv"
        df_clean.to_csv(raw_path, index=False)
        print(f"💾 Saved to: {raw_path}")

        # Step 2: Normalize
        print("\n3️⃣ Normalizing transliterations...")
        from src.preprocessing.normalization import normalize_dataset

        df_normalized = normalize_dataset(df_clean)

        # Step 3: Extract lemmas
        print("\n4️⃣ Extracting lemmas...")
        from src.preprocessing.lemmatization import extract_lemmas_from_dataset

        df_with_lemmas = extract_lemmas_from_dataset(df_normalized)

        # Step 4: Split train/test
        print("\n5️⃣ Creating train/test split...")
        from src.preprocessing.split import split_train_test, save_splits

        df_train, df_test = split_train_test(df_with_lemmas)
        save_splits(df_train, df_test)

        # Step 5: Generate embeddings
        print("\n6️⃣ Generating embeddings...")
        from src.embeddings.embedder import generate_embeddings_for_dataset

        df_train = generate_embeddings_for_dataset(df_train)

        # Save train with embeddings
        train_path = f"{settings.DATA_PROCESSED_PATH}/tla_train_with_embeddings.csv"
        df_train.to_csv(train_path, index=False)
        print(f"💾 Saved to: {train_path}")

        # Step 6: Build Qdrant database
        print("\n7️⃣ Building Qdrant vector database...")
        from src.retrieval.qdrant_store import QdrantStore

        qdrant_store = QdrantStore(in_memory=False)
        qdrant_store.create_collection()
        qdrant_store.upload_data(df_train)
        qdrant_store.verify_database()

        # Step 7: Build BM25 index
        print("\n8️⃣ Building BM25 index...")
        from src.retrieval.bm25_index import build_bm25_from_dataset

        bm25_index = build_bm25_from_dataset(df_train)

        # Success
        print("\n" + "=" * 70)
        print("✅ Setup Complete!")
        print("=" * 70)
        print("\n📊 Summary:")
        print(f"   Training records: {len(df_train)}")
        print(f"   Test records: {len(df_test)}")
        print(f"   Embeddings dimension: {settings.VECTOR_DIM}")
        print(f"   Vector database: {settings.QDRANT_PATH}")
        print("\n🚀 You can now run translations using:")
        print('   python main.py "<egyptian_text>"')
        print("\n   Example:")
        print('   python main.py "ḥtp dj njswt"')
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
