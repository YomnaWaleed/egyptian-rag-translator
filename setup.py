# setup.py
"""
Setup script to prepare the entire Egyptian RAG Translator system
Smart version: Detects existing files and skips completed steps
"""
import sys
import os
import pandas as pd
from pathlib import Path
from src.config import settings


def check_file_exists(filepath):
    """Check if file exists and is not empty"""
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 0:
            return True
    return False


def check_qdrant_collection():
    """Check if Qdrant collection exists"""
    try:
        from src.retrieval.qdrant_store import QdrantStore

        qdrant_store = QdrantStore(in_memory=False)
        count = qdrant_store.verify_database()
        return count > 0
    except:
        return False


def main():
    print("\n" + "=" * 70)
    print("🏛️ Egyptian RAG Translator - Smart Setup Script")
    print("=" * 70)

    try:
        # Validate configuration
        print("\n✅ Validating configuration...")
        settings.validate_config()

        # Define file paths
        raw_path = f"{settings.DATA_RAW_PATH}/tla_original.csv"
        train_path = f"{settings.DATA_PROCESSED_PATH}/tla_train.csv"
        test_path = f"{settings.DATA_PROCESSED_PATH}/tla_test.csv"
        embeddings_path = f"{settings.DATA_EMBEDDINGS_PATH}/train_embeddings.npy"
        bm25_path = f"{settings.DATA_PROCESSED_PATH}/bm25_corpus.pkl"

        # ================================================================
        # STEP 1: Check if raw data exists
        # ================================================================
        print("\n" + "=" * 70)
        print("📊 STEP 1: Raw Data")
        print("=" * 70)

        if check_file_exists(raw_path):
            print(f"✅ Raw data already exists: {raw_path}")
            df_clean = pd.read_csv(raw_path)
            print(f"   Loaded {len(df_clean)} records")
        else:
            print("📥 Downloading and cleaning dataset...")
            from src.preprocessing.loader import load_tla_dataset, clean_dataset

            df = load_tla_dataset()
            df_clean = clean_dataset(df)
            df_clean.to_csv(raw_path, index=False)
            print(f"💾 Saved to: {raw_path}")

        # ================================================================
        # STEP 2: Check if train/test split exists
        # ================================================================
        print("\n" + "=" * 70)
        print("📊 STEP 2: Train/Test Split")
        print("=" * 70)

        if check_file_exists(train_path) and check_file_exists(test_path):
            print(f"✅ Train/test split already exists")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            print(f"   Train: {len(df_train)} records")
            print(f"   Test:  {len(df_test)} records")
            print(
                "⏭️  Skipping normalization and lemmatization (already in split files)"
            )
        else:
            print("🔄 Creating train/test split...")

            # Normalize
            print("\n   📝 Normalizing transliterations...")
            from src.preprocessing.normalization import normalize_dataset

            df_normalized = normalize_dataset(df_clean)

            # Extract lemmas
            print("\n   📝 Extracting lemmas...")
            from src.preprocessing.lemmatization import extract_lemmas_from_dataset

            df_with_lemmas = extract_lemmas_from_dataset(df_normalized)

            # Split
            print("\n   ✂️  Splitting data...")
            from src.preprocessing.split import split_train_test, save_splits

            df_train, df_test = split_train_test(df_with_lemmas)
            save_splits(df_train, df_test)

        # ================================================================
        # STEP 3: Check if embeddings exist
        # ================================================================
        print("\n" + "=" * 70)
        print("📊 STEP 3: Embeddings")
        print("=" * 70)

        if check_file_exists(embeddings_path):
            print(f"✅ Embeddings already exist: {embeddings_path}")
            import numpy as np

            embeddings = np.load(embeddings_path)
            print(f"   Loaded {len(embeddings)} embeddings (dim={len(embeddings[0])})")

            # Add embeddings to df_train
            if "embedding" not in df_train.columns:
                df_train["embedding"] = list(embeddings)
                print("   ✅ Added embeddings to train dataframe")
        else:
            print("🔢 Generating embeddings (this may take 15-30 minutes)...")
            from src.embeddings.embedder import generate_embeddings_for_dataset

            df_train = generate_embeddings_for_dataset(df_train)

            # Save train with embeddings
            train_with_emb_path = (
                f"{settings.DATA_PROCESSED_PATH}/tla_train_with_embeddings.csv"
            )
            df_train.to_csv(train_with_emb_path, index=False)
            print(f"💾 Saved to: {train_with_emb_path}")

        # ================================================================
        # STEP 4: Check if Qdrant database exists
        # ================================================================
        print("\n" + "=" * 70)
        print("📊 STEP 4: Qdrant Vector Database")
        print("=" * 70)

        if check_qdrant_collection():
            print(f"✅ Qdrant collection already exists")
            print("⏭️  Skipping database build")
        else:
            print("🗄️  Building Qdrant vector database...")
            from src.retrieval.qdrant_store import QdrantStore

            qdrant_store = QdrantStore(in_memory=False)
            qdrant_store.create_collection()
            qdrant_store.upload_data(df_train)
            qdrant_store.verify_database()

        # ================================================================
        # STEP 5: Check if BM25 index exists
        # ================================================================
        print("\n" + "=" * 70)
        print("📊 STEP 5: BM25 Sparse Index")
        print("=" * 70)

        if check_file_exists(bm25_path):
            print(f"✅ BM25 index already exists: {bm25_path}")
            print("⏭️  Skipping BM25 build")
        else:
            print("🔍 Building BM25 index...")
            from src.retrieval.bm25_index import build_bm25_from_dataset

            bm25_index = build_bm25_from_dataset(df_train)

        # ================================================================
        # SUCCESS SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("✅ Setup Complete!")
        print("=" * 70)
        print("\n📊 System Status:")
        print(f"   ✅ Raw data:        {raw_path}")
        print(f"   ✅ Train records:   {len(df_train)}")
        print(f"   ✅ Test records:    {len(df_test)}")
        print(f"   ✅ Embeddings:      {embeddings_path}")
        print(f"   ✅ Vector DB:       {settings.QDRANT_PATH}")
        print(f"   ✅ BM25 index:      {bm25_path}")
        print(f"\n🎯 Embedding dim:    {settings.VECTOR_DIM}")
        print(f"🤖 LLM model:        {settings.LLM_MODEL}")
        print("\n" + "=" * 70)
        print("🚀 You can now run translations!")
        print("=" * 70)
        print('\n   python main.py "<egyptian_text>"')
        print("\n   Example:")
        print('   python main.py "ḥtp dj njswt"')
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
