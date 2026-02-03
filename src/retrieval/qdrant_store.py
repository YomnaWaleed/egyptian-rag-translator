# src/retrieval/qdrant_store.py
"""
Qdrant vector database operations
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm
from src.config import settings


class QdrantStore:
    """Manage Qdrant vector database"""

    def __init__(self, path=None, in_memory=False):
        """
        Initialize Qdrant client

        Args:
            path: Path to persistent storage (None for in-memory)
            in_memory: Use in-memory storage
        """
        if in_memory:
            print("🔧 Initializing Qdrant (in-memory)")
            self.client = QdrantClient(":memory:")
        else:
            path = path or settings.QDRANT_PATH
            print(f"🔧 Initializing Qdrant (persistent: {path})")
            self.client = QdrantClient(path=path)

        self.collection_name = settings.COLLECTION_NAME
        print(f"✅ Qdrant client initialized")

    def create_collection(self):
        """Create vector collection"""
        print("\n" + "=" * 70)
        print("🗄️ Creating Qdrant collection")
        print("=" * 70)

        # Delete if exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"   Deleted existing collection: {self.collection_name}")
        except:
            pass

        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.VECTOR_DIM, distance=Distance.COSINE
            ),
        )

        print(f"✅ Collection created: {self.collection_name}")
        print(f"   Vector size: {settings.VECTOR_DIM}")
        print(f"   Distance metric: COSINE")

    def upload_data(self, df):
        """
        Upload data to Qdrant

        Args:
            df: DataFrame with embeddings and metadata
        """
        print("\n" + "=" * 70)
        print(f"📤 Uploading {len(df)} records to Qdrant")
        print("=" * 70)

        # Prepare points
        points = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing points"):
            # Parse embedding if it's a string
            embedding = row["embedding"]
            if isinstance(embedding, str):
                import ast

                embedding = ast.literal_eval(embedding)

            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "transliteration_original": row["transliteration"],
                    "transliteration_normalized": row["transliteration_normalized"],
                    "lemmas": row["lemmas"] if isinstance(row["lemmas"], list) else [],
                    "UPOS": row.get("UPOS", ""),
                    "glossing": row.get("glossing", ""),
                    "translation_de": row["translation"],
                },
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        print(f"\n📦 Uploading in batches of {batch_size}...")

        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print(f"\n✅ Upload complete!")
        print(f"   Total records in database: {len(points)}")

    def verify_database(self):
        """Verify database contents"""
        print("\n" + "=" * 70)
        print("✅ Verifying database")
        print("=" * 70)

        count_info = self.client.count(collection_name=self.collection_name, exact=True)

        print(f"📊 Collection statistics:")
        print(f"   Name: {self.collection_name}")
        print(f"   Points count: {count_info.count}")

        return count_info.count

    def search(self, query_vector, limit=10):
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return

        Returns:
            list: Search results
        """
        results = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=limit
        ).points

        return results

    def get_client(self):
        """Get Qdrant client instance"""
        return self.client


if __name__ == "__main__":
    import pandas as pd
    from src.config import settings

    # Load training data with embeddings
    df_train = pd.read_csv(f"{settings.DATA_PROCESSED_PATH}/tla_train.csv")

    # Initialize Qdrant
    qdrant_store = QdrantStore(in_memory=False)

    # Create collection
    qdrant_store.create_collection()

    # Upload data
    qdrant_store.upload_data(df_train)

    # Verify
    qdrant_store.verify_database()
